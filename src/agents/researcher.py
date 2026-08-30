"""
研究员 Agent (ResearcherAgent)

执行搜索和分析类 SubTask，实现多轮 tool-calling 循环。
设计为项目一 ToolAgentLoop 的简化版：
  - 单 trajectory，无批处理
  - 支持 7 种工具：web_search, arxiv_reader, code_sandbox, browser,
    file_reader, calculator, notepad
  - 通过 VLLMPolicy 进行 LLM 调用
  - 工具结果回写后自动继续，直到模型不再调用工具或达到 max_turns
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from .base_agent import BaseAgent
from ..orchestrator.schemas import SubTask, AgentResult, AgentStatus
from ..utils.tracing import trace_agent


__all__ = ["ResearcherAgent"]


class ResearcherAgent(BaseAgent):
    """研究员 Agent：负责搜索、分析、验证类任务。

    可用工具（7 个）：
      - web_search:   网页搜索，返回标题/链接/摘要
      - browser:      网页阅读器，打开 URL 提取正文
      - arxiv_reader: ArXiv 论文元数据检索
      - file_reader:  本地文件阅读（.txt/.md/.pdf/.csv/.json/.docx）
      - code_sandbox: Python 代码沙箱执行
      - calculator:   轻量数学计算（比沙箱更快更安全）
      - notepad:      草稿笔记（记录中间结论/待办/搜索策略）

    Attributes:
        max_turns: 最大交互轮数，防止无限循环。
        tool_map: 工具名称到工具实例的映射。
    """

    def __init__(
        self,
        name: str,
        policy,
        tools: list | None = None,
        max_turns: int = 15,
    ) -> None:
        super().__init__(name, policy, tools)
        self.max_turns = max_turns
        self.tool_map: dict[str, Any] = {t.name: t for t in (tools or [])}

    @trace_agent(name="researcher.run", tags=["agent", "researcher"])
    async def run(self, task: SubTask, context: dict) -> AgentResult:
        """执行 Researcher 任务。

        流程:
          1. 构建初始 system + user messages
          2. 循环调用 policy，解析 tool_calls
          3. 执行工具，将结果追加为 tool message
          4. 直到无 tool_calls 或达到 max_turns
        """
        trajectory: list[dict] = []
        total_tokens: int = 0

        # 构建任务描述
        task_desc = self._build_task_prompt(task, context)

        # 查询可行性判断：如果任务明显无法通过网络搜索获得答案，直接基于已知信息分析
        if self._is_non_searchable(task, context):
            messages = [
                {"role": "system", "content": self._system_prompt_direct_analysis()},
                {"role": "user", "content": task_desc},
            ]
            try:
                response = self.policy(messages)
                content = response.get("content", "") or ""
                content = self._clean_output(content)
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.SUCCESS,
                    output=content,
                    trajectory=[{"role": "assistant", "content": content}],
                    token_usage=len(content) // 3,
                    confidence=self._extract_confidence(content),
                )
            except Exception as e:
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.FAILED,
                    output=f"Direct analysis failed: {e}",
                    trajectory=[{"error": str(e)}],
                    token_usage=0,
                    confidence=0.0,
                )

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": task_desc},
        ]

        # 若 policy 支持 tool 设置，则注册可用工具
        if hasattr(self.policy, "set_tools") and self.tools:
            schemas = [t.get_openai_tool_schema() for t in self.tools]
            self.policy.set_tools(schemas)

        # 根据任务类型确定 fallback 工具
        desc_lower = (task.description or "").lower()
        academic_keywords = ["论文", "paper", "publication", "学术", "arxiv", "neurips", "icml", "iclr", "scholar", "citation", "文献"]
        fallback_tool = "arxiv_reader" if any(kw in desc_lower for kw in academic_keywords) else "web_search"
        
        for turn in range(self.max_turns):
            # Fallback: if last turn had no tool_calls, force a search instruction
            if turn > 0 and messages and messages[-1].get("role") == "assistant":
                last_tool_calls = messages[-1].get("tool_calls", [])
                if not last_tool_calls:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You did not use any tools. "
                            f"You MUST call the '{fallback_tool}' tool now to search for information. "
                            f"Do not write a summary without searching first."
                        ),
                    })

            try:
                # 使用线程池执行同步 policy，避免阻塞 asyncio 事件循环
                response = await asyncio.to_thread(self.policy, messages)
            except RuntimeError as e:
                # 上下文长度超限等致命错误
                trajectory.append({"turn": turn, "error": str(e)})
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.FAILED,
                    output=str(e),
                    trajectory=trajectory,
                    token_usage=total_tokens,
                    confidence=0.0,
                )

            content = response.get("content", "") or ""
            tool_calls = response.get("tool_calls", []) or []

            trajectory.append({
                "turn": turn,
                "role": "assistant",
                "content": content,
                "tool_calls": [dict(tc) for tc in tool_calls],
            })

            # 估算 token（简化：字符数 / 3）
            total_tokens += len(json.dumps(messages, ensure_ascii=False)) // 3

            # 无工具调用 → 任务完成
            if not tool_calls:
                # 清理输出中的工具失败/API 密钥信息
                cleaned = self._clean_output(content)
                # B方案：检测 LLM 回复是否包含明显的工具失败说明（使用清理后的内容）
                if self._is_tool_failure_explanation(cleaned):
                    return AgentResult(
                        task_id=task.task_id,
                        status=AgentStatus.FAILED,
                        output=cleaned,
                        trajectory=trajectory,
                        token_usage=total_tokens,
                        confidence=0.0,
                    )
                confidence = self._extract_confidence(cleaned)
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.SUCCESS,
                    output=cleaned,
                    trajectory=trajectory,
                    token_usage=total_tokens,
                    confidence=confidence,
                )

            # 执行工具调用
            tool_results = []
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                result = await self._execute_tool(tool_name, args)

                # B方案：工具返回错误 → 将错误告知 LLM 让它尝试其他工具（不立即失败）
                if isinstance(result, dict) and result.get("error"):
                    error_msg = result["error"]
                    trajectory.append({
                        "turn": turn,
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "name": tool_name,
                        "error": error_msg,
                    })
                    # 将错误作为正常工具结果记录，让 LLM 看到并自行决定换工具
                    tool_results.append({
                        "tool_call_id": tc.get("id", ""),
                        "name": tool_name,
                        "result": result,
                    })
                    continue

                tool_results.append({
                    "tool_call_id": tc.get("id", ""),
                    "name": tool_name,
                    "result": result,
                })
                trajectory.append({
                    "turn": turn,
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "name": tool_name,
                    "result": result,
                })

            # 检测是否有工具报错 → 引导 LLM 尝试其他可用工具（如 arxiv_reader、browser）
            tool_errors = [tr for tr in tool_results if isinstance(tr.get("result"), dict) and tr["result"].get("error")]
            if tool_errors:
                error_names = ", ".join(tr["name"] for tr in tool_errors)
                # 列出仍然可用的替代工具
                available_alternatives = []
                for t_name in ["arxiv_reader", "browser", "calculator", "code_sandbox", "notepad", "web_search"]:
                    if t_name in self.tool_map and t_name not in [tr["name"] for tr in tool_errors]:
                        available_alternatives.append(t_name)
                alt_list = ", ".join(f"'{t}'" for t in available_alternatives) if available_alternatives else "none available"

                messages.append({
                    "role": "user",
                    "content": (
                        f"⚠️ CRITICAL: The following tool(s) FAILED: {error_names}. "
                        f"These tools will NOT work no matter how many times you retry. "
                        f"STOP calling them immediately.\n\n"
                        f"ALTERNATIVE tools still available: {alt_list}.\n"
                        f"MUST DO: Call one of the alternative tools NOW to complete your research. "
                        f"For academic/paper topics, use 'arxiv_reader'. "
                        f"For general web content, use 'browser' with a known URL. "
                        f"If no alternative works, you MUST still write a summary based on what you know — "
                        f"do NOT say 'unable to search' in your output."
                    ),
                })

            # 检测搜索结果是否全为空（工具返回了但无有效内容）
            all_empty = True
            for tr in tool_results:
                if tr["name"] == "web_search":
                    res = tr["result"]
                    if isinstance(res, dict) and res.get("results"):
                        for r in res["results"]:
                            if r.get("snippet", "").strip():
                                all_empty = False
                                break
            
            # 如果已搜索 2+ 轮或搜索结果全空，强制要求总结
            search_count = sum(1 for t in trajectory if t.get("role") == "tool" and t.get("name") == "web_search")
            force_summary = False
            if search_count >= 2:
                force_summary = True
            if all_empty and tool_results:
                force_summary = True

            # 将 assistant message 和 tool results 追加到 messages
            assistant_msg = {
                "role": "assistant",
                "content": content,
            }
            if tool_calls:
                assistant_msg["tool_calls"] = [dict(tc) for tc in tool_calls]
            # 保留 reasoning_content（DeepSeek 推理模型需要传回）
            if response.get("reasoning_content"):
                assistant_msg["reasoning_content"] = response["reasoning_content"]
            messages.append(assistant_msg)

            for tr in tool_results:
                msg_content = json.dumps(tr["result"], ensure_ascii=False, default=str)
                # 如果强制总结，给工具结果附加提示
                if force_summary:
                    if all_empty and tool_results:
                        msg_content += (
                            "\n\n[SYSTEM NOTICE] Some tools returned empty results, but you still have information "
                            "from other tools and your own knowledge. Write a substantive summary NOW based on ALL "
                            "available information (both tool results and context). Do NOT say 'unable to find' or "
                            "'no results'. Extract every useful fact and insight from what you have. "
                            "Do NOT call any more tools."
                        )
                    else:
                        msg_content += (
                            "\n\n[SYSTEM NOTICE] You have already searched enough. "
                            "Write your final summary NOW. Do NOT call any more tools."
                        )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tr["tool_call_id"],
                    "content": msg_content,
                })

        # 达到 max_turns
        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.TIMEOUT,
            output="Reached max_turns without final answer.",
            trajectory=trajectory,
            token_usage=total_tokens,
            confidence=0.0,
        )

    def _system_prompt(self) -> str:
        return (
            "You are a meticulous research assistant. "
            "Your job is to gather and analyze information using the RIGHT tool for each task. "
            "\n\nAVAILABLE TOOLS:\n"
            "- web_search: General web search for news, market data, industry reports, current events. "
            "  Note: web_search may be unavailable. If it fails, switch IMMEDIATELY to arxiv_reader or browser.\n"
            "- arxiv_reader: Academic paper search (ArXiv / OpenAlex / Semantic Scholar). "
            "  WORKS WITHOUT API KEY. USE for: papers, publications, academic research, literature reviews.\n"
            "- browser: Open a URL and extract full webpage text. "
            "  WORKS WITHOUT API KEY. USE to read known web pages or follow links from search results.\n"
            "- code_sandbox: Execute Python code for calculations, data processing, simulations.\n"
            "- calculator: Quick math evaluation (+, -, *, /, sqrt, log, mean, etc.).\n"
            "- notepad: Write/read intermediate notes to avoid forgetting findings during multi-step research.\n"
            "- file_reader: Read local files (.txt, .md, .pdf, .csv, .json, .docx).\n"
            "\nIMPORTANT RULES:\n"
            "1. You MUST use a tool to find factual information. Do NOT answer from your own knowledge.\n"
            "2. If one tool fails (returns an error), switch to a DIFFERENT tool immediately. NEVER retry a failed tool.\n"
            "3. For academic/paper topics, PREFER arxiv_reader over web_search — it always works without API keys.\n"
            "4. For any topic, arxiv_reader can find relevant academic papers. Use it as your primary fallback.\n"
            "5. You may call tools AT MOST 3 times total. After that you MUST summarize.\n"
            "6. After gathering information, provide a detailed, comprehensive summary in Chinese (中文).\n"
            "   The summary MUST be at least 500 Chinese characters long. Include:\n"
            "   - Key findings with specific details (numbers, names, dates)\n"
            "   - Relevant sources and references\n"
            "   - Contrasting viewpoints or limitations\n"
            "   - A confidence score (0-1)\n"
            "   ALL your output — summaries, analysis, tool queries — MUST be in Chinese.\n"
            "7. NEVER output 'unable to search', 'API key invalid', or apologies. Always produce a substantive summary.\n"
            "8. NEVER greet the user or ask what they want to search — just execute immediately.\n"
            "9. If you have already performed 3 tool calls, do NOT call more — write the final summary now."
        )

    def _system_prompt_direct_analysis(self) -> str:
        return (
            "You are a thoughtful analyst. "
            "The user has asked a question that cannot be answered by web search "
            "(e.g., analyzing a specific private individual, personal advice, or subjective judgment). "
            "Your job is to provide a reasoned analysis based ONLY on the information already provided in the context. "
            "Do NOT make up facts. Clearly state what is known, what can be reasonably inferred, and what remains unknown. "
            "End with a confidence score (0-1)."
        )

    def _is_non_searchable(self, task: SubTask, context: dict) -> bool:
        """启发式判断任务是否无法通过网络搜索获取答案。"""
        desc = (task.description or "").lower()
        query = context.get("query", "").lower()
        combined = desc + " " + query

        # 模式 1：分析/评价特定私人个体（姓名 + 描述性分析）
        if "朋友" in combined or "同学" in combined or "同事" in combined:
            if any(w in combined for w in ["分析", "评价", "是什么样", "性格", "人品"]):
                return True

        # 模式 2：主观建议类（基于个人情况）
        if any(w in combined for w in ["建议我", "我该怎么", "适合我吗", "要不要"]):
            if "朋友" in combined or "我" in query:
                return True

        # 模式 3：明显的个人隐私分析
        if "叫" in combined and any(w in combined for w in ["分析", "评价", "是什么样"]):
            return True

        return False

    def _build_task_prompt(self, task: SubTask, context: dict) -> str:
        """根据 SubTask 和全局上下文构建 user prompt。"""
        desc_lower = (task.description or "").lower()
        
        # 智能工具推荐：根据任务描述关键词匹配
        tool_recommendations = []
        
        # 学术论文类
        academic_keywords = ["论文", "paper", "publication", "学术", "arxiv", "neurips", "icml", "iclr", "scholar", "citation", "文献"]
        if any(kw in desc_lower for kw in academic_keywords):
            tool_recommendations.append("arxiv_reader")
        
        # 计算/数学类
        calc_keywords = ["计算", "flops", "显存", "内存", "参数量", "延迟", "成本", "公式", "数值", "统计", "数学", "公式", "推导"]
        if any(kw in desc_lower for kw in calc_keywords):
            tool_recommendations.append("calculator")
            tool_recommendations.append("code_sandbox")
        
        # 深度阅读类（需要读原文）
        browser_keywords = ["详细", "原文", "全文", "深度", "详细内容", "网页内容", "文章正文"]
        if any(kw in desc_lower for kw in browser_keywords):
            tool_recommendations.append("browser")
        
        # 文件类
        file_keywords = ["文件", "文档", "dataset", "数据集", "pdf", "csv", "json"]
        if any(kw in desc_lower for kw in file_keywords):
            tool_recommendations.append("file_reader")
        
        # 确定首选工具：学术论文类优先用 arxiv_reader，其他先用 web_search
        is_academic = "arxiv_reader" in tool_recommendations
        if is_academic:
            # 学术论文任务：arxiv_reader 优先，web_search 备选
            tool_recommendations = ["arxiv_reader"] + [t for t in tool_recommendations if t != "arxiv_reader"]
        elif not tool_recommendations:
            tool_recommendations.insert(0, "web_search")
        
        primary_tool = tool_recommendations[0]
        secondary_tools = tool_recommendations[1:]
        
        lines = [
            f"## Task: {task.description}",
            f"Type: {task.task_type.value}",
            f"Expected output: {task.expected_type}",
            "",
            f"## RECOMMENDED TOOLS (in priority order): {', '.join(tool_recommendations)}",
        ]

        if secondary_tools:
            lines.append(f"Start with '{primary_tool}'. If the task involves numbers/calculations, also use {', '.join(secondary_tools)}.")
        else:
            lines.append(f"Use '{primary_tool}' to gather information.")

        # 按任务类型追加差异化指令（区分 search 收集信息 vs analyze 综合分析）
        if task.task_type.value == "search":
            lines.extend([
                "",
                "## 任务目标：信息检索",
                "本任务是【信息检索】任务，目标是收集尽可能全面、准确的事实性信息。",
                "请积极调用搜索工具，多轮检索，覆盖不同角度和数据源。",
            ])
        elif task.task_type.value == "analyze":
            lines.extend([
                "",
                "## 任务目标：综合分析",
                "本任务是【综合分析】任务，目标是对已有信息进行推理、比较、归纳、得出结论。",
                "请优先阅读下方 Context 中的上游任务结果，基于它们进行分析，而非从头重新搜索。",
                "仅在上下文信息明显不足时才补充搜索。",
            ])

        # 根据任务类型定制首条指令
        if task.task_type.value == "analyze":
            first_instruction = (
                f"1. First, READ the Context (upstream task results) below. "
                f"If the information is sufficient, analyze directly. "
                f"Only if insufficient, call '{primary_tool}' to supplement."
            )
        else:
            first_instruction = (
                f"1. First, call the '{primary_tool}' tool with a relevant query to gather information."
            )

        lines.extend([
            "",
            "## INSTRUCTIONS:",
            first_instruction,
            "2. Review the results.",
            f"3. If needed, call '{primary_tool}' ONE MORE time with a refined query.",
            "   You may call tools AT MOST 3 times total. After the 3rd call, you MUST write the final summary.",
            "4. If search results are too short, you may use 'browser' to read the full article (counts as 1 tool call).",
            "5. If calculations are needed, use 'calculator' or 'code_sandbox' (counts as 1 tool call).",
            "6. Finally, provide a DETAILED summary in Chinese (中文) with at least 500 characters. Include specific findings, sources, and a confidence score (0-1).",
            "   The more detail you provide, the better the final report will be.",
            "   ALL output text (summaries, explanations, analysis) MUST be in Chinese.",
            "7. DO NOT greet the user or ask clarifying questions — just execute immediately.",
            "8. IMPORTANT: Your query MUST directly address the task description. Write tool queries in Chinese for Chinese topics.",
        ])
        if task.search_hints:
            lines.insert(1, f"Search hints (MUST use these as primary keywords): {', '.join(task.search_hints)}")

        # 修复：用 dependencies 注入的 dep:{id} 展示依赖结果（而非 context_keys）
        if task.dependencies:
            ctx_parts = []
            for dep_id in task.dependencies:
                dep_key = f"dep:{dep_id}"
                if dep_key in context:
                    dep_text = self._extract_dependency_text(context[dep_key])
                    if dep_text.strip():
                        ctx_parts.append(f"【上游任务 {dep_id} 的结果】\n{dep_text}")
            if ctx_parts:
                lines.append("\n## Context (上游任务结果，请优先基于这些信息进行分析):")
                lines.extend(ctx_parts)
        return "\n".join(lines)

    @staticmethod
    def _extract_dependency_text(value: Any) -> str:
        """从依赖结果中提取文本。兼容 AgentResult / dict / str 三种类型。"""
        if value is None:
            return ""
        # AgentResult 对象
        if hasattr(value, "output"):
            return str(value.output or "")[:800]
        # dict
        if isinstance(value, dict) and "output" in value:
            return str(value["output"] or "")[:800]
        # 字符串或其他
        return str(value)[:800]

    async def _execute_tool(self, tool_name: str, args: dict) -> dict:
        """调用具体工具实例。"""
        tool = self.tool_map.get(tool_name)
        if tool is None:
            return {"error": f"Tool '{tool_name}' not found"}
        try:
            return await tool.execute(**args)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def _is_tool_failure_explanation(self, content: str) -> bool:
        """检测 LLM 回复是否是工具失败的解释说明而非真实研究结果。

        仅当回复内容短且包含失败关键词，或几乎全是失败描述时才返回 True。
        避免误杀同时包含失败描述和实质性结果的长回复。
        """
        if not content:
            return False
        c = content.lower()

        # 实质性内容标记：如果回复包含这些特征，说明有真实研究成果
        substantive_markers = [
            "## ", "### ", "**",  # Markdown 结构
            "http://", "https://", "arxiv.org", "doi.org",  # 引用链接
            "参考文献", "reference", "来源", "source",  # 引用列表
            "根据", "总结", "summary", "findings", "conclusion",  # 总结标记
        ]
        has_substantive = any(m in c for m in substantive_markers)

        # 失败关键词（仅当缺少实质性内容时才算失败）
        failure_keywords = [
            "无法通过", "无法执行", "无法使用", "无法获取", "无法访问",
            "额度已用尽", "配额已用完", "额度已用完", "搜索配额",
            "cannot search", "unable to search", "quota exceeded",
            "api key", "额度不足", "余额不足", "余额为", "余额：0",
            "网络错误", "连接失败", "无法连接到",
        ]
        has_failure = any(kw in c for kw in failure_keywords)

        # 仅当有失败关键词且无实质性内容时，才判定为失败
        if has_failure and not has_substantive:
            return True

        # 即使有实质性内容，如果内容极短（< 100 字符）且全是失败描述，也算失败
        if has_failure and len(content.strip()) < 100:
            return True

        return False

    def _clean_output(self, content: str) -> str:
        """清理 LLM 输出中的工具失败/API 密钥等无关信息。

        移除开头常见的"由于搜索API无效"等引导语，保留实质性研究内容。
        避免这些负面描述传播到 Summarizer，导致其输出道歉信息。
        """
        import re
        if not content:
            return content

        cleaned = content.strip()

        # 模式1: 移除开头以"由于搜索API..."或类似短语开头的段落
        # 匹配 "由于搜索API密钥无效，..." 或 "由于web搜索无法使用，..." 等
        apology_patterns = [
            r'^由于(搜索|web\s*search|API|api).*?(?:密钥|无效|无法|失败|不可用).*?[。\n]',
            r'^因为(搜索|web\s*search|API|api).*?(?:密钥|无效|无法|失败|不可用).*?[。\n]',
            r'^由于.*?(?:API|api)\s*(?:key|密钥).*?(?:无效|不可用|未配置|错误).*?[。\n]',
            r'^(?:搜索)?API.*?(?:密钥|key).*?(?:无效|不可用|未配置).*?[。\n]',
        ]
        for pat in apology_patterns:
            cleaned = re.sub(pat, '', cleaned, count=1, flags=re.IGNORECASE)

        # 模式2: 如果整个输出只有道歉（短且无实质内容），返回空
        if len(cleaned.strip()) < 50:
            # 检查是否只有道歉
            apology_only = [
                "由于搜索", "无法获取", "api密钥", "建议用户",
                "未能获取", "无法完成", "无法进行",
            ]
            if any(kw in cleaned.lower() for kw in apology_only):
                return ""  # 纯道歉，返回空

        return cleaned.strip()

    def _extract_confidence(self, content: str) -> float:
        """从输出文本中尝试提取置信度分数。"""
        import re
        # 匹配 "Confidence: 0.85" 或 "置信度: 0.85"
        patterns = [
            r"[Cc]onfidence[:\s]+(0\.\d+|1\.0|1)",
            r"置信度[:\s]+(0\.\d+|1\.0|1)",
        ]
        for pat in patterns:
            m = re.search(pat, content)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue
        # 默认中等置信度
        return 0.6
