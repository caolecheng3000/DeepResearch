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

from ..utils.execution import call_policy, bounded, check_deadline

import asyncio
from dataclasses import asdict
import json
from typing import Any

from .base_agent import BaseAgent
from ..orchestrator.schemas import SubTask, AgentResult, AgentStatus
from ..utils.tracing import trace_agent, trace_tool
from ..utils.recovery import (ToolExecutor, ToolOutcome, Failure, RecoveryConfig,
                              classify_error, public_message)


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
        recovery_config=None,
    ) -> None:
        super().__init__(name, policy, tools)
        self.max_turns = max_turns
        self.recovery_config = RecoveryConfig.from_dict(recovery_config)
        self.tool_map: dict[str, Any] = {t.name: t for t in (tools or [])}

    @trace_agent(name="researcher.run", tags=["agent", "researcher"])
    async def run(self, task: SubTask, context: dict) -> AgentResult:
        """L1–L3 stay inside this task; only Orchestrator may request L4 Replan."""
        self._executor = ToolExecutor(self.tools, self.recovery_config)
        self.partial_result = None
        trajectory = []
        total_tokens = 0
        usable_evidence = 0
        last_failure = None
        direct = self._is_non_searchable(task, context)
        messages = [
            {"role": "system", "content": self._system_prompt_direct_analysis() if direct else self._system_prompt()},
            {"role": "user", "content": self._build_task_prompt(task, context)},
        ]

        def finish(content, gaps=None):
            # Inspect the original output before cleanup can erase a failure explanation.
            gaps = list(gaps or [])
            if not content.strip() or self._is_tool_failure_explanation(content) or content.lstrip().startswith("Error:"):
                gaps.append("empty_or_error_output")
            upstream = any(getattr(context.get(f"dep:{dep}"), "status", None) == AgentStatus.SUCCESS
                           for dep in task.dependencies)
            if self.tools and not direct and usable_evidence == 0 and not upstream:
                gaps.append("no_usable_evidence")
            requirements = context.get("task_acceptance", {}).get(task.task_id, {})
            for term in requirements.get("required_terms", []):
                if term not in content:
                    gaps.append(f"missing_required_term:{term}")
            if len(content.strip()) < requirements.get("min_chars", 1):
                gaps.append("output_too_short")
            passed = not gaps
            if not content.strip() and usable_evidence:
                recovered = [event["result"] for event in trajectory
                             if event.get("role") == "tool" and event.get("recovery", {}).get("status") in ("success", "partial")]
                content = "部分结果，任务尚未通过验收。缺口：" + ", ".join(gaps)
                content += "\n" + json.dumps(recovered, ensure_ascii=False, default=str)[:6000]
            cleaned = content.strip()  # Preserve limitations and partial evidence; do not hide failure text.
            status = AgentStatus.SUCCESS if passed else (
                AgentStatus.PARTIAL if cleaned and (usable_evidence or upstream) else AgentStatus.FAILED)
            return AgentResult(
                task_id=task.task_id, status=status, output=cleaned,
                trajectory=trajectory, token_usage=total_tokens,
                confidence=self._extract_confidence(cleaned) if passed else 0.0,
                acceptance={"passed": passed, "gaps": list(dict.fromkeys(gaps)),
                            "usable_tool_results": usable_evidence, "scope": "structural_and_evidence_availability_not_truth_verification"},
                recovery={"level": 0 if passed else 4, "needs_replan": not passed,
                          "last_error": asdict(last_failure) if last_failure else None,
                          "tool_calls": self._executor.calls, "repairs": self._executor.repairs,
                          "fallbacks": self._executor.fallbacks, "history": self._executor.history},
            )

        for turn in range(1 if direct else self.max_turns):
            if hasattr(self.policy, "set_tools"):
                self.policy.set_tools(None if direct else [self.tool_map[name].get_openai_tool_schema()
                    for name in self._executor.available() if hasattr(self.tool_map[name], "get_openai_tool_schema")])
            try:
                response = await call_policy(self.policy, messages)
            except asyncio.TimeoutError:
                raise
            except Exception as exc:
                last_failure = classify_error(exc, model=True)
                trajectory.append({"turn": turn, "error": public_message(exc)})
                return finish("", [last_failure.code])
            content = response.get("content", "") or ""
            tool_calls = []
            for index, tc in enumerate(response.get("tool_calls") or []):
                function = dict(tc["function"])
                if isinstance(function.get("arguments"), dict):
                    function["arguments"] = json.dumps(function["arguments"], ensure_ascii=False)
                tool_calls.append({**tc, "id": tc.get("id") or f"turn_{turn}_call_{index}", "function": function})
            trajectory.append({"turn": turn, "role": "assistant", "content": content,
                               "tool_calls": [dict(tc) for tc in tool_calls]})
            total_tokens += len(json.dumps(messages, ensure_ascii=False, default=str)) // 3
            if not tool_calls:
                return finish(content)
            # Preserve protocol ordering: assistant(call) -> ALL tool results -> feedback.
            assistant = {"role": "assistant", "content": content, "tool_calls": [dict(tc) for tc in tool_calls]}
            if response.get("reasoning_content"):
                assistant["reasoning_content"] = response["reasoning_content"]
            messages.append(assistant)
            feedback = []
            terminal = None
            for call_index, tc in enumerate(tool_calls):
                func = tc.get("function") or {}
                name = func.get("name", "")
                call_id = tc.get("id") or f"turn_{turn}_call_{call_index}"
                if terminal:
                    outcome = ToolOutcome("error", failure=Failure(4, "BATCH_STOPPED", terminal, "task"), tool_name=name)
                else:
                    outcome = await self._execute_tool_outcome(name, func.get("arguments", "{}"), call_id)
                if outcome.status in ("success", "partial") and name in (
                        "web_search", "arxiv_reader", "browser", "file_reader", "calculator", "code_sandbox"):
                    usable_evidence += 1
                event = {"turn": turn, "role": "tool", "tool_call_id": call_id, "name": name,
                         "result": outcome.data if outcome.status != "error" else {"error": asdict(outcome.failure)},
                         "recovery": outcome.to_dict()}
                if outcome.failure:
                    last_failure = outcome.failure
                    event["error"] = outcome.failure.message
                    if outcome.failure.level == 1:
                        if self._executor.repair_feedback():
                            feedback.append(f"L1: {name}: {outcome.failure.message}. Correct the call arguments; do not repeat invalid input.")
                        else:
                            terminal = "argument_repair_budget_exhausted"
                    elif outcome.failure.level == 4:
                        terminal = outcome.failure.code
                    else:
                        alternatives = self._recovery_alternatives(name)
                        feedback.append(f"L3: {name}: {outcome.failure.message}. Equivalent capabilities available: {alternatives}. "
                                        "Use only a relevant alternative or finish with explicit evidence gaps. Never invent evidence.")
                elif outcome.status == "empty":
                    feedback.append(f"{name} returned no matching content. Change the query or source; do not repeat the same empty request.")
                trajectory.append(event)
                messages.append({"role": "tool", "tool_call_id": call_id,
                                 "content": json.dumps(outcome.to_dict(), ensure_ascii=False, default=str)})
                if usable_evidence:
                    self.partial_result = finish("", ["interrupted_before_completion"])
            if terminal:
                return finish("", [terminal])
            if feedback:
                messages.append({"role": "user", "content": "\n".join(feedback)})
        return finish("", ["max_turns_exhausted"])

    def _recovery_alternatives(self, failed_name):
        families = ({"web_search", "arxiv_reader", "browser", "file_reader"},
                    {"calculator", "code_sandbox"}, {"notepad"})
        family = next((group for group in families if failed_name in group), {failed_name})
        return [name for name in self._executor.available() if name != failed_name and name in family]

    @trace_tool(name="researcher.tool", tags=["tool"])
    async def _execute_tool_outcome(self, tool_name, arguments, call_id=None):
        return await self._executor.run(tool_name, arguments, call_id)

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
            "1. Search tasks must collect evidence with registered tools. Analyze tasks may use successful upstream evidence directly. Do NOT invent evidence from your own knowledge.\n"
            "2. The runtime retries temporary errors. Follow structured recovery feedback; correct invalid arguments or choose a relevant available alternative.\n"
            "3. For academic topics, prefer arxiv_reader when registered and available. No tool is guaranteed to succeed.\n"
            "4. Alternatives must match the missing capability: academic papers cannot replace current event evidence; browser requires a known URL.\n"
            "5. Work within the runtime tool and turn budgets. State evidence gaps explicitly if recovery fails.\n"
            "6. After gathering information, provide a detailed, comprehensive summary in Chinese (中文).\n"
            "   The summary MUST be at least 500 Chinese characters long. Include:\n"
            "   - Key findings with specific details (numbers, names, dates)\n"
            "   - Relevant sources and references\n"
            "   - Contrasting viewpoints or limitations\n"
            "   - A confidence score (0-1)\n"
            "   ALL your output — summaries, analysis, tool queries — MUST be in Chinese.\n"
            "7. Never invent missing facts. Distinguish supported findings from unverified conclusions and missing evidence.\n"
            "8. NEVER greet the user or ask what they want to search — just execute immediately.\n"
            "9. Finish when the task has sufficient evidence; do not repeat empty queries."
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
            "   Respect runtime budgets. If evidence remains unavailable, state the gap explicitly.",
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

    @trace_tool(name="researcher.tool")
    async def _execute_tool(self, tool_name: str, args: dict) -> dict:
        """Compatibility entry point; regular loop consumes the full ToolOutcome."""
        executor = getattr(self, "_executor", None)
        if executor is None:
            executor = self._executor = ToolExecutor(self.tools, self.recovery_config)
        outcome = await executor.run(tool_name, args)
        return outcome.data if not outcome.failure else {"error": asdict(outcome.failure)}

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
