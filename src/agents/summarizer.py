"""
合成 Agent (SummarizerAgent)

将多个 SubTask 的执行结果合成为结构化的研究报告。
区别于 ResearcherAgent 的多轮 tool-calling，Summarizer 是单轮长上下文生成任务：
  - 把所有子结果按置信度排序后拼接为上下文
  - 调用 LLM 一次性生成 Markdown 格式报告
  - 提取引用来源，计算整体置信度
"""
from __future__ import annotations

from ..utils.execution import call_policy, bounded, check_deadline

import json
import re
from typing import Any

from .base_agent import BaseAgent
from ..orchestrator.schemas import SubTask, AgentResult, AgentStatus, ResearchReport
from ..utils.tracing import trace_agent


__all__ = ["SummarizerAgent"]


class SummarizerAgent(BaseAgent):
    """合成 Agent：将子任务结果合成为最终研究报告。

    Attributes:
        max_output_tokens: 报告生成的最大 token 数（通过 policy.max_tokens 控制）。
    """

    def __init__(self, name: str, policy, tools: list | None = None, min_report_chars: int = 5000) -> None:
        super().__init__(name, policy, tools)
        if min_report_chars < 0:
            raise ValueError("min_report_chars must be nonnegative")
        self.min_report_chars = min_report_chars

    @trace_agent(name="summarizer.run", tags=["agent", "summarizer"])
    async def run(self, task: SubTask, context: dict) -> AgentResult:
        """执行合成任务。

        Args:
            task: 通常是一个特殊的 "synthesize" 类型任务。
            context: 全局上下文，必须包含 "results" 和 "query" 键。
                results: list[AgentResult]
                query: str 原始研究问题

        Returns:
            AgentResult，output 字段为 ResearchReport 实例。
        """
        query = context.get("query", "")
        self._used_fallback = False
        results: list[AgentResult] = context.get("results", [])

        if not results:
            report = ResearchReport(
                query=query,
                content="No sub-task results available to synthesize.",
                confidence=0.0,
                status="failed",
            )
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                output=report,
                trajectory=[],
                token_usage=0,
                confidence=0.0,
            )

        # 预先计算 success_results（用于重试和降级）
        sorted_all = sorted(results, key=lambda r: r.confidence, reverse=True)
        success_results = [r for r in sorted_all if r.status in (AgentStatus.SUCCESS, AgentStatus.PARTIAL) and r.output]

        # 构建 synthesis prompt
        prompt = self._build_synthesis_prompt(query, results, success_results)
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": prompt},
        ]

        content = ""
        try:
            # 合成任务不需要工具调用，临时禁用 tools 避免模型进入 tool-calling 模式
            old_tools = getattr(self.policy, "tools", None)
            self.policy.tools = None
            try:
                response = await call_policy(self.policy, messages)
            finally:
                self.policy.tools = old_tools
            content = response.get("content", "") or ""
        except TimeoutError:
            raise
        except RuntimeError as e:
            # LLM 调用失败 → 直接用降级报告
            content = self._build_fallback_report(query, success_results)
        except Exception as e:
            content = self._build_fallback_report(query, success_results)

        # 后处理：检测输出是否为道歉/API 错误描述/过短内容
        # 若是则用更强 prompt 重新生成，再失败则用降级报告
        if self._is_apology_output(content):
            if success_results:
                content = await self._retry_with_stronger_prompt(query, success_results)
            else:
                content = self._build_fallback_report(query, success_results)

        # 最终兜底：如果此时内容仍然是道歉/过短，强制构建降级报告
        if self._is_apology_output(content):
            content = self._build_fallback_report(query, success_results)

        token_usage = len(content) // 3  # 简化估算

        # 解析报告内容，提取来源和置信度
        report = self._parse_report(query, content, results)
        if not success_results:
            report.status = "failed"
        elif self._used_fallback or any(r.status != AgentStatus.SUCCESS for r in results):
            report.status = "partial"
        gaps = [f"{r.task_id}: {', '.join(r.acceptance.get('gaps', [])) or r.status.value}"
                for r in results if r.status != AgentStatus.SUCCESS]
        if gaps:
            report.content += "\n\n## 证据缺口与限制\n\n" + "\n".join(f"- {gap}" for gap in gaps)

        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.SUCCESS,
            output=report,
            trajectory=[{"role": "assistant", "content": content}],
            token_usage=token_usage,
            confidence=report.confidence,
        )

    def _system_prompt(self) -> str:
        return (
            "You are an expert research synthesizer. "
            "Your task is to integrate multiple research findings into a coherent, well-structured report. "
            "Use Markdown formatting. Cite sources explicitly. "
            f"The report body should contain at least {self.min_report_chars} Chinese characters; follow the user's requested scope. "
            "Write in depth: include background, key findings, detailed analysis, comparisons, and implications. "
            "Every section must have at least 2-3 substantial paragraphs. "
            "DO NOT describe what you will do — directly output the synthesized report. "
            "At the end, provide a '## 参考来源' section with numbered Markdown links: [Title](URL). "
            "Each reference must be on its own line: 'N. [Title](URL) — brief description'. "
            "Also provide an overall confidence score (0-1).\n\n"
            "🌐 LANGUAGE REQUIREMENT — CRITICAL:\n"
            "The ENTIRE report MUST be written in Chinese (中文). "
            "All section titles, analysis, and conclusions must be in Chinese. "
            "Only preserve original English paper titles and author names in references.\n\n"
            "⚠️ CRITICAL RULES — VIOLATING THESE IS UNACCEPTABLE:\n"
            "1. NEVER output an apology or disclaimer (e.g., 'I apologize', '抱歉', '由于API', '密钥无效'). "
            "Your job is to synthesize available findings, not to apologize for missing data.\n"
            "2. NEVER mention 'API', 'API key', '搜索工具', 'search tool failure', or '无法搜索' in your report. "
            "These are system-level issues that the reader does NOT need to know about.\n"
            "3. If tasks failed or returned partial evidence, state the uncovered questions and evidence limitations. "
            "Partial findings have not passed task acceptance; do not present them as verified conclusions.\n"
            "4. Only include conclusions supported by the provided evidence. Never fill gaps to meet a length target.\n"
            "5. Output directly — do NOT include meta-commentary about the synthesis process.\n"
            "6. REQUIRED STRUCTURE: Your report MUST contain at least 4 ## level-2 sections "
            "(执行摘要, 背景, 关键发现, 结论与展望), "
            "each with 200+ characters of detailed analysis."
        )

    def _build_synthesis_prompt(self, query: str, results: list[AgentResult],
                                 success_results: list | None = None) -> str:
        """构建合成 prompt，按置信度降序排列结果。

        关键改进：分离成功/失败结果，重点展示成功结果，
        仅简短提及失败任务（避免 LLM 聚焦于失败信息）。
        """
        sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)
        if success_results is None:
            success_results = [r for r in sorted_results if r.status in (AgentStatus.SUCCESS, AgentStatus.PARTIAL) and r.output]
        failed_results = [r for r in sorted_results if r.status != AgentStatus.SUCCESS]

        parts = [
            f"# Research Question\n{query}\n",
        ]

        # 优先展示成功结果（让 LLM 聚焦于此）
        if success_results:
            parts.append(f"# Available Research Findings ({len(success_results)} tasks)\n")
            parts.append("⚠️ CRITICAL: Use ONLY the findings below to write the report. "
                         "IGNORE any error messages you see in the output — they are tool-level errors, "
                         "not research conclusions. The actual research data is embedded in the results.\n")
            for i, r in enumerate(success_results, 1):
                parts.append(
                    f"## 子任务结果 {i} (置信度: {r.confidence:.2f})\n"
                    f"任务: {r.task_id}，状态: {r.status.value}，验收缺口: {r.acceptance.get('gaps', [])}\n"
                    f"输出:\n{r.output}\n"
                )
        else:
            parts.append("# ⚠️ No successful research findings available\n")

        # 仅简短提及失败任务（不展示详细信息，避免 LLM 输出道歉信息）
        if failed_results:
            failed_task_ids = [r.task_id for r in failed_results]
            parts.append(
                f"\n# Note: {len(failed_results)} task(s) did not complete successfully "
                f"({', '.join(failed_task_ids)}). "
                f"State uncovered questions and evidence limitations; partial results are unverified. "
                f"Do NOT apologize or mention API errors in the report.\n"
            )

        parts.append(
            "\n# Instructions\n"
            "1. Directly write the synthesized report in Chinese (中文). Do NOT say 'I will synthesize'.\n"
            f"2. The report should cover the requested scope (at least {self.min_report_chars} Chinese characters).\n"
            "3. REQUIRED SECTIONS (all titles and content in Chinese, with ## headers, 200+ chars each):\n"
            "   - 执行摘要 (overview of all findings)\n"
            "   - 背景 (context and significance of the topic)\n"
            "   - 关键发现 (3+ subsections with detailed analysis)\n"
            "   - 对比分析 (compare different viewpoints/methods)\n"
            "   - 展望与启示 (future trends and practical impact)\n"
            "   - 结论 (synthesize and summarize)\n"
            "4. Resolve any contradictions between sources. When sources disagree, note the disagreement explicitly.\n"
            "5. Explicitly list all sources cited at the end with title '## 参考来源'.\n"
            "   EVERY reference MUST use Markdown link format: NUMBER. [Title](URL) — brief description\n"
            "   Example: 1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 提出Transformer架构\n"
            "   Do NOT use plain text lists without Markdown links. Do NOT use '关键来源列表' as section title.\n"
            "6. End with: 整体置信度: X.XX\n"
            "7. Explicitly state evidence gaps and unresolved questions. Never invent findings to complete the report.\n"
            "8. If the outputs contain research content mixed with error messages, extract ONLY the research content and ignore the errors.\n"
            "9. LANGUAGE: The ENTIRE report MUST be in Chinese. Only paper titles in references may remain in English."
        )
        return "\n".join(parts)

    def _parse_report(self, query: str, content: str, results: list[AgentResult]) -> ResearchReport:
        """从 LLM 输出中解析 ResearchReport，并基于子任务成功率校准置信度。"""
        # 1. 从文本中提取 LLM 自评置信度
        llm_confidence = 0.5
        m = re.search(r"[Oo]verall\s+[Cc]onfidence[:\s]+(0\.\d+|1\.0|1)", content)
        if m:
            try:
                llm_confidence = float(m.group(1))
            except ValueError:
                pass

        # 2. 基于子任务成功率计算客观置信度
        total = len(results)
        success = sum(1 for r in results if r.status == AgentStatus.SUCCESS)
        success_rate = success / max(total, 1)

        # 3. 综合置信度 = LLM 自评 × 成功率开根（降低成功率的影响权重）
        confidence = llm_confidence * (success_rate ** 0.5)
        confidence = round(max(0.0, min(1.0, confidence)), 2)

        # 收集来源（从各个子结果的轨迹中提取）
        sources: list[dict] = []
        for r in results:
            if r.status not in (AgentStatus.SUCCESS, AgentStatus.PARTIAL):
                continue
            # 简单启发式：从 trajectory 的 tool 结果中提取 url
            for step in r.trajectory:
                if step.get("role") == "tool" and isinstance(step.get("result"), dict):
                    res = step["result"]
                    if "results" in res and isinstance(res["results"], list):
                        for item in res["results"]:
                            if isinstance(item, dict) and "url" in item:
                                sources.append({
                                    "url": item["url"],
                                    "title": item.get("title", ""),
                                    "snippet": item.get("snippet", ""),
                                    "task_id": r.task_id,
                                })
                    elif "papers" in res and isinstance(res["papers"], list):
                        for paper in res["papers"]:
                            if isinstance(paper, dict) and "pdf_url" in paper:
                                sources.append({
                                    "url": paper["pdf_url"],
                                    "title": paper.get("title", ""),
                                    "snippet": paper.get("summary", "")[:200],
                                    "task_id": r.task_id,
                                })

        # 去重
        seen = set()
        unique_sources = []
        for s in sources:
            key = s["url"]
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        # 统计实际工具调用次数（遍历所有子任务的 trajectory）
        num_searches = sum(
            len([t for t in r.trajectory if t.get("role") == "tool"])
            for r in results
        )

        return ResearchReport(
            query=query,
            content=content,
            sources=unique_sources,
            confidence=confidence,
            num_searches=num_searches,
        )

    def _is_apology_output(self, content: str) -> bool:
        """检测输出是否为道歉/API 错误描述/过短内容而非真正的研究报告。

        判断标准：
        1. 内容极短（< 200 字符）且含道歉关键词
        2. 内容开头就是 API/搜索错误描述
        3. 内容中实质性研究标记极少
        4. 内容过短（< 500 字符且缺乏报告结构）—— 也触发重试
        """
        if not content or len(content.strip()) < 50:
            return True

        stripped_len = len(content.strip())
        c = content.strip().lower()

        # 道歉/API 错误关键词
        apology_keywords = [
            "由于搜索api", "由于api", "api密钥无效", "api密钥",
            "搜索api", "博查ai", "密钥无效",
            "无法获取任何搜索", "未能获取任何",
            "建议用户检查", "检查搜索工具配置",
            "无法对.*进行实质性分析",
        ]
        import re
        has_apology = any(re.search(kw, c) for kw in apology_keywords)

        # 实质性研究报告标记（需要至少匹配 2 个才算有报告结构）
        # 注意：移除太常见的词如 "背景"、"展望"，避免误判
        report_markers = [
            "执行摘要", "executive summary",
            "关键发现", "key findings",
            "## ", "### ",  # Markdown 标题（需要 2+ 个才说明有结构）
            "结论", "conclusion",
            "implications",
        ]
        # 对于 Markdown 标题，要求至少 2 个 ## 标题才判定为有结构
        heading_count = c.count("## ")
        has_report_structure = (
            heading_count >= 2 or
            any(m in c for m in report_markers if m not in ("## ", "### "))
        )

        # 如果开头就是道歉且没有报告结构 → 判定为道歉输出
        if has_apology and not has_report_structure:
            return True
        if has_apology and stripped_len < 300:
            return True

        # 过短且缺乏报告结构 → 也需要重试（提升阈值匹配 5000 字要求）
        if stripped_len < 800 and not has_report_structure:
            return True

        return False

    async def _retry_with_stronger_prompt(self, query: str, success_results: list) -> str:
        """用更强硬的 prompt 重新生成报告。

        移除所有可能暗示失败的内容，仅传入净化的成功结果。
        """
        # 净化结果：移除可能的工具错误痕迹
        clean_parts = ["# Research Question\n" + query + "\n"]
        clean_parts.append("# Available Research Data (YOU MUST USE THIS)\n")

        for i, r in enumerate(success_results, 1):
            output = str(r.output) if r.output else ""
            # 进一步清理：移除常见的工具失败前缀
            clean_parts.append(f"## Data {i} ({r.status.value}; gaps={r.acceptance.get('gaps', [])})\n{output}\n")

        clean_parts.append(
            "\n# YOUR TASK\n"
            "Write a COMPLETE, LONG research report using ONLY the data above. "
            "Rules:\n"
            "1. Output the report DIRECTLY — no meta-commentary, no disclaimers, no preambles.\n"
            f"2. Cover the requested scope with at least {self.min_report_chars} Chinese characters.\n"
            "3. Required sections (use ## headers, 200+ chars each): Executive Summary, Background, "
            "Key Findings (3+ subsections), Comparative Analysis, Implications & Outlook, Conclusion.\n"
            "4. Write DETAILED analysis in each section — at least 300 chars per section.\n"
            "5. DO NOT mention 'API', 'search key', 'tool failure', or any system errors.\n"
            "6. End with '## 参考来源' section using Markdown links: N. [Title](URL) — brief description\n"
            "7. End the report with: 整体置信度: X.XX\n"
            "8. State evidence gaps. Do not invent content to satisfy the length target. Partial data is unverified.\n"
            "9. Start writing the report NOW. First line must be a ## heading."
        )
        prompt = "\n".join(clean_parts)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional research report writer. "
                    "You MUST write a complete report based on the provided data. "
                    "NEVER apologize. NEVER mention API errors or search failures. "
                    "Output the report DIRECTLY with no preamble."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            old_tools = getattr(self.policy, "tools", None)
            self.policy.tools = None
            try:
                response = await call_policy(self.policy, messages)
            finally:
                self.policy.tools = old_tools
            content = response.get("content", "") or ""

            # 如果重试后仍然是道歉，强制构造降级报告
            if self._is_apology_output(content):
                return self._build_fallback_report(query, success_results)
            return content
        except TimeoutError:
            raise
        except Exception:
            return self._build_fallback_report(query, success_results)

    def _sanitize_output(self, output: str) -> str:
        """清理任务输出，移除工具失败痕迹。"""
        import re
        # 移除开头的"由于搜索API..."等句子
        output = re.sub(
            r'^由于(?:搜索|web\s*search|API|api|博查).*?(?:密钥|无效|无法|失败|不可用|未配置).*?[。\n]',
            '', output.strip(), count=1, flags=re.IGNORECASE
        )
        # 移除 "搜索API密钥无效" 等短句
        output = re.sub(
            r'(?:搜索|web\s*search)?API.*?(?:密钥|key).*?(?:无效|不可用|未配置|错误|失败).*?[。\n]',
            '', output, count=1, flags=re.IGNORECASE
        )
        return output.strip()

    def _build_fallback_report(self, query: str, success_results: list) -> str:
        """最后手段：手工拼接成功结果构建降级报告。

        输出格式与正常报告一致（中文标题、编号引用来源），而非 "Finding" 格式。
        """
        self._used_fallback = True
        parts = [f"# 研究报告：{query}\n\n"]
        parts.append("## 执行摘要\n\n")
        parts.append("本报告基于多个来源的研究发现综合而成。")
        parts.append("以下为各子任务的研究成果汇总。\n\n")

        parts.append("## 关键发现\n\n")
        for i, r in enumerate(success_results, 1):
            output = str(r.output)[:1000] if r.output else ""
            if output:
                parts.append(f"### {i}. {r.task_id}（{r.status.value}）\n\n")
                parts.append(f"{output}\n\n")

        # 从 trajectory 提取真实引用来源
        refs = self._extract_refs_from_results(success_results)
        if refs:
            parts.append("## 参考来源\n\n")
            for j, ref in enumerate(refs, 1):
                url = ref.get("url", "")
                title = ref.get("title", "未知来源")
                snippet = ref.get("snippet", "")[:200]
                if url:
                    parts.append(f"{j}. [{title}]({url}) — {snippet}\n")
                else:
                    parts.append(f"{j}. {title} — {snippet}\n")
            parts.append("\n")

        parts.append("## 结论\n\n")
        parts.append("以上仅保留已有结果，未完成完整报告生成与最终核验；未验收的部分结果不能视为确定结论。\n\n")
        parts.append("整体置信度: 0.00")

        return "".join(parts)

    @staticmethod
    def _extract_refs_from_results(results: list) -> list[dict]:
        """从 AgentResult 列表中提取引用来源（URL + 标题 + 摘要）。"""
        refs: list[dict] = []
        seen_urls: set[str] = set()
        for r in results:
            for step in getattr(r, "trajectory", []) or []:
                if not isinstance(step, dict):
                    continue
                tool_result = step.get("result")
                if not isinstance(tool_result, dict):
                    continue
                # web_search / browser 结果
                for item in tool_result.get("results", []):
                    if isinstance(item, dict):
                        url = item.get("url", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            refs.append({
                                "url": url,
                                "title": item.get("title", ""),
                                "snippet": item.get("snippet", ""),
                            })
                # arxiv_reader 结果
                for paper in tool_result.get("papers", []):
                    if isinstance(paper, dict):
                        url = paper.get("pdf_url", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            refs.append({
                                "url": url,
                                "title": paper.get("title", ""),
                                "snippet": paper.get("summary", "")[:200],
                            })
        return refs
