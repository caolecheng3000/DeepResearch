"""
自适应规划器 (Adaptive Planner)

LLM 驱动的规划器，负责将研究问题分解为结构化的子任务 DAG。
核心能力:
  - 初始规划: 生成 3-8 个子问题的 DAG
  - 增量重规划: 保留通过验收且定义未变更的结果，重做失败任务及失效后代
  - 健壮性 JSON 解析: 支持 markdown 代码块、多余换行等噪声
"""
from __future__ import annotations

from ..utils.execution import call_policy, bounded, check_deadline

import asyncio
import json
import logging
import re
from typing import Any

from .dag import DAG, DAGCycleError
from .budget_tracker import BudgetTracker
from ..orchestrator.schemas import SubTask, TaskType, AgentResult
from ..utils.tracing import trace_chain, trace_tool, record_event
from ..utils.recovery import current_recovery


__all__ = ["Planner", "PlanParseError"]

logger = logging.getLogger("planner")

# 规划前侦察阶段允许注册的工具白名单（只放搜索类，绝不放开 browser / file_reader 等）
PLANNER_SEARCH_TOOL_WHITELIST = ("web_search", "arxiv_reader")

# 单次工具结果进入上下文的字符上限（防止搜索结果撑爆推理模型的上下文）
PLANNER_TOOL_RESULT_MAX_CHARS = 1200

# 侦察摘要（digest）总字符上限
PLANNER_SEARCH_DIGEST_MAX_CHARS = 2500

# vllm_policy 的正则回退分支不看 self.tools，最后一轮必须自己剥掉工具调用标记。
# 用拼接构造，避免该字面量在源码/日志里被误识别成真实的调用边界。
_TOOL_CALL_OPEN = "<" + "tool_call" + ">"
_TOOL_CALL_CLOSE = "</" + "tool_call" + ">"


class PlanParseError(Exception):
    """规划结果解析失败时抛出。"""
    pass


# ============================================================================
# Prompt 常量
# ============================================================================

INITIAL_PLAN_PROMPT = """\
You are an expert research planner. Your task is to decompose a complex research question into a directed acyclic graph (DAG) of sub-tasks.

## Input
Research Question: {query}

## Output Format
Return a JSON object with this exact structure (no markdown, no extra text):
{{
  "sub_tasks": [
    {{
      "task_id": "task_1",
      "task_type": "search",
      "description": "What is ...",
      "dependencies": [],
      "context_keys": [],
      "timeout_seconds": 120,
      "priority": 1,
      "expected_type": "factual",
      "search_hints": ["keyword1", "keyword2"]
    }}
  ]
}}

## Rules
1. task_type must be one of: search, analyze
2. dependencies must reference existing task_id values
3. The graph must be a DAG (no cycles)
4. Generate 4 to 8 sub_tasks
5. More fundamental/information-gathering tasks should have fewer dependencies
6. Verification tasks should depend on analysis tasks
7. Use concise but clear descriptions
8. CRITICAL — RELEVANCE CONSTRAINT: Each sub-task description MUST directly address the research question.
9. The search_hints field MUST contain keywords directly from the query.
10. Prefer specific, actionable queries over broad, vague ones.
11. LANGUAGE: All task descriptions and search_hints MUST be in the SAME language as the query.
   If the query is in Chinese (中文), ALL descriptions MUST be in Chinese.

## Dependency Design Examples
好的依赖设计（信息流方向 search → analyze，search 是无依赖的叶子节点）：
- task_1 (search: 收集投资数据) → dependencies: []
- task_2 (search: 收集政策) → dependencies: []
- task_3 (analyze: 对比投资与政策) → dependencies: ["task_1", "task_2"]
✅ search 任务无依赖（叶子节点），analyze 任务依赖 search 的结果。

坏的依赖设计（严禁这样做）：
- task_1 (search) → dependencies: ["task_3"]   ❌ search 依赖 analyze，方向反了
- task_3 (analyze) → dependencies: []           ❌ analyze 无依赖，拿不到上游信息

## Anti-examples
- Query: "How to find an internship at a big tech company" → BAD tasks about unrelated topics
- Query: "大模型Agent技术趋势" → descriptions MUST be in Chinese like "搜索大模型Agent的主要技术方法"

## Context (if any)
{memory_context}
"""

REPLAN_PROMPT = """\
You are an expert research planner. Some sub-tasks failed and need to be re-planned.

## Original Question
{query}

## Failed Tasks
{failed_tasks_json}

## Accepted Successful Results to Preserve
{preserved_results_json}

## Reason for Failure
{reason}

## Output Format
Return a JSON object with new sub_tasks. You may:
1. Modify failed tasks (new task_id, same or different description)
2. Add new tasks to fill gaps
3. Remove tasks that are no longer needed
4. Keep dependencies consistent. Accepted tasks omitted from your response will be
   restored from their original definitions and may be referenced by task_id.
   Include a task explicitly only when you intend to change its definition.

Structure:
{{
  "sub_tasks": [...]
}}

Only return the JSON. No markdown, no extra text.
"""

# ---------------------------------------------------------------------------
# 规划前侦察阶段（Reconnaissance）Prompt
# ---------------------------------------------------------------------------
# 设计目标：在生成 DAG 之前，用固定次数的搜索调用拿到真实情报，
# 让 search_hints / 子任务维度建立在事实而非模型先验之上。
# 该阶段与最终规划阶段使用两套完全独立的 message list（见 generate_plan）。

PLANNER_SEARCH_SYSTEM_PROMPT = """\
你是 DeepResearch 系统的「规划前侦察助手」。你的职责不是拆解任务，而是在规划开始前，
用极其有限的搜索机会验证研究问题的关键事实，为规划器提供可靠情报。

【当前阶段】侦察阶段（Reconnaissance）
【搜索预算】你总共只有 {total} 次工具调用机会，每次调用只能传入一个 query。
【可用工具】仅 2 个：
  - web_search: 通用网页搜索，适合新闻、行业数据、市场报告、时效性信息
  - arxiv_reader: 学术论文检索（ArXiv / OpenAlex / Semantic Scholar），无需 API Key，始终可用

【使用规则】
1. 搜索关键词必须直接取自研究问题本身，并使用与研究问题相同的语言（中文问题 → 中文关键词）。
2. 一次调用只查一个主题，不要试图用一个 query 覆盖多个子问题。
3. 如果 web_search 返回错误或无结果，立即改用 arxiv_reader；绝对不要重试同一个已失败的工具。
4. 优先搜索「仅凭问题文本最容易搞错的事实」：核心实体的准确名称、时间范围、关键数字、领域术语。
5. 搜索预算用完后，系统会切换阶段并要求你输出侦察摘要，届时严禁再调用工具。

【侦察目标】（按重要性排序）
a) 确认研究问题中核心实体 / 人物 / 机构 / 技术 / 时间范围的准确名称与最新状态
b) 发现 1-2 个仅凭问题文本容易遗漏的重要子维度
c) 沉淀 3-5 个高质量关键词，供规划器生成 search_hints
"""

PLANNER_SEARCH_USER_PROMPT = """\
## 研究问题
{query}

{memory_block}

## 任务
你现在位于规划前的侦察阶段。共 {total} 次搜索机会，已用 {used} 次，剩余 {remaining} 次。

请立即调用一次搜索工具（web_search 或 arxiv_reader），获取与研究问题最相关的信息。
要求：
- 选择当前最不确定、对后续拆解子任务影响最大的一个角度进行搜索
- 查询词使用与研究问题相同的语言
- 如果是第 2 次及以后的机会，必须换一个与前面不同的角度，严禁重复已搜过的关键词
"""

PLANNER_SEARCH_NUDGE_PROMPT = """\
⚠️ 你刚才没有调用任何工具。侦察阶段的任务就是调用搜索工具，不是写分析。
你必须【现在立刻】调用一次搜索工具（web_search 或 arxiv_reader）来获取信息。
不要输出分析文字，不要输出 JSON，只发起一次工具调用。
"""

PLANNER_SEARCH_CONTINUE_PROMPT = """\
【搜索预算提醒】共 {total} 次机会，已用 {used} 次，剩余 {remaining} 次。
请立即再调用一次搜索工具，必须换一个与之前不同的角度，严禁重复已搜过的关键词。
"""

PLANNER_FINAL_SWITCH_PROMPT = """\

## 阶段切换：侦察结束 → 开始规划

你的 {total} 次搜索机会已全部用完，工具调用功能已被系统在 API 层禁用。

⛔ 从现在起严禁再调用任何工具。即使对话历史中仍然展示着工具定义，也不允许再发起 tool call。
⛔ 严禁在正文中手写任何伪 XML 形式的工具调用块或其裸标记。
✅ 你必须直接输出最终的任务 DAG JSON 对象，除此之外不要输出任何解释、前言或道歉。

{digest_block}"""

_DIGEST_WITH_RESULTS = """\
下方是你在侦察阶段获得的真实检索结果，请把它们作为拆解子任务的事实依据，
而不是完全依赖你自己的先验知识：

{entries}"""

_DIGEST_NO_RESULTS = """\
侦察阶段未获得有效检索结果（工具调用失败或未执行）。请完全基于研究问题文本和你自己的知识进行规划，
并在 search_hints 中给出你认为最有效的关键词。"""


class Planner:
    """自适应规划器。

    Attributes:
        policy: VLLMPolicy 实例，用于调用 LLM。
        budget_tracker: 可选的预算追踪器，监控 planning 阶段的 token 消耗。
    """

    def __init__(
        self,
        policy,
        budget_tracker: BudgetTracker | None = None,
        tools: list | None = None,
        search_calls: int = 1,
        search_tools: list[str] | None = None,
        max_sub_tasks: int | None = None,
    ) -> None:
        self.policy = policy
        if max_sub_tasks is not None and (type(max_sub_tasks) is not int or max_sub_tasks < 1):
            raise ValueError("max_sub_tasks must be a positive integer or None")
        self.max_sub_tasks = max_sub_tasks
        self.budget_tracker = budget_tracker or BudgetTracker()
        self._last_raw_json: str = ""
        # 固定搜索次数：负数 / None / 非法值一律归零，保证循环次数有确定上界
        try:
            self.search_calls = max(0, int(search_calls))
        except (TypeError, ValueError):
            self.search_calls = 0
            logger.warning(f"[Planner] search_calls={search_calls!r} 无法解析，已降级为 0")
        # 只注册白名单内的搜索工具
        whitelist = self._resolve_whitelist(search_tools)
        self._tools = [t for t in (tools or []) if getattr(t, "name", "") in whitelist]
        self._tool_map: dict[str, Any] = {t.name: t for t in self._tools}
        # 可观测性：最近一次侦察阶段拿到的检索条目
        self.last_search_entries: list[dict[str, Any]] = []

    @staticmethod
    def _resolve_whitelist(search_tools: list[str] | None) -> tuple[str, ...]:
        """把配置里的工具名单收敛到硬编码安全集内。

        配置写错（拼错、或试图放入 browser / file_reader）只会被忽略并告警，
        不会让 Planner 获得白名单之外的能力。返回顺序遵循 PLANNER_SEARCH_TOOL_WHITELIST
        的优先级（web_search 优先，arxiv_reader 兜底）。
        """
        if not search_tools:
            return PLANNER_SEARCH_TOOL_WHITELIST
        requested = [str(t).strip() for t in search_tools if str(t).strip()]
        rejected = [t for t in requested if t not in PLANNER_SEARCH_TOOL_WHITELIST]
        if rejected:
            logger.warning(
                f"[Planner] 配置 search_tools 中的 {rejected} 不在安全白名单 "
                f"{list(PLANNER_SEARCH_TOOL_WHITELIST)} 内，已忽略"
            )
        kept = tuple(t for t in PLANNER_SEARCH_TOOL_WHITELIST if t in requested)
        if not kept:
            logger.warning("[Planner] search_tools 收敛后为空，回退到默认白名单")
            return PLANNER_SEARCH_TOOL_WHITELIST
        return kept

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    @trace_chain(name="planner.generate_plan", tags=["m2", "planner"])
    async def generate_plan(self, query: str, memory_context: str = "") -> DAG:
        """生成初始执行计划（DAG）。

        两阶段流程：
          Phase A  规划前侦察：用固定 `search_calls` 次搜索工具调用获取真实情报。
                   该阶段是「增强项」而非「前置依赖」——任何失败都只降级不中断。
          Phase B  规划生成：禁用工具，把侦察摘要注入 prompt，输出 DAG JSON。
                   带 Schema 校验，LLM 输出格式不合法时自动回传错误并重试（最多 2 次）。

        Raises:
            PlanParseError: LLM 输出无法解析为合法 DAG 时抛出。
        """
        digest = await self._run_search_rounds(query, memory_context)
        return await self._finalize_plan(query, memory_context, digest)

    # ------------------------------------------------------------------
    # Phase A: 规划前侦察（固定次数搜索）
    # ------------------------------------------------------------------

    @trace_chain(name="planner.reconnaissance")
    async def _run_search_rounds(self, query: str, memory_context: str) -> str:
        """执行固定次数的搜索侦察，返回可注入 prompt 的摘要文本。

        边界保证：
          - search_calls <= 0 或无可用工具 → 直接返回 ""（行为与关闭侦察完全一致）
          - 工具抛异常 / 返回 error → 记为一条 digest 条目，不中断循环
          - LLM 调用 raise（402 鉴权、上下文超限）→ 放弃侦察，返回已有摘要
          - 模型未发起 tool call → 同轮内 nudge 一次，仍不调则提前结束
          - 每轮只执行响应中的第一个 tool call（与「一次一个 query」的约定一致）
        """
        from src.utils.recovery import ToolExecutor
        self._search_executor = ToolExecutor(self._tools)
        total = self.search_calls
        if total <= 0 or not self._tool_map:
            if total > 0:
                logger.warning(
                    f"[Planner] search_calls={total} 但无可用搜索工具，跳过侦察阶段"
                )
            return ""

        has_memory = bool(memory_context and memory_context.strip() and memory_context != "None")
        memory_block = (
            f"## 已有研究记忆（可用于判断哪些维度已被覆盖）\n{memory_context}"
            if has_memory
            else "（无历史研究记忆，这是首次研究该主题）"
        )

        messages: list[dict] = [
            {"role": "system", "content": PLANNER_SEARCH_SYSTEM_PROMPT.format(total=total)},
            {
                "role": "user",
                "content": PLANNER_SEARCH_USER_PROMPT.format(
                    query=query,
                    memory_block=memory_block,
                    total=total,
                    used=0,
                    remaining=total,
                ),
            },
        ]

        entries: list[dict[str, Any]] = []
        if hasattr(self.policy, "set_tools"):
            self.policy.set_tools([t.get_openai_tool_schema() for t in self._tools])

        try:
            for round_idx in range(total):
                if not self._search_executor.available():
                    break
                if hasattr(self.policy, "set_tools"):
                    self.policy.set_tools([self._tool_map[name].get_openai_tool_schema()
                                           for name in self._search_executor.available()])
                if round_idx > 0:
                    messages.append({
                        "role": "user",
                        "content": PLANNER_SEARCH_CONTINUE_PROMPT.format(
                            total=total, used=round_idx, remaining=total - round_idx
                        ),
                    })

                resp = await self._call_policy(messages)
                if resp is None:
                    break  # 致命错误，放弃侦察

                tool_calls = list(resp.get("tool_calls") or [])
                content = resp.get("content") or ""

                if not tool_calls:
                    # 模型没调工具：同轮内给一次强制 nudge，再不调就结束侦察
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": PLANNER_SEARCH_NUDGE_PROMPT})
                    resp = await self._call_policy(messages)
                    if resp is None:
                        break
                    tool_calls = list(resp.get("tool_calls") or [])
                    content = resp.get("content") or ""
                    if not tool_calls:
                        logger.info("[Planner] 侦察阶段模型两次均未调用工具，提前结束")
                        break

                tc = dict(tool_calls[0])
                tc_id = tc["id"] = tc.get("id") or f"planner_search_{round_idx}"
                entry = await self._execute_search_tool(tc)
                entries.append(entry)
                self.budget_tracker.track(len(entry.get("text", "")) // 3)

                # assistant 消息只挂「已执行的那一个」tool_call，
                # 否则 API 会因 tool_call 缺少对应 tool 响应而报错
                assistant_msg = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [dict(tc)],
                }
                if resp.get("reasoning_content"):
                    assistant_msg["reasoning_content"] = resp["reasoning_content"]
                messages.append(assistant_msg)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": entry.get("tool", ""),
                    "content": entry.get("text", ""),
                })
        finally:
            # 无论侦察是否成功，都清掉工具注册，防止状态残留影响后续调用
            if hasattr(self.policy, "set_tools"):
                try:
                    self.policy.set_tools(None)
                except Exception as e:  # pragma: no cover - 防御性
                    logger.warning(f"[Planner] 恢复 set_tools(None) 失败: {e}")

        self.last_search_entries = entries
        digest = self._build_search_digest(entries)
        logger.info(
            f"[Planner] 侦察阶段结束: 预算={total}, 实际检索={len(entries)} 次, "
            f"digest={len(digest)} 字符"
        )
        return digest

    async def _call_policy(self, messages: list[dict]):
        """把同步 policy 包进线程池；返回 None 表示本次调用应被跳过。

        侦察阶段的 LLM 失败绝不能中断整体规划，因此这里吞掉所有异常。
        """
        try:
            return await call_policy(self.policy, messages)
        except TimeoutError:
            raise
        except Exception as e:
            logger.warning(f"[Planner] 侦察阶段 LLM 调用失败，跳过剩余侦察: {e}")
            return None

    @trace_tool(name="planner.search_tool")
    async def _execute_search_tool(self, tool_call: Any) -> dict[str, Any]:
        """侦察共用 L1–L3 控制器；L5、超时和取消向外传播。"""
        from src.utils.recovery import ToolExecutor, current_recovery
        func = (tool_call.get("function") or {}) if isinstance(tool_call, dict) else {}
        tool_name = str(func.get("name") or "")
        try:
            args = func.get("arguments", "{}")
            args = json.loads(args) if isinstance(args, str) else args
        except (ValueError, TypeError):
            args = {}
        query_hint = str(args.get("query") or args.get("paper_id") or "") if isinstance(args, dict) else ""
        executor = getattr(self, "_search_executor", None)
        if executor is None or (current_recovery() is not None and executor.state is not current_recovery()):
            executor = self._search_executor = ToolExecutor(self._tools)
        outcome = await executor.run(tool_name, func.get("arguments", "{}"),
                                     tool_call.get("id") if isinstance(tool_call, dict) else None)
        if outcome.failure and outcome.failure.level == 1:
            # The next bounded reconnaissance turn may repair the call.
            if not executor.repair_feedback():
                executor.disabled.update(executor.tools)
        return {
            "tool": tool_name,
            "query": query_hint,
            "text": self._format_tool_result(tool_name, outcome.data) if not outcome.failure else
                    f"[L{outcome.failure.level} {outcome.failure.code}] {outcome.failure.message}",
            "error": outcome.status == "error",
            "recovery": outcome.to_dict(),
        }

    @staticmethod
    def _format_tool_result(
        tool_name: str,
        result: Any,
        max_chars: int = PLANNER_TOOL_RESULT_MAX_CHARS,
    ) -> str:
        """把工具返回 dict 压成紧凑可读文本，控制进入上下文的体量。"""
        if not isinstance(result, dict):
            return str(result)[:max_chars]
        if result.get("error"):
            return f"[工具错误] {result['error']}"

        lines: list[str] = []
        if tool_name == "web_search":
            for i, r in enumerate(result.get("results", [])[:4], 1):
                if not isinstance(r, dict):
                    continue
                title = str(r.get("title", "")).strip()
                url = str(r.get("url", "")).strip()
                snippet = " ".join(str(r.get("snippet", "")).split())[:220]
                if not (title or snippet):
                    continue
                lines.append(f"[{i}] {title}" + (f" — {url}" if url else ""))
                if snippet:
                    lines.append(f"    {snippet}")
        elif tool_name == "arxiv_reader":
            papers = result.get("papers") or result.get("results") or []
            for i, p in enumerate(papers[:4], 1):
                if not isinstance(p, dict):
                    continue
                title = str(p.get("title", "")).strip()
                url = str(p.get("pdf_url") or p.get("url") or "").strip()
                summary = " ".join(str(p.get("summary") or p.get("abstract") or "").split())[:220]
                if not (title or summary):
                    continue
                lines.append(f"[{i}] {title}" + (f" — {url}" if url else ""))
                if summary:
                    lines.append(f"    {summary}")

        if not lines:
            return json.dumps(result, ensure_ascii=False, default=str)[:max_chars]
        return "\n".join(lines)[:max_chars]

    @staticmethod
    def _build_search_digest(
        entries: list[dict[str, Any]],
        max_chars: int = PLANNER_SEARCH_DIGEST_MAX_CHARS,
    ) -> str:
        """把检索条目压成 digest；超限时优先保留最近一次检索的完整内容。"""
        if not entries:
            return ""
        blocks = [
            f"### 检索 {i} | 工具: {e.get('tool', '?')} | query: \"{e.get('query', '')}\"\n"
            f"{e.get('text', '')}"
            for i, e in enumerate(entries, 1)
        ]
        joined = "\n\n".join(blocks)
        if len(joined) <= max_chars:
            return joined

        kept: list[str] = []
        budget = max_chars
        for block in reversed(blocks):  # 新的检索优先保留
            if budget <= 0:
                break
            if len(block) <= budget:
                kept.insert(0, block)
                budget -= len(block) + 2
            else:
                kept.insert(0, block[:max(budget, 200)])
                budget = 0
        return "\n\n".join(kept)[:max_chars]

    # ------------------------------------------------------------------
    # Phase B: 规划生成（禁用工具 + Schema 校验重试）
    # ------------------------------------------------------------------

    async def _finalize_plan(self, query: str, memory_context: str, digest: str) -> DAG:
        """禁用工具，基于侦察摘要生成最终 DAG JSON。

        这里用一套全新的 message list，而不是接着侦察阶段的历史：
          1. 侦察历史的 assistant 消息带 tool_calls，tools 禁用后续传部分服务会 400；
          2. 原始搜索结果很占 token，用 digest 比塞全文更省；
          3. Schema 重试要 append 消息，历史干净更好控制。
        """
        from src.utils.schema_validator import validate_planner_output

        prompt = self._build_prompt(query, memory_context)
        if self.search_calls > 0 and self._tool_map:
            digest_block = (
                _DIGEST_WITH_RESULTS.format(entries=digest)
                if digest
                else _DIGEST_NO_RESULTS
            )
            prompt += PLANNER_FINAL_SWITCH_PROMPT.format(
                total=self.search_calls, digest_block=digest_block
            )

        messages: list[dict] = [
            {"role": "system", "content": "You are a research planning assistant. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ]

        content = ""
        data = None
        for attempt in range(3):  # 原始调用 + 2 次重试
            if attempt and current_recovery() is not None and not current_recovery().action():
                raise PlanParseError("Plan format repair budget exhausted")
            record_event("planner_attempt", {"attempt": attempt + 1, "kind": "plan"})
            try:
                response = await call_policy(self.policy, messages)
                content = response.get("content", "") or ""
            except RuntimeError as e:
                raise PlanParseError(f"LLM call failed during planning: {e}") from e

            # 双保险：API 层已禁用工具，但模型可能在正文里手写调用标记，先剥掉再解析
            content = self._strip_tool_call_markers(content)

            # 尝试解析 JSON
            data = self._try_parse_json(content)
            if data is not None:
                # Schema 校验
                validation = validate_planner_output(data)
                if validation.valid:
                    try:
                        self._check_task_limit(data)
                    except PlanParseError as exc:
                        messages.append({"role": "user", "content": str(exc) + "; regenerate the entire valid DAG within the task budget."})
                        data = None
                        continue
                    break
                # 校验失败 → 回传错误重试
                if attempt < 2:
                    messages.append({"role": "user", "content": validation.to_feedback()})
                data = None  # 重置，触发下次重试
            elif attempt < 2:
                messages.append({
                    "role": "user",
                    "content": (
                        "⚠️ 你的上次输出不是合法的 JSON 格式。"
                        "请只输出 JSON 对象，不要包含任何其他文字。"
                    )
                })

        self._last_raw_json = content
        self.budget_tracker.track(len(content) // 3)

        if data is not None:
            return self._build_dag_from_data(data)
        return self._parse_plan(content)  # 降级：用原有容错解析

    @staticmethod
    def _strip_tool_call_markers(text: str) -> str:
        """剥掉模型在正文里手写的工具调用块及其裸标记。

        vllm_policy 的正则回退分支不看 self.tools，即使 API 层已禁用工具，
        模型仍可能把调用写进正文并被解析成 tool_calls，因此最后一轮必须自己清一遍。
        """
        if not text:
            return text
        cleaned = text
        start = cleaned.find(_TOOL_CALL_OPEN)
        while start != -1:
            end = cleaned.find(_TOOL_CALL_CLOSE, start)
            if end == -1:
                cleaned = cleaned[:start]
                break
            cleaned = cleaned[:start] + cleaned[end + len(_TOOL_CALL_CLOSE):]
            start = cleaned.find(_TOOL_CALL_OPEN)
        return cleaned.replace(_TOOL_CALL_OPEN, "").replace(_TOOL_CALL_CLOSE, "").strip()

    def _try_parse_json(self, raw: str) -> dict | None:
        """尝试从 LLM 原始输出中解析 JSON 对象。"""
        import json as _json
        def decode(text):
            value = _json.loads(text)
            return value if isinstance(value, dict) else None
        raw = raw.strip()
        try:
            return decode(raw)
        except _json.JSONDecodeError:
            pass
        # 提取 ```json 代码块
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            try:
                return decode(m.group(1).strip())
            except _json.JSONDecodeError:
                pass
        # 提取 {...}
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return decode(m.group(0))
            except _json.JSONDecodeError:
                pass
        return None

    def _build_dag_from_data(self, data: dict) -> DAG:
        """从校验通过的 dict 构建 DAG（跳过 JSON 解析，直接构建）。"""
        self._check_task_limit(data)
        sub_tasks_raw = data.get("sub_tasks", [])
        dag = DAG()
        for item in sub_tasks_raw:
            task = self._deserialize_subtask(item)
            dag.add_node(task.task_id)
        for item in sub_tasks_raw:
            task_id = item.get("task_id", "")
            for dep in item.get("dependencies", []):
                if not dag.has_node(dep):
                    dag.add_node(dep)
                dag.add_edge(dep, task_id)
        # 验证无环
        try:
            dag.topological_sort()
        except DAGCycleError as e:
            raise PlanParseError(f"Planner generated a cyclic graph: {e}") from e
        return dag

    @trace_chain(name="planner.replan", tags=["m2", "planner"])
    def replan(self, *args, **kwargs):
        """Synchronous compatibility entry; async callers should await areplan()."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.areplan(*args, **kwargs))
        raise RuntimeError("Use await planner.areplan(...) inside an event loop")

    async def areplan(
        self,
        query: str,
        failed_tasks: list[SubTask],
        existing_results: list[AgentResult],
        reason: str,
    ) -> DAG:
        """增量重规划：保留高置信度结果，修改失败任务（带 Schema 校验 + retry）。

        Args:
            query: 原始研究问题。
            failed_tasks: 执行失败的 SubTask 列表。
            existing_results: 所有历史执行结果。
            reason: 失败原因描述。

        Returns:
            DAG: 新的执行计划。
        """
        from src.utils.schema_validator import validate_planner_output
        # Acceptance, rather than model self-reported confidence, controls reuse.
        preserved = [
            {
                "task_id": r.task_id,
                "output": str(r.output)[:500] if r.output else "",
                "confidence": r.confidence,
            }
            for r in existing_results
            if r.status.value == "success" and r.acceptance.get("passed", True)
        ]
        old_data = self._try_parse_json(self._last_raw_json) or {}
        accepted_ids = {r["task_id"] for r in preserved}
        preserved_tasks = [t for t in old_data.get("sub_tasks", []) if t.get("task_id") in accepted_ids]

        failed_json = json.dumps(
            [{"task_id": t.task_id, "description": t.description, "type": t.task_type.value} for t in failed_tasks],
            ensure_ascii=False,
            indent=2,
        )
        preserved_json = json.dumps(preserved, ensure_ascii=False, indent=2)

        prompt = REPLAN_PROMPT.format(
            query=query,
            failed_tasks_json=failed_json,
            preserved_results_json=preserved_json,
            reason=reason,
        )
        if self.max_sub_tasks is not None:
            prompt += f"\nHard budget: generate at most {self.max_sub_tasks} sub_tasks in this DAG."
        messages = [
            {"role": "system", "content": "You are a research planning assistant. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ]

        content = ""
        data = None
        for attempt in range(3):
            if attempt and current_recovery() is not None and not current_recovery().action():
                raise PlanParseError("Replan format repair budget exhausted")
            record_event("planner_attempt", {"attempt": attempt + 1, "kind": "replan"})
            try:
                response = await call_policy(self.policy, messages)
                content = response.get("content", "") or ""
            except RuntimeError as e:
                raise PlanParseError(f"LLM call failed during replanning: {e}") from e

            data = self._try_parse_json(content)
            if data is not None:
                present = {t.get("task_id") for t in data.get("sub_tasks", [])
                           if isinstance(t, dict) and isinstance(t.get("task_id"), str)} if isinstance(data.get("sub_tasks"), list) else set()
                if isinstance(data.get("sub_tasks"), list):
                    data["sub_tasks"].extend(t for t in preserved_tasks if t["task_id"] not in present)
                validation = validate_planner_output(data)
                if validation.valid:
                    try:
                        self._check_task_limit(data)
                    except PlanParseError as exc:
                        messages.append({"role": "user", "content": str(exc) + "; regenerate the entire DAG within budget."})
                        data = None
                        continue
                    break
                if attempt < 2:
                    messages.append({"role": "user", "content": validation.to_feedback()})
                data = None
            elif attempt < 2:
                messages.append({
                    "role": "user",
                    "content": (
                        "⚠️ 你的上次输出不是合法的 JSON 格式。"
                        "请只输出 JSON 对象，不要包含任何其他文字。"
                    )
                })

        self._last_raw_json = json.dumps(data, ensure_ascii=False) if data is not None else content
        self.budget_tracker.track(len(content) // 3)

        if data is not None:
            return self._build_dag_from_data(data)
        return self._parse_plan(content)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, memory: str) -> str:
        """构建初始规划 prompt。"""
        # 首次运行（无历史记忆）时，提示 Planner 更激进地拆解子任务
        has_memory = bool(memory and memory.strip() and memory != "None")
        if self.max_sub_tasks is not None:
            extra_hint = f"\nHard budget: at most {self.max_sub_tasks} sub_tasks. Honor a smaller user-requested count. Do not invent tasks just to fill the budget."
        elif not has_memory:
            extra_hint = (
                "\n## Note\n"
                "No previous research memory is available for this topic. "
                "Please be MORE AGGRESSIVE in decomposition: generate 6-10 sub_tasks to thoroughly cover the topic, "
                "rather than the usual 3-5. Each sub-task should focus on a distinct angle or data source.\n"
                "IMPORTANT: Each sub-task description must directly reflect the user's original intent. "
                "If the user asks about 'internship application strategies', do NOT generate tasks about '2025 tech trends' or 'annual science summary'."
            )
        else:
            extra_hint = (
                "\n## Note\n"
                "Use the preserved successful results above to inform new sub-tasks. "
                "New tasks should fill gaps and avoid duplicating existing coverage."
            )
        prompt = INITIAL_PLAN_PROMPT.format(query=query, memory_context=memory or "None")
        if self.max_sub_tasks is not None:
            prompt = prompt.replace("Generate 4 to 8 sub_tasks", f"Generate 1 to {self.max_sub_tasks} sub_tasks")
        return prompt + extra_hint

    def _check_task_limit(self, data: dict) -> None:
        if self.max_sub_tasks is not None and len(data.get("sub_tasks", [])) > self.max_sub_tasks:
            raise PlanParseError(f"Task count exceeds hard budget {self.max_sub_tasks}")

    def _parse_plan(self, json_str: str) -> DAG:
        """健壮性 JSON 解析：处理 markdown 代码块、多余换行等噪声。

        解析策略:
          1. 先尝试直接 json.loads
          2. 失败则提取 markdown 代码块内容
          3. 清理常见噪声（尾部逗号、注释等）
          4. 验证 DAG 无环
        """
        raw = json_str.strip()

        # 尝试提取 markdown 代码块
        if raw.startswith("```"):
            # 去掉首行 ```json 或 ```
            lines = raw.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        # 尝试提取 ```json...``` 中间的内容（即使不在开头）
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if code_block_match:
            raw = code_block_match.group(1).strip()

        # 尝试直接找最外层的 JSON 对象
        if not raw.startswith("{"):
            obj_match = re.search(r"(\{.*\})", raw, re.DOTALL)
            if obj_match:
                raw = obj_match.group(1).strip()

        # 清理尾部逗号（JSON 不允许 trailing comma）
        raw = re.sub(r",(\s*[}\]])", r"\1", raw)

        # 解析 JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            # 最后尝试：逐行修复（去掉注释）
            cleaned_lines = []
            for line in raw.splitlines():
                # 去掉 // 注释
                if "//" in line:
                    line = line[: line.index("//")]
                cleaned_lines.append(line)
            try:
                data = json.loads("\n".join(cleaned_lines))
            except json.JSONDecodeError:
                raise PlanParseError(
                    f"Failed to parse planner output as JSON. Raw snippet: {json_str[:500]}"
                ) from e

        if not isinstance(data, dict) or "sub_tasks" not in data:
            raise PlanParseError(f"Planner output missing 'sub_tasks' key. Keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")

        from src.utils.schema_validator import validate_planner_output
        validation = validate_planner_output(data)
        if not validation.valid:
            raise PlanParseError(validation.to_feedback())
        self._last_raw_json = json.dumps(data, ensure_ascii=False)

        self._check_task_limit(data)
        sub_tasks_raw = data["sub_tasks"]
        if not isinstance(sub_tasks_raw, list):
            raise PlanParseError(f"'sub_tasks' must be a list, got {type(sub_tasks_raw)}")

        dag = DAG()
        for item in sub_tasks_raw:
            task = self._deserialize_subtask(item)
            dag.add_node(task.task_id)

        # 第二遍添加边
        for item in sub_tasks_raw:
            task_id = item.get("task_id", "")
            for dep in item.get("dependencies", []):
                if not dag.has_node(dep):
                    # 依赖指向不存在的任务，创建占位节点
                    dag.add_node(dep)
                dag.add_edge(dep, task_id)  # dep -> task_id (task_id 依赖 dep)

        # 验证无环
        try:
            dag.topological_sort()
        except DAGCycleError as e:
            raise PlanParseError(f"Planner generated a cyclic graph: {e}") from e

        return dag

    def _deserialize_subtask(self, item: dict[str, Any]) -> SubTask:
        """将 JSON dict 反序列化为 SubTask。"""
        task_type_str = item.get("task_type", "search")
        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            task_type = TaskType.SEARCH  # 默认值降级

        return SubTask(
            task_id=item.get("task_id", "unknown"),
            task_type=task_type,
            description=item.get("description", ""),
            dependencies=list(item.get("dependencies", [])),
            context_keys=list(item.get("context_keys", [])),
            timeout_seconds=item.get("timeout_seconds", 120),
            priority=int(item.get("priority", 1)),
            expected_type=item.get("expected_type", "factual"),
            search_hints=list(item.get("search_hints", [])),
        )

    def get_task_map_from_dag(self, dag: DAG, raw_json: str) -> dict[str, SubTask]:
        """从 DAG 和原始 JSON 重建 task_id -> SubTask 映射。

        通常在 generate_plan 后由编排器调用。
        """
        # 复用 _parse_plan 中的解析逻辑，但返回映射
        # 这里重新解析 raw_json 以获取完整 SubTask 信息
        raw = raw_json.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if code_block_match:
            raw = code_block_match.group(1).strip()
        if not raw.startswith("{"):
            obj_match = re.search(r"(\{.*\})", raw, re.DOTALL)
            if obj_match:
                raw = obj_match.group(1).strip()
        raw = re.sub(r",(\s*[}\]])", r"\1", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}

        sub_tasks_raw = data.get("sub_tasks", [])
        return {item.get("task_id", f"task_{i}"): self._deserialize_subtask(item)
                for i, item in enumerate(sub_tasks_raw)}
