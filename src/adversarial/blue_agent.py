"""
M5 Blue Agent — 修复与防御器

Blue Agent 接收 Red Agent 的 Verdict，按优先级排序并执行三类修复：
1. In-place Fix：数字/日期与 source 不一致 → 直接替换
2. Supplementary Search：unsourced claims → 触发新搜索
3. Removal：高置信幻觉 → 删除段落

修复后执行 self_verify，确保不引入新矛盾。
"""
from __future__ import annotations

import copy
import logging
from typing import Any

from src.adversarial.verdict import (
    FixOperation,
    FixType,
    Issue,
    RedVerdict,
    Severity,
    VerdictEngine,
)
from src.orchestrator.schemas import ResearchReport
from src.utils.tracing import trace_agent


logger = logging.getLogger(__name__)


__all__ = ["BlueAgent"]


# ============================================================================
# Prompt 模板
# ============================================================================

SYSTEM_BLUE_AGENT = (
    "你是一位严谨的研究报告修订员（Blue Agent）。你的任务是根据审查意见修复研究报告，"
    "确保所有修改都有据可依，不引入新错误。输出必须是 JSON 格式。"
)

PROMPT_SELF_VERIFY = """请验证以下修复后的研究报告是否存在新引入的矛盾或错误。

请按以下 JSON 格式输出：
{
  "has_new_issue": bool,
  "new_issues": [
    {
      "severity": "critical|major|minor",
      "description": "string",
      "location": "string"
    }
  ]
}

--- 原始报告 ---
{original}

--- 修复后报告 ---
{revised}

--- 已执行的修复 ---
{fixes}
"""

PROMPT_IN_PLACE_FIX = """请根据以下审查意见，对研究报告进行【原地修正】。

要求：
1. 仅修改与 source 不一致的具体数字、日期、名字等事实性内容。
2. 保持原文结构和叙述风格不变。
3. 所有修改必须基于提供的 sources，不能引入新信息。
4. 输出修正后的完整段落。

请按以下 JSON 格式输出：
{
  "fixed_content": "string",   // 修正后的报告全文
  "changes": [                 // 变更记录列表
    {
      "location": "string",
      "before": "string",
      "after": "string"
    }
  ]
}

--- 审查意见 ---
{issue_desc}

--- 原始报告 ---
{content}

--- 来源 ---
{sources}
"""

PROMPT_SUPPLEMENTARY_SEARCH = """请根据以下审查意见，对研究报告进行【补充搜索后修正】。

要求：
1. 对无 source 支撑的 claim，使用搜索结果补充证据。
2. 如果搜索结果无法证实，则删除该 claim 或标注为"未经证实"。
3. 输出修正后的完整报告。

请按以下 JSON 格式输出：
{
  "fixed_content": "string",
  "changes": [
    {
      "location": "string",
      "action": "added|removed|modified",
      "detail": "string"
    }
  ]
}

--- 审查意见 ---
{issue_desc}

--- 原始报告 ---
{content}

--- 搜索结果 ---
{search_results}
"""

PROMPT_REMOVAL = """请根据以下审查意见，对研究报告进行【移除修正】。

要求：
1. 删除高置信度幻觉段落或无法验证的 claim。
2. 删除后确保上下文连贯，必要时添加过渡句。
3. 输出修正后的完整报告。

请按以下 JSON 格式输出：
{
  "fixed_content": "string",
  "removed_segments": [
    {
      "location": "string",
      "original_text": "string",
      "reason": "string"
    }
  ]
}

--- 审查意见 ---
{issue_desc}

--- 原始报告 ---
{content}
"""


# ============================================================================
# Blue Agent 实现
# ============================================================================

class BlueAgent:
    """Blue Agent — 修复与防御器。

    Attributes:
        policy: VLLMPolicy 实例。
        tools: 可用工具列表，至少包含搜索工具用于 supplementary_search。
        max_tokens: 单次修复调用的最大输出 token。
    """

    # 禁止出现在修复后报告中的内容模式（防止 Blue Agent 污染报告）
    _FORBIDDEN_PATTERNS = [
        "由于搜索api", "由于api", "api密钥无效", "api密钥",
        "搜索api", "博查ai", "密钥无效", "无法获取任何搜索",
        "未能获取任何", "建议用户检查", "检查搜索工具配置",
        "本报告内容为空", "无法补充来源", "由于搜索API错误",
        "请读者谅解", "无法完成研究",
        "报告已截断", "仅显示前",  # 截断标记污染
    ]

    def __init__(self, policy, tools: list[Any] | None = None, max_tokens: int = 4096):
        self.policy = policy
        self.tools = tools or []
        self.max_tokens = max_tokens
        # 缓存搜索工具
        self._search_tool = self._find_search_tool()

    def _find_search_tool(self) -> Any | None:
        """从 tools 列表中查找搜索工具。"""
        for t in self.tools:
            name = getattr(t, "name", "")
            if "search" in name.lower():
                return t
        return None

    def _is_forbidden_content(self, content: str) -> bool:
        """检测内容是否包含禁止的 API 错误/道歉等污染模式。"""
        if not content:
            return False
        c = content.lower()
        for pat in self._FORBIDDEN_PATTERNS:
            if pat in c:
                return True
        return False

    @trace_agent(name="blue_agent.defend", tags=["m5", "blue", "adversarial"])
    async def defend(
        self, report: ResearchReport, verdict: RedVerdict
    ) -> tuple[ResearchReport, list[FixOperation]]:
        """根据 Red Verdict 修复研究报告。

        执行流程：
        1. 按优先级对 issues 排序。
        2. 逐个执行修复（in_place / search / removal）。
        3. 每轮修复后执行 self_verify，检测是否引入新问题。
        4. 返回修复后的报告和所有 FixOperation 记录。

        Args:
            report: 原始研究报告（不会被修改，内部深拷贝）。
            verdict: Red Agent 的审查结果。

        Returns:
            (fixed_report, fix_operations)
        """
        current = copy.deepcopy(report)
        original_content = report.content  # 保存原始完整报告（用于回退）
        original_len = len(original_content) if original_content else 0
        operations: list[FixOperation] = []

        if not verdict.issues:
            return current, operations

        # 按优先级降序排序
        sorted_issues = sorted(
            verdict.issues,
            key=lambda issue: VerdictEngine.compute_priority(issue),
            reverse=True,
        )

        supplementary_count = 0  # SUPPLEMENTARY 修复计数器（每轮最多 3 次）
        total_issues = len(sorted_issues)
        fix_type_counts = {"in_place": 0, "search": 0, "removal": 0, "skipped": 0}

        for issue in sorted_issues:
            fix_type_counts[issue.fix_type.value] = fix_type_counts.get(issue.fix_type.value, 0) + 1

            # SUPPLEMENTARY 修复每轮最多执行 3 次（避免大量搜索调用）
            if issue.fix_type == FixType.SUPPLEMENTARY:
                if supplementary_count >= 3:
                    fix_type_counts["skipped"] += 1
                    op = FixOperation(
                        issue=issue,
                        action="supplementary_skipped",
                        success=False,
                        detail=f"SUPPLEMENTARY limit reached ({supplementary_count}/3 in this round)",
                    )
                    operations.append(op)
                    continue
                supplementary_count += 1
                logger.info(
                    f"[BlueAgent] SUPPLEMENTARY fix #{supplementary_count}/3: "
                    f"severity={issue.severity.value}, dim={issue.dimension.value}"
                )

            content_before_fix = current.content  # 保存修复前内容用于回退
            op = await self._fix_single_issue(current, issue)
            operations.append(op)

            # 验证修复后的内容没有被污染（API 错误/道歉等）
            if self._is_forbidden_content(current.content):
                current.content = content_before_fix
                op.success = False
                op.detail = f"REJECTED (forbidden content): {op.detail[:200]}"

            # 验证修复后内容没有被截断（LLM 只能看到 4000 字，返回固定长度）
            # 如果原始报告 >4000 字，但修复后 <60% 原始长度 → 修复引入了截断
            new_len = len(current.content) if current.content else 0
            if original_len > 4000 and new_len < original_len * 0.6:
                current.content = content_before_fix
                op.success = False
                op.detail = f"REJECTED (truncation detected: {original_len}→{new_len} chars): {op.detail[:200]}"

            # self_verify：检查修复是否引入新矛盾
            verify_pass, verify_issues = await self._self_verify(
                original_content, current.content, operations
            )
            if not verify_pass:
                # 引入新问题：记录但继续（优先处理高优先级 issue，避免死锁）
                for vi in verify_issues:
                    operations.append(
                        FixOperation(
                            issue=vi,
                            action="self_verify_detected_new_issue",
                            success=False,
                            detail=vi.description,
                        )
                    )

        logger.info(
            f"[BlueAgent] defend done: {total_issues} issues → "
            f"IN_PLACE={fix_type_counts.get('in_place', 0)}, "
            f"SUPP_SEARCH={fix_type_counts.get('search', 0)} "
            f"(executed={supplementary_count}, skipped={fix_type_counts.get('skipped', 0)}), "
            f"REMOVAL={fix_type_counts.get('removal', 0)}"
        )
        return current, operations

    async def _fix_single_issue(
        self, report: ResearchReport, issue: Issue
    ) -> FixOperation:
        """对单个 Issue 执行修复。"""
        old_max = getattr(self.policy, "max_tokens", None)
        if old_max is not None:
            self.policy.max_tokens = self.max_tokens

        try:
            if issue.fix_type == FixType.IN_PLACE:
                result = await self._do_in_place_fix(report, issue)
            elif issue.fix_type == FixType.SUPPLEMENTARY:
                result = await self._do_supplementary_search(report, issue)
            elif issue.fix_type == FixType.REMOVAL:
                result = await self._do_removal(report, issue)
            else:
                result = FixOperation(
                    issue=issue,
                    action="unknown_fix_type",
                    success=False,
                    detail=f"未知的 fix_type: {issue.fix_type}",
                )
        finally:
            if old_max is not None:
                self.policy.max_tokens = old_max

        return result

    async def _do_in_place_fix(
        self, report: ResearchReport, issue: Issue
    ) -> FixOperation:
        """执行原地修正。"""
        prompt = PROMPT_IN_PLACE_FIX
        prompt = prompt.replace("{issue_desc}", issue.description)
        prompt = prompt.replace("{content}", self._truncate_content(report.content))
        prompt = prompt.replace("{sources}", self._format_sources(report.sources, max_items=15))
        messages = [
            {"role": "system", "content": SYSTEM_BLUE_AGENT},
            {"role": "user", "content": prompt},
        ]
        resp = self.policy(messages)
        raw = resp.content or ""

        fixed_content, changes = self._parse_fix_json(raw)
        if fixed_content:
            report.content = fixed_content
            return FixOperation(
                issue=issue,
                action=f"in_place_fix: {changes}",
                success=True,
                detail=f"changes={changes}",
            )
        return FixOperation(
            issue=issue,
            action="in_place_fix_failed",
            success=False,
            detail=raw[:500],
        )

    async def _do_supplementary_search(
        self, report: ResearchReport, issue: Issue
    ) -> FixOperation:
        """执行补充搜索后修正。"""
        search_results = ""
        if self._search_tool is not None:
            try:
                # 假设搜索工具有 async execute 或同步 execute 接口
                query = issue.description
                if hasattr(self._search_tool, "execute"):
                    if hasattr(self._search_tool.execute, "__call__"):
                        import inspect
                        if inspect.iscoroutinefunction(self._search_tool.execute):
                            sr = await self._search_tool.execute(query)
                        else:
                            sr = self._search_tool.execute(query)
                    else:
                        sr = None
                else:
                    sr = None
                search_results = str(sr) if sr else "（搜索工具未返回结果）"
            except Exception as e:
                search_results = f"（搜索失败: {e}）"
        else:
            search_results = "（无可用搜索工具）"

        prompt = PROMPT_SUPPLEMENTARY_SEARCH
        prompt = prompt.replace("{issue_desc}", issue.description)
        prompt = prompt.replace("{content}", self._truncate_content(report.content))
        # 截断搜索结果避免膨胀
        search_results = search_results[:3000] if len(search_results) >3000 else search_results
        prompt = prompt.replace("{search_results}", search_results)
        messages = [
            {"role": "system", "content": SYSTEM_BLUE_AGENT},
            {"role": "user", "content": prompt},
        ]
        resp = self.policy(messages)
        raw = resp.content or ""

        fixed_content, changes = self._parse_fix_json(raw)
        if fixed_content:
            report.content = fixed_content
            return FixOperation(
                issue=issue,
                action=f"supplementary_search: {changes}",
                success=True,
                detail=f"search_results_len={len(search_results)}, changes={changes}",
            )
        return FixOperation(
            issue=issue,
            action="supplementary_search_failed",
            success=False,
            detail=raw[:500],
        )

    async def _do_removal(
        self, report: ResearchReport, issue: Issue
    ) -> FixOperation:
        """执行移除修正。"""
        prompt = PROMPT_REMOVAL
        prompt = prompt.replace("{issue_desc}", issue.description)
        prompt = prompt.replace("{content}", self._truncate_content(report.content))
        messages = [
            {"role": "system", "content": SYSTEM_BLUE_AGENT},
            {"role": "user", "content": prompt},
        ]
        resp = self.policy(messages)
        raw = resp.content or ""

        fixed_content, removed = self._parse_removal_json(raw)
        if fixed_content:
            report.content = fixed_content
            return FixOperation(
                issue=issue,
                action=f"removal: {removed}",
                success=True,
                detail=f"removed={removed}",
            )
        return FixOperation(
            issue=issue,
            action="removal_failed",
            success=False,
            detail=raw[:500],
        )

    async def _self_verify(
        self, original: str, revised: str, operations: list[FixOperation]
    ) -> tuple[bool, list[Issue]]:
        """修复后自验证，检查是否引入新矛盾。

        Returns:
            (是否通过, 新发现的 issues 列表)
        """
        if not revised or revised == original:
            return True, []

        fixes_text = "\n".join(
            f"- [{op.issue.dimension.value}] {op.action}: {op.detail[:200]}"
            for op in operations[-5:]  # 只取最近5条，避免 prompt 过长
        )
        prompt = PROMPT_SELF_VERIFY
        prompt = prompt.replace("{original}", original[:2000])
        prompt = prompt.replace("{revised}", revised[:2000])
        prompt = prompt.replace("{fixes}", fixes_text)
        messages = [
            {"role": "system", "content": SYSTEM_BLUE_AGENT},
            {"role": "user", "content": prompt},
        ]
        resp = self.policy(messages)
        raw = resp.content or ""

        try:
            import json

            data = json.loads(raw.strip())
            has_new = bool(data.get("has_new_issue", False))
            new_issues = []
            for item in data.get("new_issues", []):
                new_issues.append(
                    Issue(
                        severity=Severity(item.get("severity", "minor")),
                        dimension=Dimension(item.get("dimension", "logical")),
                        description=item.get("description", ""),
                        location=item.get("location", ""),
                        fix_type=FixType.IN_PLACE,
                    )
                )
            return not has_new, new_issues
        except Exception:
            # self_verify 解析失败视为通过，避免阻塞修复流程
            return True, []

    def _format_sources(self, sources: list[dict], max_items: int = 15) -> str:
        """格式化来源列表，截断以避免上下文膨胀。"""
        if not sources:
            return "（无来源）"
        lines = []
        for i, s in enumerate(sources[:max_items], 1):
            title = s.get("title", "未知标题")
            url = s.get("url", "")
            snippet = s.get("snippet", "")[:300]
            lines.append(f"[{i}] {title}\nURL: {url}\nSnippet: {snippet}\n")
        if len(sources) > max_items:
            lines.append(f"... 还有 {len(sources) - max_items} 个来源未显示")
        return "\n".join(lines)

    def _truncate_content(self, content: str, max_len: int = 4000) -> str:
        """截断报告内容，避免 prompt 过长。

        注意：仅用于 LLM prompt 上下文截断，不修改 report.content。
        不加截断标记——避免标记文字被 LLM 写入 fixed_content 后替换原始报告。
        """
        if len(content) <= max_len:
            return content
        return content[:max_len]

    def _parse_fix_json(self, raw: str) -> tuple[str, list[dict]]:
        """解析 in_place / search 修复的 JSON 输出。"""
        import json
        import re

        raw = raw.strip()
        if not raw:
            return "", []
        try:
            data = json.loads(raw)
            return data.get("fixed_content", ""), data.get("changes", [])
        except json.JSONDecodeError:
            pass
        code = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
        for m in code.findall(raw):
            try:
                data = json.loads(m.strip())
                return data.get("fixed_content", ""), data.get("changes", [])
            except json.JSONDecodeError:
                continue
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace:
            try:
                data = json.loads(brace.group(0))
                return data.get("fixed_content", ""), data.get("changes", [])
            except json.JSONDecodeError:
                pass
        return "", []

    def _parse_removal_json(self, raw: str) -> tuple[str, list[dict]]:
        """解析 removal 修复的 JSON 输出。"""
        import json
        import re

        raw = raw.strip()
        if not raw:
            return "", []
        try:
            data = json.loads(raw)
            return data.get("fixed_content", ""), data.get("removed_segments", [])
        except json.JSONDecodeError:
            pass
        code = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
        for m in code.findall(raw):
            try:
                data = json.loads(m.strip())
                return data.get("fixed_content", ""), data.get("removed_segments", [])
            except json.JSONDecodeError:
                continue
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace:
            try:
                data = json.loads(brace.group(0))
                return data.get("fixed_content", ""), data.get("removed_segments", [])
            except json.JSONDecodeError:
                pass
        return "", []
