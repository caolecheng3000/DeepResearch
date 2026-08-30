"""
M5 Red Agent — 多维度对抗攻击器（并行优化版）

Red Agent 的职责是对研究报告进行多维度"攻击"，找出事实错误、幻觉、逻辑矛盾、
来源可信度问题和覆盖缺失。合并为 3 个组合维度并行执行，减少 API 调用。

设计决策：
1. 3 个组合 prompt 并行调用（asyncio.gather），替代原 5 个串行调用。
2. max_tokens 在并行执行前一次性设置，消除竞态条件。
3. 每个组合 prompt 输出多个子维度的评分，5 维度 → 3 次 LLM 调用。
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from src.adversarial.verdict import (
    Dimension,
    FixType,
    Issue,
    RedVerdict,
    Severity,
    VerdictEngine,
)
from src.orchestrator.schemas import ResearchReport
from src.utils.tracing import trace_agent


__all__ = ["RedAgent"]


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_RED_AGENT = (
    "你是一位极其严苛的研究报告审查员（Red Agent）。你的任务是以批判性思维深度审查研究报告，"
    "找出所有事实错误、幻觉、逻辑漏洞、来源缺陷和覆盖缺失。你必须基于客观证据给出评分，"
    "不能因报告写作流畅而放松标准。评分标准要严格——多数研究报告默认只有 5-6 分而非 8-9 分。"
    "输出必须是严格的 JSON 格式。"
)


# ============================================================================
# 三个组合维度 Prompt
# ============================================================================

# --- 组合 1: 事实准确性 + 幻觉检测（都需对照 sources）---
PROMPT_FACTUAL_HALLUCINATION = """请对以下研究报告进行【事实准确性与幻觉检测】综合评分。

你需要从两个子维度分别评估：

## 子维度 A: 事实准确性 (factual_accuracy)
评分标准（0-10分）：
- 10分：所有可验证的事实（数字、日期、人名、机构名、统计数据）均有可靠来源支撑，且与来源完全一致。
- 7-9分：个别非核心事实缺少直接来源，或存在轻微数值偏差（<5%）。
- 4-6分：存在明显事实错误（日期错误、数据引用错误）但核心论点仍成立。
- 1-3分：多处核心事实错误，严重损害报告可信度。
- 0分：大量事实完全错误，报告基本不可信。

## 子维度 B: 幻觉检测 (hallucination)
评分标准（0-10分）：
- 10分：报告中的每一条信息都能在 sources 中找到明确支撑，无 hallucination。
- 7-9分：存在少量"合理的推断"但未明确标注为推断，可能误导读者。
- 4-6分：存在明显的无来源陈述，尤其是具体数字、事件细节或因果关系。
- 1-3分：大量段落包含无来源信息，部分信息疑似模型编造。
- 0分：报告充斥着模型幻觉，几乎无可信内容。

审查要求：
1. 逐条提取报告中的 factual claims（数字、日期、比例、排名等），与 sources 比对。
2. 逐段检查无 sources 支撑的 claim，区分"合理推断"与"无依据断言"。
3. 标记不一致、无法验证或疑似编造的内容。

请按以下 JSON 格式输出（不要有任何额外文字）：
{{
  "factual_accuracy": {{
    "score": float,
    "issues": [
      {{
        "severity": "critical|major|minor",
        "description": "string",
        "location": "string",
        "fix_type": "in_place|search|removal",
        "evidence": "string"
      }}
    ]
  }},
  "hallucination": {{
    "score": float,
    "issues": [
      {{
        "severity": "critical|major|minor",
        "description": "string",
        "location": "string",
        "fix_type": "in_place|search|removal",
        "evidence": "string"
      }}
    ]
  }}
}}

--- 研究报告 ---
Query: {query}

Content:
{content}

--- 来源列表 ---
{sources}
"""

# --- 组合 2: 逻辑一致性（独立）---
PROMPT_LOGICAL = """请对以下研究报告进行【逻辑一致性】评分。

评分标准（0-10分）：
- 10分：论证链条完整，前提与结论一致，无矛盾陈述。
- 7-9分：个别推断稍显跳跃，但不影响整体结论。
- 4-6分：存在内部矛盾（如前文说A，后文说非A）或因果谬误。
- 1-3分：多处逻辑断裂、自相矛盾，核心论点无法自洽。
- 0分：报告逻辑混乱，论证完全不可信。

审查要求：
1. 检查是否存在前后矛盾的陈述。
2. 检查因果关系是否合理（避免 post hoc / 因果倒置）。
3. 检查样本推断总体是否存在以偏概全。
4. 检查比较类论述的基准是否一致。

请按以下 JSON 格式输出（不要有任何额外文字）：
{{
  "logical": {{
    "score": float,
    "issues": [
      {{
        "severity": "critical|major|minor",
        "description": "string",
        "location": "string",
        "fix_type": "in_place|search|removal",
        "evidence": "string"
      }}
    ]
  }}
}}

--- 研究报告 ---
Query: {query}

Content:
{content}
"""

# --- 组合 3: 来源可信度 + 覆盖完整度（都评估信息质量）---
PROMPT_SOURCE_COVERAGE = """请对以下研究报告进行【来源可信度与覆盖完整度】综合评分。

你需要从两个子维度分别评估：

## 子维度 A: 来源可信度 (source_credibility)
评分标准（0-10分）：
- 10分：所有来源均为高权威的一手资料（顶级期刊、政府官网、官方财报），且时效性强。
- 7-9分：以权威二手资料为主，个别来源时效稍旧但非核心数据。
- 4-6分：混有低权威来源（匿名论坛、未验证自媒体）且未做交叉验证。
- 1-3分：主要依赖低质量来源，或存在来源循环引用。
- 0分：无来源或来源完全不可信。

## 子维度 B: 覆盖完整度 (coverage)
评分标准（0-10分）：
- 10分：完全覆盖 query 要求的所有子话题，无重要遗漏，正反方观点均衡呈现。
- 7-9分：覆盖了主要子话题，个别边缘视角缺失，但不影响核心结论。
- 4-6分：遗漏了 query 隐含的关键子话题，或只呈现单方面观点。
- 1-3分：严重跑题或大量子话题未覆盖。
- 0分：完全未回答 query。

审查要求：
1. 评估每个 source 的域名权威性和内容类型（一手数据/分析报道/用户生成内容）。
2. 将 query 拆解为应覆盖的子话题列表，逐一检查是否在报告中得到充分讨论。
3. 检查 sources 是否与 query 主题相关——若返回的 sources 与 query 主题无关，必须标记为 critical。
4. 检查是否存在立场偏差（只呈现正方而忽略反方）。

请按以下 JSON 格式输出（不要有任何额外文字）：
{{
  "source_credibility": {{
    "score": float,
    "issues": [
      {{
        "severity": "critical|major|minor",
        "description": "string",
        "location": "string",
        "fix_type": "in_place|search|removal",
        "evidence": "string"
      }}
    ]
  }},
  "coverage": {{
    "score": float,
    "issues": [
      {{
        "severity": "critical|major|minor",
        "description": "string",
        "location": "string",
        "fix_type": "in_place|search|removal",
        "evidence": "string"
      }}
    ]
  }}
}}

--- 原始问题 ---
{query}

--- 研究报告 ---
{content}

--- 来源列表 ---
{sources}
"""


# ============================================================================
# 维度 → (子维度 key, Prompt 模板, 是否需要 sources) 映射
# ============================================================================

# 每个 entry: (display_name, prompt_template, needs_sources)
_COMBINED_DIMENSIONS: list[tuple[str, str, bool]] = [
    ("factual_hallucination", PROMPT_FACTUAL_HALLUCINATION, True),
    ("logical", PROMPT_LOGICAL, False),
    ("source_coverage", PROMPT_SOURCE_COVERAGE, True),
]

# 子维度 key → Dimension 枚举映射
_SUB_DIMENSION_MAP: dict[str, Dimension] = {
    "factual_accuracy": Dimension.FACTUAL,
    "hallucination": Dimension.HALLUCINATION,
    "logical": Dimension.LOGICAL,
    "source_credibility": Dimension.SOURCE_CREDIBILITY,
    "coverage": Dimension.COVERAGE,
}


# ============================================================================
# Red Agent 实现
# ============================================================================

class RedAgent:
    """Red Agent — 多维度对抗攻击器（并行优化版）。

    Attributes:
        policy: VLLMPolicy 实例，提供 LLM 调用能力。
        max_tokens: 单次评估的最大输出 token 数。
    """

    def __init__(self, policy, max_tokens: int = 2048):
        self.policy = policy
        self.max_tokens = max_tokens

    @trace_agent(name="red_agent.attack", tags=["m5", "red", "adversarial"])
    async def attack(self, report: ResearchReport) -> RedVerdict:
        """对研究报告执行多维度攻击（3 路并行）。

        执行流程：
        1. 准备截断内容和来源文本。
        2. asyncio.gather 并行调用 3 个组合维度评估 Prompt。
        3. 解析每个组合输出，提取各子维度的分数和 issues。
        4. 汇总生成 RedVerdict。

        Args:
            report: 待审查的研究报告。

        Returns:
            RedVerdict: 包含五维度分数、overall_score 和 issues 列表。
        """
        # 准备共享数据
        content_truncated = report.content[:4000] if len(report.content) > 4000 else report.content
        sources_text = self._format_sources(report.sources, max_items=15)

        # 一次性设置 max_tokens（消除竞态条件）
        old_max = getattr(self.policy, "max_tokens", None)
        if old_max is not None:
            self.policy.max_tokens = self.max_tokens

        try:
            # 3 路并行执行
            tasks = [
                self._eval_dimension(name, prompt, report.query, content_truncated,
                                     sources_text if needs_sources else "")
                for name, prompt, needs_sources in _COMBINED_DIMENSIONS
            ]
            results = await asyncio.gather(*tasks)
        finally:
            # 恢复原始 max_tokens
            if old_max is not None:
                self.policy.max_tokens = old_max

        # 汇总结果
        dimension_scores: dict[Dimension, float] = {}
        all_issues: list[Issue] = []
        raw_feedbacks: list[str] = []

        for name, raw, sub_results in results:
            raw_feedbacks.append(f"[{name}]\n{raw}\n")
            for sub_key, score, issues in sub_results:
                dim = _SUB_DIMENSION_MAP.get(sub_key)
                if dim is not None:
                    dimension_scores[dim] = score
                    all_issues.extend(issues)

        # 确保所有 5 个维度都有分数（解析失败时回退到 5.0）
        for dim in Dimension:
            if dim not in dimension_scores:
                dimension_scores[dim] = 5.0

        overall = VerdictEngine.compute_overall(dimension_scores)
        return RedVerdict(
            dimension_scores=dimension_scores,
            overall_score=overall,
            issues=all_issues,
            raw_feedback="\n".join(raw_feedbacks),
        )

    async def _eval_dimension(
        self, name: str, prompt_template: str, query: str,
        content: str, sources: str,
    ) -> tuple[str, str, list[tuple[str, float, list[Issue]]]]:
        """评估单个组合维度（在 asyncio.gather 中并行执行）。

        带 Schema 校验：LLM 输出格式不合法时自动回传错误并重试（最多 2 次）。

        Returns:
            (dimension_name, raw_response, [(sub_key, score, issues), ...])
        """
        prompt = prompt_template
        prompt = prompt.replace("{query}", query)
        prompt = prompt.replace("{content}", content)
        prompt = prompt.replace("{sources}", sources)

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_RED_AGENT},
            {"role": "user", "content": prompt},
        ]

        # 确定该组合维度应包含的子维度 key（用于校验）
        if name == "factual_hallucination":
            expected_keys = ["factual_accuracy", "hallucination"]
        elif name == "logical":
            expected_keys = ["logical"]
        elif name == "source_coverage":
            expected_keys = ["source_credibility", "coverage"]
        else:
            expected_keys = []

        # Schema 校验 + 重试
        from src.utils.schema_validator import (
            validate_red_agent_output,
            retry_with_schema_feedback,
        )

        async def _call_policy(msgs):
            return self.policy(msgs)

        data = await retry_with_schema_feedback(
            policy_call=_call_policy,
            messages=messages,
            validate_fn=lambda d: validate_red_agent_output(d, expected_keys),
            max_retries=2,
        )

        # Schema 校验全部失败 → 降级：用原始解析
        if data is None:
            try:
                resp = self.policy(messages[:2])  # 用初始 messages 重试最后一次
                raw = resp.content or ""
            except Exception:
                raw = "{}"
            parsed = self._parse_combined_json(raw)
        else:
            # 校验通过，但需要确保缺失的子维度有兜底分数
            raw = str(data)
            parsed = {}
            for key in expected_keys:
                sub = data.get(key, {})
                if isinstance(sub, dict):
                    parsed[key] = sub
                else:
                    parsed[key] = {"score": 5.0, "issues": []}

        # 解析各子维度
        sub_results: list[tuple[str, float, list[Issue]]] = []
        for sub_key, sub_data in parsed.items():
            dim = _SUB_DIMENSION_MAP.get(sub_key)
            if dim is None:
                continue
            score = float(sub_data.get("score", 5.0))
            score = max(0.0, min(10.0, score))
            issues = self._extract_issues(sub_data.get("issues", []), dim)
            sub_results.append((sub_key, score, issues))

        return (name, raw, sub_results)

    # ------------------------------------------------------------------
    # JSON 解析
    # ------------------------------------------------------------------

    def _parse_combined_json(self, raw: str) -> dict[str, dict]:
        """解析组合 prompt 的 JSON 输出 → {sub_key: {score, issues}}。

        兼容策略：
        1. 直接 json.loads
        2. 提取 ```json 代码块
        3. 正则提取 { ... } 块
        4. 修复常见 JSON 错误后重试
        """
        raw = raw.strip()
        if not raw:
            return {}

        for attempt in range(4):
            candidate = raw
            if attempt == 1:
                m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
                candidate = m.group(1).strip() if m else raw
            elif attempt == 2:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                candidate = m.group(0) if m else raw
            elif attempt == 3:
                candidate = self._fix_common_json_errors(raw) or raw

            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    # 过滤出子维度数据（只保留含 score 字段的 dict）
                    result = {}
                    for key, val in data.items():
                        if isinstance(val, dict) and "score" in val:
                            result[key] = val
                    if result:
                        return result
            except (json.JSONDecodeError, TypeError):
                continue

        return {}

    def _extract_issues(self, raw_issues: list, dimension: Dimension) -> list[Issue]:
        """从解析后的 issues 列表中提取 Issue 对象。"""
        sev_map = {
            "critical": Severity.CRITICAL, "严重": Severity.CRITICAL,
            "major": Severity.MAJOR, "重要": Severity.MAJOR, "较大": Severity.MAJOR,
            "minor": Severity.MINOR, "轻微": Severity.MINOR, "一般": Severity.MINOR,
        }
        fix_map = {
            "in_place": FixType.IN_PLACE, "就地修复": FixType.IN_PLACE, "修正": FixType.IN_PLACE,
            "search": FixType.SUPPLEMENTARY, "supplementary": FixType.SUPPLEMENTARY,
            "补充搜索": FixType.SUPPLEMENTARY, "搜索": FixType.SUPPLEMENTARY,
            "removal": FixType.REMOVAL, "删除": FixType.REMOVAL, "移除": FixType.REMOVAL,
        }

        if isinstance(raw_issues, str):
            return []

        issues: list[Issue] = []
        for item in raw_issues:
            if not isinstance(item, dict):
                continue
            try:
                sev_raw = str(item.get("severity", "minor")).lower().strip()
                fix_raw = str(item.get("fix_type", "in_place")).lower().strip()
                issues.append(Issue(
                    severity=sev_map.get(sev_raw, Severity.MINOR),
                    dimension=dimension,
                    description=str(item.get("description", "")),
                    location=str(item.get("location", "")),
                    fix_type=fix_map.get(fix_raw, FixType.IN_PLACE),
                    evidence=str(item.get("evidence", "")),
                ))
            except (ValueError, KeyError, TypeError):
                continue
        return issues

    def _fix_common_json_errors(self, raw: str) -> str | None:
        """修复常见的 JSON 格式错误。"""
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        text = raw[start:end + 1]
        text = re.sub(r"//.*?\n", "\n", text)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        text = text.replace("'", '"')
        text = re.sub(r",(\s*[}\]])", r"\1", text)
        return text

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _format_sources(self, sources: list[dict], max_items: int = 15) -> str:
        """将来源列表格式化为文本，供 Prompt 使用。"""
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
