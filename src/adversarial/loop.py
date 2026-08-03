"""
M5 Red-Blue 对抗降噪循环 — 主控制器

AdversarialLoop 驱动 Red Agent → Blue Agent → 评分的完整对抗流程，
具备死循环检测、震荡检测、收敛判断等鲁棒机制。

设计决策：
1. 维护 resolved_issues 集合：已修复的 issue 如果在新轮次中重新出现，判定为震荡。
2. 收敛条件三选一：round >= 3 / overall >= 8.0 / Δscore < 0.3。
3. 每轮记录完整评分历史，便于后续分析和审计。
"""
from __future__ import annotations

import copy
import logging
from typing import Any

from src.adversarial.blue_agent import BlueAgent
from src.adversarial.red_agent import RedAgent
from src.adversarial.verdict import (
    Dimension,
    FixOperation,
    Issue,
    RedVerdict,
    VerdictEngine,
)
from src.orchestrator.schemas import ResearchReport
from src.utils.tracing import trace_chain

# Blue Agent 的禁止内容模式（用于输出验证）
_FORBIDDEN_PATTERNS = [
    "由于搜索api", "由于api", "api密钥无效", "api密钥",
    "搜索api", "博查ai", "密钥无效", "无法获取任何搜索",
    "未能获取任何", "建议用户检查", "检查搜索工具配置",
    "本报告内容为空", "无法补充来源", "由于搜索API错误",
    "请读者谅解", "无法完成研究",
]


__all__ = ["AdversarialLoop"]

logger = logging.getLogger(__name__)


class AdversarialLoop:
    """Red-Blue 对抗降噪循环主控制器。

    Attributes:
        red_agent: Red Agent 实例，负责攻击。
        blue_agent: Blue Agent 实例，负责修复。
        policy: 用于 self_verify 或辅助评分的策略对象（可选）。
        max_rounds: 硬上限轮数。
        score_threshold: 综合分达标阈值。
        delta_threshold: 轮间变化量收敛阈值。
    """

    def __init__(
        self,
        red_agent: RedAgent,
        blue_agent: BlueAgent,
        policy: Any | None = None,
        max_rounds: int = 3,
        score_threshold: float = 8.0,
        delta_threshold: float = 0.3,
    ):
        self.red_agent = red_agent
        self.blue_agent = blue_agent
        self.policy = policy
        self.max_rounds = max(max_rounds, 1)
        self.score_threshold = score_threshold
        self.delta_threshold = delta_threshold

    @trace_chain(name="adversarial_loop.run", tags=["m5", "loop", "adversarial"])
    async def run(
        self, report: ResearchReport
    ) -> tuple[ResearchReport, list[dict[str, Any]]]:
        """运行完整的对抗降噪循环。

        流程：
        1. 每轮用 Red Agent 攻击当前报告。
        2. Blue Agent 根据 Verdict 修复。
        3. 记录本轮评分和修复操作。
        4. 检查收敛条件或震荡/死循环。
        5. 返回最终报告和完整历史。

        Args:
            report: 初始研究报告（不会被修改，内部深拷贝）。

        Returns:
            (修复后的报告, 每轮评分记录列表)
            每轮记录包含: round, dimension_scores, overall_score, delta, issues_count,
            fix_operations, resolved_count, oscillation_detected, stop_reason
        """
        current = copy.deepcopy(report)
        original_content_len = len(report.content) if report.content else 0  # 保存原始长度
        history: list[dict[str, Any]] = []
        prev_scores: dict[Dimension, float] | None = None
        resolved_issues: set[Issue] = set()  # 已修复的 issue 集合
        oscillation_detected = False
        stop_reason = ""
        best_report = copy.deepcopy(report)
        best_score = 0.0
        degrading_count = 0  # 连续未改善计数

        for round_idx in range(1, self.max_rounds + 1):
            logger.info(f"[AdversarialLoop] Round {round_idx} starting...")

            # ---- Step 1: Red Attack ----
            verdict = await self.red_agent.attack(current)
            logger.info(
                f"[AdversarialLoop] Red attack done: overall={verdict.overall_score:.2f}, "
                f"issues={len(verdict.issues)}"
            )

            # ---- Step 2: 震荡检测 ----
            # 如果当前 issues 中有已修复过的 issue 重新出现，判定震荡
            reappeared = resolved_issues.intersection(set(verdict.issues))
            if reappeared:
                oscillation_detected = True
                logger.warning(
                    f"[AdversarialLoop] Oscillation detected at round {round_idx}: "
                    f"{len(reappeared)} previously resolved issues reappeared."
                )
                stop_reason = f"oscillation_at_round_{round_idx}"
                history.append(self._build_round_record(
                    round_idx, verdict, [], len(reappeared), oscillation_detected, stop_reason
                ))
                break

            # ---- Step 3: Blue Defend ----
            fixed_report, operations = await self.blue_agent.defend(current, verdict)
            logger.info(
                f"[AdversarialLoop] Blue defend done: operations={len(operations)}"
            )

            # 将本轮被修复的 issues 加入 resolved 集合
            for op in operations:
                if op.success:
                    resolved_issues.add(op.issue)

            # ---- Step 4: 计算 delta ----
            delta = 0.0
            if prev_scores is not None:
                delta = VerdictEngine.compute_delta(prev_scores, verdict.dimension_scores)
            prev_scores = copy.deepcopy(verdict.dimension_scores)

            # ---- Step 4.5: 跟踪最佳版本，检测分数退化 ----
            if verdict.overall_score > best_score:
                best_score = verdict.overall_score
                best_report = copy.deepcopy(fixed_report)
                degrading_count = 0
            else:
                degrading_count += 1

            # 连续 3 轮未改善 → 判定为不收敛，停止
            if degrading_count >= 3:
                stop_reason = f"degrading({degrading_count}_rounds_no_improvement)"
                logger.warning(
                    f"[AdversarialLoop] Stopping: score not improving for {degrading_count} rounds. "
                    f"Best score was {best_score:.2f} at round {round_idx - degrading_count}."
                )
                history.append(self._build_round_record(
                    round_idx, verdict, operations, len(resolved_issues),
                    oscillation_detected, stop_reason
                ))
                break

            # ---- Step 5: 记录本轮 ----
            stop_reason = self._check_convergence(round_idx, verdict.overall_score, delta)
            record = self._build_round_record(
                round_idx=round_idx,
                verdict=verdict,
                operations=operations,
                resolved_count=len(resolved_issues),
                oscillation=oscillation_detected,
                stop_reason=stop_reason,
            )
            history.append(record)

            # 更新当前报告为修复后的版本
            current = fixed_report
            current.adversarial_rounds = round_idx

            # ---- Step 6: 判断是否停止 ----
            if stop_reason:
                logger.info(f"[AdversarialLoop] Stopping: {stop_reason}")
                break

        # 循环结束后：返回最佳版本而非最后版本
        result = current
        if degrading_count > 0 and best_score > 0:
            logger.info(
                f"[AdversarialLoop] Returning best version (score={best_score:.2f}) "
                f"instead of last version."
            )
            result = best_report

        if history:
            result.final_score = history[-1]["overall_score"]
        result.adversarial_rounds = len(history)

        # 验证返回的报告是否有效（无禁止内容，无截断）
        if not self._is_report_valid(result, original_content_len):
            logger.warning("[AdversarialLoop] Report degraded during adversarial loop. "
                           "Returning original pre-adversarial report.")
            report.adversarial_rounds = 0
            report.final_score = 0.0
            return report, history

        return result, history

    def _is_report_valid(self, report: ResearchReport, original_len: int = 0) -> bool:
        """验证报告内容不包含禁止模式且未被截断。"""
        content = (report.content or "").lower()
        for pat in _FORBIDDEN_PATTERNS:
            if pat in content:
                return False
        # 检测内容被截断：原始 >4000 字但处理后 <60% 原始长度
        current_len = len(report.content or "")
        if original_len > 4000 and current_len < original_len * 0.6:
            logger.warning(
                f"[AdversarialLoop] Truncation detected: "
                f"{original_len}→{current_len} chars"
            )
            return False
        return True

    def _check_convergence(
        self, round_idx: int, overall_score: float, delta: float
    ) -> str:
        """检查是否满足任一收敛条件。

        Returns:
            空字符串表示继续；非空字符串为停止原因。
        """
        if round_idx >= self.max_rounds:
            return f"max_rounds_reached({self.max_rounds})"
        if overall_score >= self.score_threshold:
            return f"score_threshold_met({overall_score:.2f}>={self.score_threshold})"
        # 第一轮没有上一轮做对比，delta 恒为 0，跳过 delta 收敛检测
        if round_idx > 1 and delta < self.delta_threshold:
            return f"delta_converged({delta:.3f}<{self.delta_threshold})"
        return ""

    def _build_round_record(
        self,
        round_idx: int,
        verdict: RedVerdict,
        operations: list[FixOperation],
        resolved_count: int,
        oscillation: bool,
        stop_reason: str,
    ) -> dict[str, Any]:
        """构造单轮记录字典。"""
        return {
            "round": round_idx,
            "dimension_scores": {
                k.value: round(v, 3) for k, v in verdict.dimension_scores.items()
            },
            "overall_score": round(verdict.overall_score, 3),
            "issues_count": len(verdict.issues),
            "fix_operations": [op.to_dict() for op in operations],
            "resolved_count": resolved_count,
            "oscillation_detected": oscillation,
            "stop_reason": stop_reason,
            "raw_feedback": verdict.raw_feedback,
        }
