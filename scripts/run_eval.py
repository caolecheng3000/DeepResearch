#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_eval.py
================================================================================
标准评测集入口脚本（合并了原 run_evaluation.py）。

支持:
  --benchmark research_bench : 自建深度研究评测集（规则指标）
  --benchmark hotpotqa      : 公共多跳 QA 评测集（EM/F1）

Usage:
    python scripts/run_eval.py --benchmark research_bench --num_questions 20
    python scripts/run_eval.py --benchmark hotpotqa --num_questions 100
================================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.runner import initialize_modules, load_config, run_research, setup_logging
from evaluation.benchmarks.research_bench import ResearchBench
from evaluation.benchmarks.hotpotqa import HotpotQABenchmark
from evaluation.report import EvaluationReport


def evaluate_research_bench(
    num_questions: int,
    domain: str | None,
    config: dict,
) -> EvaluationReport:
    """在 ResearchBench 上运行评测。"""
    from evaluation.experiments import run_experiment
    from datetime import datetime
    from pathlib import Path
    bench = ResearchBench()
    questions = bench.get_questions(domain=domain, n=num_questions)
    output = Path("outputs/evaluation") / ("research_bench_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    result = asyncio.run(run_experiment(config,
        {"name": "ResearchBench", "version": "content-hash", "examples": questions},
        output, variants={"full": {}}))
    report = EvaluationReport(name="ResearchBench_Evaluation", num_questions=len(questions))
    for row in result["details"]:
        report.add_detail(row)
    report.set_summary({**result["summary"]["variants"]["full"], "artifact_dir": str(output)})
    return report


def _extract_report_answer(report_text: str) -> str:
    """从研究报告文本中提取可用于 EM/F1 的「答案」正文。

    报告格式（由 src.core.runner._format_report 生成）:
        # 研究报告：<query>
        ---
        # <报告自身标题>
        ## 执行摘要
        ... 正文 ...

    首行永远是通用的包裹标题「# 研究报告：xxx」，不能当作答案。
    这里跳过包裹标题和分隔线，提取报告正文（用于 EM/F1 与深度指标）。
    """
    if not report_text:
        return ""
    lines = [l.rstrip() for l in report_text.strip().split("\n")]
    body_lines = []
    seen_separator = False
    for line in lines:
        stripped = line.strip()
        # 跳过包裹标题
        if not seen_separator and stripped.startswith("# 研究报告"):
            continue
        if stripped == "---":
            seen_separator = True
            continue
        body_lines.append(line)
    # 合并正文，去掉空行首尾
    body = "\n".join(body_lines).strip()
    # 去掉报告自身的一级标题（若存在）
    body = "\n".join(
        l for l in body.split("\n") if not l.strip().startswith("# ")
    ).strip()
    return body


def evaluate_hotpotqa(
    num_questions: int,
    config: dict,
    use_mock: bool = False,
) -> EvaluationReport:
    """在 HotpotQA 上运行评测（深度研究变体：评估完整报告质量）。"""
    logger = logging.getLogger("run_eval")
    bench = HotpotQABenchmark(use_mock=use_mock)
    questions = bench.get_samples(n=num_questions, shuffle=True)
    logger.info(f"HotpotQA 加载 {len(questions)} 道题目")

    from evaluation.experiments import run_experiment
    from datetime import datetime
    from pathlib import Path
    dataset = {"name": "HotpotQA", "version": "mock" if use_mock else "real", "examples": [
        {"id": f"hotpot_{idx}", "query": q["query"], "ground_truth": {str(q["expected_answer"]): "reference answer"}}
        for idx, q in enumerate(questions, 1)]}
    output = Path("outputs/evaluation") / ("hotpot_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    result = asyncio.run(run_experiment(config, dataset, output, variants={"full": {}}))
    report = EvaluationReport(name="HotpotQA_DeepResearch_Evaluation", num_questions=len(questions))
    predictions = []
    for idx, (q, row) in enumerate(zip(questions, result["details"]), 1):
        text = (output / row["report_path"]).read_text(encoding="utf-8")
        prediction = _extract_report_answer(text) if row["status"] == "success" else ""
        gold = q["expected_answer"]
        predictions.append({"query_id": idx, "prediction": prediction, "gold": gold, "report": text})
        report.add_detail({**row, "prediction": prediction, "gold": gold,
                           "depth_metrics": bench.evaluate_report(text, gold) if text else {}})
    metrics = bench.evaluate(predictions, metrics=["em", "f1", "pass@1"])
    report.set_summary({**metrics, **result["summary"]["variants"]["full"], "artifact_dir": str(output)})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepResearch Agent 标准评测脚本")
    parser.add_argument("--benchmark", type=str, choices=["research_bench", "hotpotqa"],
                        required=True, help="评测基准")
    parser.add_argument("--num_questions", type=int, default=20, help="评测题目数量")
    parser.add_argument("--domain", type=str, default=None, help="领域过滤（仅 ResearchBench）")
    parser.add_argument("--use_mock", action="store_true", help="使用内置 mock 数据（仅 HotpotQA，用于流程验证）")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation", help="输出目录")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--enable-adversarial", action="store_true", help="显式启用耗时的对抗阶段，评测默认关闭")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger("main")

    config = load_config(args.config)
    config.setdefault("adversarial", {})["enabled"] = args.enable_adversarial
    logger.info(f"配置加载完成: {args.config or 'configs/default.yaml'}")

    if args.benchmark == "research_bench":
        report = evaluate_research_bench(args.num_questions, args.domain, config)
    elif args.benchmark == "hotpotqa":
        report = evaluate_hotpotqa(args.num_questions, config, use_mock=args.use_mock)
    else:
        raise ValueError(f"未知基准: {args.benchmark}")

    filepath = report.save(args.output_dir)
    logger.info(f"评测报告已保存: {filepath}")

    print("\n" + "=" * 60)
    print("评测摘要")
    print("=" * 60)
    print(json.dumps(report.summary, ensure_ascii=False, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
