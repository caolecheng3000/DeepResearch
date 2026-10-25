#!/usr/bin/env python3
"""Run paired research experiments. Adversarial processing is OFF unless explicitly requested."""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.experiments import load_dataset, run_experiment, export_review_queue
from src.core.runner import load_config, setup_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--dataset", default=str(ROOT / "evaluation/data/research_regression_v1.json"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--variants", nargs="+", choices=["full", "no_compressor", "no_memory", "no_adversarial", "no_recon", "serial_workers"], default=["full", "no_compressor"])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--langsmith", action="store_true", help="Publish dataset/experiments using locally configured credentials")
    parser.add_argument("--enable-adversarial", action="store_true", help="Explicitly allow the slow adversarial stage")
    parser.add_argument("--review-from", help="Only export human review queue from an existing experiment directory")
    args = parser.parse_args()
    if args.review_from:
        print(export_review_queue(args.review_from))
        return
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    config = load_config(args.config)
    # Experiment-only override; production defaults are preserved.
    config.setdefault("adversarial", {})["enabled"] = args.enable_adversarial
    config.setdefault("evolution", {})["enabled"] = False
    dataset = load_dataset(args.dataset)
    if args.limit:
        dataset["examples"] = dataset["examples"][:args.limit]
        from src.utils.tracing import config_fingerprint
        dataset["content_hash"] = config_fingerprint({k: v for k, v in dataset.items() if k != "content_hash"})
    available = {"full": {}, "no_compressor": {"compressor": {"enable_multilevel": False}},
                 "no_memory": {"memory": {"enabled": False}},
                 "no_adversarial": {"adversarial": {"enabled": False}},
                 "no_recon": {"planner": {"search_calls": 0}},
                 "serial_workers": {"orchestrator": {"max_concurrent": 1}}}
    variants = {name: available[name] for name in args.variants}
    if args.enable_adversarial and config["adversarial"].get("max_rounds", 1) < 1:
        parser.error("--enable-adversarial requires adversarial.max_rounds >= 1 in the config")
    if config.get("tools", {}).get("snapshot_path"):
        path = Path(config["tools"]["snapshot_path"])
        config["tools"]["snapshot_path"] = str((ROOT / path).resolve() if not path.is_absolute() else path)
    judge = None
    if args.judge:
        from src.core.judge import LLMJudge
        sampling = config.get("model", {}).get("backend_sampling", {}).get("modules", {}).get("judge", {})
        judge = LLMJudge(backend="deepseek", **sampling)
    output = args.output_dir or str(ROOT / "outputs/evaluation" / ("experiment_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")))
    setup_logging()
    result = asyncio.run(run_experiment(config, dataset, output, variants, args.repetitions, judge, args.langsmith))
    export_review_queue(output)
    print(json.dumps({"output_dir": output, "summary": result["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
