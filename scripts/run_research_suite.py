"""Plan v5 research experiments offline; --execute explicitly starts model calls.

The historical run_capability_suite.py remains frozen to its v3 protocol.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VARIANTS = {
    "no_adversarial": {"adversarial": {"enabled": False}},
    "no_compressor": {"compressor": {"enable_multilevel": False}},
    "no_memory": {"memory": {"enabled": False}},
    "no_recon": {"planner": {"search_calls": 0}},
    "serial_workers": {"orchestrator": {"max_concurrent": 1}},
}


def prepare(config_path: Path, pilot: bool = False):
    from evaluation.experiments import load_dataset
    from src.core.runner import load_config
    dataset_path = ROOT / "evaluation/data/research_deepresearch_v5.json"
    config = load_config(str(config_path))
    configured = Path(config.get("tools", {}).get("snapshot_path", ""))
    configured = configured if configured.is_absolute() else ROOT / configured
    if configured.resolve() != dataset_path.resolve():
        raise ValueError("v5 runner requires the v5 corpus; cannot mix a new question set with old snapshot tools")
    config["tools"]["snapshot_path"] = str(dataset_path)
    adv = config.get("adversarial", {})
    if (not adv.get("enabled") or type(adv.get("max_rounds")) is not int
            or not 1 <= adv["max_rounds"] <= 3 or type(adv.get("max_issues_per_round")) is not int
            or not 1 <= adv["max_issues_per_round"] <= 5):
        raise ValueError("Require adversarial enabled, <=3 rounds and <=5 TOTAL Blue issues per round")
    dataset = load_dataset(dataset_path)
    anchors = load_dataset(ROOT / "evaluation/data/research_deepresearch_v5_ablation.json")
    if pilot:
        dataset["examples"] = [q for q in dataset["examples"] if q["split"] == "development"]
        from src.utils.tracing import config_fingerprint
        dataset["content_hash"] = config_fingerprint({k: v for k, v in dataset.items() if k != "content_hash"})
    plan = {"status": "plan_only", "dataset_version": dataset["version"],
            "annotation_status": dataset["status"], "dataset_hash": dataset["content_hash"],
            "research_cases": len(dataset["examples"]), "source_documents": len(dataset["documents"]),
            "paired_anchors": 0 if pilot else len(anchors["examples"]),
            "control_variants": [] if pilot else list(VARIANTS),
            "planned_research_attempts": len(dataset["examples"]) + (0 if pilot else len(anchors["examples"]) * len(VARIANTS)),
            "judge_calls": "only with --execute --judge; additional report chunks may require multiple calls",
            "annotation_note": "review draft, not a human-certified held-out benchmark",
            "max_adversarial_rounds": adv["max_rounds"], "max_total_blue_issues_per_round": adv["max_issues_per_round"],
            "model_calls_started": False}
    return config, dataset, anchors, plan


async def execute(args, config, dataset, anchors, plan):
    from evaluation.experiments import run_experiment, export_review_queue, summarize
    from src.core.runner import setup_logging
    if args.output_dir is None:
        raise ValueError("--execute requires --output-dir for new artifacts")
    if args.output_dir.exists():
        raise ValueError("Use a new output directory; historical results are never overwritten")
    judge = None
    if args.judge:
        from src.core.judge import LLMJudge
        judge = LLMJudge(backend="deepseek", **config["model"]["backend_sampling"]["modules"]["judge"])
    args.output_dir.mkdir(parents=True)
    setup_logging()
    manifest = {**plan, "status": "running", "model_calls_started": True}
    def save():
        (args.output_dir / "suite_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    save()
    try:
        research = await run_experiment(config, dataset, args.output_dir / "research", {"full": {}}, judge=judge)
        export_review_queue(args.output_dir / "research")
        if not args.pilot:
            controls = await run_experiment(copy.deepcopy(config), anchors, args.output_dir / "ablations", VARIANTS, judge=judge)
            export_review_queue(args.output_dir / "ablations")
            anchor_ids = {q["id"] for q in anchors["examples"]}
            baseline = [r for r in research["details"] if r["question_id"] in anchor_ids and r["variant"] == "full"]
            paired = summarize(baseline + controls["details"])
            paired["scope"] = "Six preselected matched briefs; full baseline reused from research stage, no extra model calls; exploratory, single-run comparison."
            (args.output_dir / "paired_ablation_summary.json").write_text(json.dumps(paired, ensure_ascii=False, indent=2) + "\n")
        manifest["status"] = "completed"
    except BaseException:
        manifest["status"] = "interrupted"
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/evaluation_research.yaml")
    parser.add_argument("--pilot", action="store_true", help="Six development briefs; not formal held-out scores")
    parser.add_argument("--execute", action="store_true", help="Start paid model research; omitted means offline plan only")
    parser.add_argument("--judge", action="store_true", help="Also run paid evidence-aware report judging")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config, dataset, anchors, plan = prepare(args.config, args.pilot)
    if args.execute:
        asyncio.run(execute(args, config, dataset, anchors, plan))
    else:
        print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
