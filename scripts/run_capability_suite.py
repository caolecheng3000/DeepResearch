"""Run the versioned research suite, matched module controls and Judge diagnostics."""
import argparse
import asyncio
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.experiments import load_dataset,run_experiment,export_review_queue,source_revision
from src.core.runner import load_config,setup_logging
from src.core.judge import LLMJudge
from scripts.analyze_capability_suite import collect,review_before,analyze


async def run(args):
    cfg=load_config(args.config)
    adv=cfg.get("adversarial",{})
    if not (adv.get("enabled") and type(adv.get("max_rounds")) is int and 1<=adv["max_rounds"]<=3
            and type(adv.get("max_issues_per_round")) is int and 1<=adv["max_issues_per_round"]<=5):
        raise ValueError("This suite requires adversarial ON, at most 3 rounds and 5 TOTAL Blue issues per round")
    judge=LLMJudge(backend="deepseek",**cfg["model"]["backend_sampling"]["modules"]["judge"])
    dataset=load_dataset(ROOT/"evaluation/data/research_capability_v3.json")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    plan={"status":"running","pilot":args.pilot,"max_rounds":adv["max_rounds"],
          "max_total_blue_issues_per_round":adv["max_issues_per_round"],"source":source_revision(),
          "planned_research_attempts":2 if args.pilot else 132,"planned_judge_regression_cases":0 if args.pilot else 48}
    if args.resume_from:
        plan["resume_from"]=str(args.resume_from.resolve())
    def checkpoint(stage):
        prior=args.resume_from/stage if args.resume_from else None
        return prior if prior is not None and (prior/"manifest.json").exists() else None
    plan_path=args.output_dir/"suite_manifest.json"
    plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    def config_for(path):
        config=copy.deepcopy(cfg);config.setdefault("tools",{})["snapshot_path"]=str(path)
        return config
    try:
        if args.pilot:
            dataset["examples"]=[q for q in dataset["examples"] if q["split"]=="development" and q["category"] in ("numeric","long_context")]
            result=await run_experiment(config_for(ROOT/"evaluation/data/research_capability_v3.json"),dataset,
                args.output_dir/"controlled",{"full":{}},judge=judge,resume_from=checkpoint("controlled"))
            export_review_queue(args.output_dir/"controlled")
            print(json.dumps(result["summary"],ensure_ascii=False),flush=True)
        else:
            await run_experiment(config_for(ROOT/"evaluation/data/research_capability_v3.json"),dataset,
                args.output_dir/"controlled",{"full":{}},judge=judge,resume_from=checkpoint("controlled"))
            export_review_queue(args.output_dir/"controlled")
            ablation=load_dataset(ROOT/"evaluation/data/research_capability_v3_ablation.json")
            variants={"no_adversarial":{"adversarial":{"enabled":False}},
                "no_compressor":{"compressor":{"enable_multilevel":False}},
                "no_memory":{"memory":{"enabled":False}},
                "no_recon":{"planner":{"search_calls":0}},
                "serial_workers":{"orchestrator":{"max_concurrent":1}}}
            await run_experiment(config_for(ROOT/"evaluation/data/research_capability_v3.json"),ablation,
                args.output_dir/"ablations",variants,judge=judge,resume_from=checkpoint("ablations"))
            export_review_queue(args.output_dir/"ablations")
            hotpot=load_dataset(ROOT/"evaluation/data/research_hotpotqa_v1.json")
            await run_experiment(config_for(ROOT/"evaluation/data/research_hotpotqa_v1.json"),hotpot,
                args.output_dir/"hotpot",{"full":{}},judge=judge,resume_from=checkpoint("hotpot"))
            export_review_queue(args.output_dir/"hotpot")
            from scripts.evaluate_reports import run as regression_run
            code=await regression_run(SimpleNamespace(regression=ROOT/"evaluation/data/report_judge_regression_v2.json",
                experiment_dir=None,output_dir=args.output_dir/"judge_regression",judge=True,
                backend="deepseek",max_tokens=8192,thinking_mode="disabled"))
            plan["judge_regression_exit_code"]=code
            review_before(args.output_dir,collect(args.output_dir),judge)
        plan["status"]="completed"
    except BaseException:
        plan["status"]="interrupted"
        raise
    finally:
        plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+"\n")
        analyze(args.output_dir)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(ROOT/"configs/evaluation_capability.yaml"))
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--pilot",action="store_true",help="Two development diagnostics; not part of formal scores")
    parser.add_argument("--resume-from",type=Path,help="Reuse successful research only when dataset and configuration match")
    args=parser.parse_args();setup_logging();asyncio.run(run(args))
