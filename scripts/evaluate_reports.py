#!/usr/bin/env python3
"""Evaluate saved reports or a labelled report regression set, without rerunning research."""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.judge import LLMJudge
from src.core.report_evaluation import REPORT_EVALUATOR_VERSION, report_contract, normalize_evidence
from src.utils.execution import bounded, deadline_scope, run_sync
from src.utils.tracing import research_trace, flush_traces
from evaluation.experiments import source_revision, infrastructure_error, EvaluationInfrastructureError


def check_expectations(row: dict, rows: dict) -> list[dict]:
    expected, result, checks = row.get("expected", {}), row.get("evaluation", {}), []
    def check(name, actual, target, passed):
        checks.append({"name": name, "actual": actual, "expected": target, "passed": bool(passed)})
    if "constraint_pass" in expected:
        actual = row["contract"]["constraint_pass"]
        check("constraint_pass", actual, expected["constraint_pass"], actual == expected["constraint_pass"])
    if result:
        check("judge_valid", "error" not in result, True, "error" not in result)
        if "error" in result:
            return checks
        verdicts = {f["id"]: f["verdict"] for f in result.get("facts", [])}
        for key, target in expected.get("fact_verdicts", {}).items():
            check(key, verdicts.get(key), target, verdicts.get(key) == target)
        for key, target in expected.get("max_dimensions", {}).items():
            actual = result["dimensions"][key]["score"]
            check(key + "_max", actual, target, actual <= target)
        if "min_average" in expected:
            check("min_average", result["average"], expected["min_average"], result["average"] >= expected["min_average"])
        if "lower_than" in expected:
            other = rows.get(expected["lower_than"], {}).get("evaluation", {})
            check("lower_than", result["average"], other.get("average"),
                  "average" in other and result["average"] < other["average"])
    return checks


async def run(args):
    if args.regression:
        data = json.loads(args.regression.read_text(encoding="utf-8"))
        cases = [{**c, "question": data["questions"][c["question_id"]], "evidence": c.get("evidence", data.get("documents", []))} for c in data["cases"]]
        source_hash = hashlib.sha256(args.regression.read_bytes()).hexdigest()
    else:
        dataset = json.loads((args.experiment_dir / "dataset.json").read_text(encoding="utf-8"))
        questions = {q["id"]: q for q in dataset["examples"]}
        cases = []
        for line in (args.experiment_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
            old = json.loads(line)
            directory = args.experiment_dir / old["report_path"]
            body_path = directory.parent / "report_body.md"
            if body_path.exists():
                body = body_path.read_text(encoding="utf-8")
            else:
                # Historical report.md contains a question/metadata wrapper. Never
                # score that wrapper, which leaks reference keywords into coverage.
                text = directory.read_text(encoding="utf-8")
                marker = "\n---\n\n"
                if marker not in text:
                    raise ValueError(f"Cannot identify body in legacy artifact {directory}")
                body = text.split(marker, 1)[1].split("\n---\n\n## 元信息", 1)[0].rstrip()
            q = questions[old["question_id"]]
            cases.append({"id": old["sample_id"], "question_id": q["id"], "report": body, "question": q,
                          "evidence": q.get("evidence", dataset.get("documents", [])), "original_result": old})
        source_hash = hashlib.sha256((args.experiment_dir / "results.jsonl").read_bytes()).hexdigest()
    output = args.output_dir or ROOT / "outputs/evaluation" / ("report_review_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"evaluator_version": REPORT_EVALUATOR_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
                "judge_requested": args.judge, "research_rerun": False, "backend": args.backend,
                "max_tokens": args.max_tokens, "source_artifact_hash": source_hash, **source_revision(),
                "status": "running", "num_cases": len(cases)}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report evaluation artifacts: {output}", flush=True)
    judge = LLMJudge(backend=args.backend, max_tokens=args.max_tokens,
                     thinking_mode=getattr(args, "thinking_mode", None)) if args.judge else None
    rows = []
    try:
        for case in cases:
            # External IDs are not trusted filesystem paths.
            case_key = hashlib.sha256(case["id"].encode()).hexdigest()[:16]
            directory = output / "samples" / case_key
            directory.mkdir(parents=True)
            question = case["question"]
            evidence = normalize_evidence(case["evidence"])
            row = {"id": case["id"], "question_id": case["question_id"], "expected": case.get("expected", {}),
                   "contract": report_contract(case["report"], question.get("requirements")), "artifact_dir": str(directory.relative_to(output))}
            (directory / "input.json").write_text(json.dumps({**case, "evidence": evidence}, ensure_ascii=False, indent=2), encoding="utf-8")
            started = time.monotonic()
            if judge:
                with research_trace({"phase": "report_regression", "example_id": case["id"], "evaluator_version": REPORT_EVALUATOR_VERSION}) as trace:
                    try:
                        with deadline_scope(seconds=180):
                            row["evaluation"] = await bounded(run_sync(judge.score_report, case["report"], question["query"],
                                question.get("ground_truth"), question.get("requirements", {}), evidence))
                    except Exception as exc:
                        row["evaluation"] = {"error": f"{type(exc).__name__}: {exc}"}
                snapshot = trace.snapshot()
                row["usage"] = snapshot["usage"]
                (directory / "judge_trace.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
            rows.append(row)
            (directory / "result.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{len(rows)}/{len(cases)} {row['id']}: contract={row['contract']['constraint_pass']} quality={row.get('evaluation', {}).get('average')} error={row.get('evaluation', {}).get('error')}", flush=True)
            if judge:
                resource = infrastructure_error({"judge": row.get("evaluation", {})}, snapshot)
                if resource:
                    raise EvaluationInfrastructureError(resource)
    finally:
        index = {r["id"]: r for r in rows}
        for row in rows:
            row["checks"] = check_expectations(row, index)
        checks = [c for r in rows for c in r["checks"]]
        failed = [{"id": r["id"], **c} for r in rows for c in r["checks"] if not c["passed"]]
        summary = {"num_completed": len(rows), "num_expected": len(cases),
                   "judge_valid": sum("evaluation" in r and "error" not in r["evaluation"] for r in rows),
                   "expectation_checks": len(checks), "expectation_checks_passed": sum(c["passed"] for c in checks),
                   "failed_expectations": failed,
                   "all_labelled_expectations_evaluated": bool(args.judge and len(rows) == len(cases)),
                   "total_tokens": sum(r.get("usage", {}).get("total_tokens", 0) for r in rows),
                   "scope": "Small development diagnostic set, not held-out evaluator accuracy"}
        (output / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["status"] = "completed" if len(rows) == len(cases) else "interrupted"
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        flush_traces()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failed or summary["num_completed"] != summary["num_expected"] or (args.judge and summary["judge_valid"] != len(cases)) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--regression", type=Path)
    group.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--judge", action="store_true", help="Call the configured real model; without this only contracts are checked")
    parser.add_argument("--backend", default="deepseek")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--thinking-mode", choices=["enabled", "disabled"])
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
