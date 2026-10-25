"""Versioned, isolated research experiments with local artifacts and optional LangSmith.

Quality, completion and evaluator failures are separate. Every attempted sample
is persisted immediately; failed runs remain in the end-to-end denominator.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import shutil
from pathlib import Path
import subprocess
import tempfile
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from src.core.runner import initialize_modules, run_research
from src.utils.tracing import config_fingerprint, safe_data, research_trace, flush_traces
from src.utils.execution import deadline_scope, bounded, run_sync
from evaluation.metrics.rule_based import RuleBasedMetrics as Rules
from evaluation.metrics.stats import bootstrap_ci_paired
from src.core.report_evaluation import REPORT_EVALUATOR_VERSION, report_contract, validate_requirements, normalize_evidence

ROOT = Path(__file__).resolve().parents[1]
SCORER_VERSION = "rules-v2-keycoverage"


class EvaluationInfrastructureError(RuntimeError):
    """Stop a suite on account/auth/quota failures; preserve completed artifacts."""


def infrastructure_error(row: dict, trace: dict | None = None) -> str | None:
    for event in (trace or {}).get("events", []):
        if event.get("name") == "llm_error" and event.get("data", {}).get("resource_failure"):
            return f"provider_http_{event['data'].get('status_code')}"
    for message in (row.get("error", ""), row.get("judge", {}).get("error", "")):
        if "[LLM_AUTH_OR_QUOTA_ERROR]" in message:
            return message[:250]
    return None


def load_dataset(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_dataset(data)
    data["content_hash"] = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return data


def validate_dataset(data: dict) -> None:
    examples = data.get("examples", [])
    ids = [q.get("id") for q in examples]
    if not examples or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Dataset needs nonempty examples with unique string IDs")
    for q in examples:
        if not isinstance(q.get("query"), str) or not q["query"].strip():
            raise ValueError("Each example needs a query")
        if not isinstance(q.get("ground_truth", {}), dict) or not isinstance(q.get("expected_topics", []), list):
            raise ValueError("Invalid ground_truth / expected_topics")
        validate_requirements(q.get("requirements"))
        normalize_evidence(q.get("evidence", data.get("documents", [])))


def merge_config(config: dict, overrides: dict) -> dict:
    result = copy.deepcopy(config)
    for key, value in overrides.items():
        result[key] = merge_config(result.get(key, {}), value) if isinstance(value, dict) else copy.deepcopy(value)
    return result


def score_report(report: str, question: dict) -> dict:
    """Cheap stable proxies. Key coverage and citation appearance are not truth verification."""
    hallucination = Rules.hallucination_rate(report)
    metrics = {
        "factual_accuracy": Rules.fact_accuracy(report, question.get("ground_truth", {})),
        "logical_consistency": Rules.logical_consistency(report),
        "citation_coverage": Rules.citation_coverage(report),
        "bias": max(0.0, 1.0 - hallucination),
        "comprehensiveness": Rules.comprehensiveness(report, question.get("expected_topics", [])),
    }
    return {"metrics": metrics, "composite_score": Rules.composite_score(metrics),
            "hallucination_risk_proxy": hallucination, "scorer_version": SCORER_VERSION,
            "metric_semantics": {"factual_accuracy": "reference_key_coverage", "bias": "1_minus_heuristic_risk"}}


def source_revision() -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
    digest = hashlib.sha256()
    for directory in ("src", "evaluation", "scripts"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "source_hash": digest.hexdigest()}


async def run_sample(question: dict, config: dict, output_dir: Path, metadata: dict,
                     judge=None) -> dict:
    """Fresh modules and temporary SQLite per sample; production DB never opened."""
    sample_id = str(uuid.uuid4())
    directory = output_dir / "samples" / sample_id
    directory.mkdir(parents=True)
    row: dict[str, Any] = {"sample_id": sample_id, "question_id": question["id"],
                           **metadata, "status": "failed", "scorer_version": SCORER_VERSION}
    row.update({k: question[k] for k in ("category", "difficulty", "family_id", "split", "ablation") if k in question})
    modules = {}
    started = time.monotonic()
    report_text = ""
    cfg = copy.deepcopy(config)
    with tempfile.TemporaryDirectory(prefix="deepr-eval-") as tmp:
        cfg.setdefault("memory", {})["db_path"] = str(Path(tmp) / "memory.db")
        try:
            modules = initialize_modules(cfg, session_id=sample_id)
            seeds = question.get("memory_seed", [])
            row["memory_seed_count"] = len(seeds)
            store = modules.get("memory_store")
            if store is not None:
                from src.memory.long_term import MemoryEntry
                for i, seed in enumerate(seeds):
                    store.put(MemoryEntry(entry_id=f"prior_{i}", claim=seed["text"], source=seed["url"],
                        confidence=1.0, agent_id="evaluation_prior_session", timestamp=time.time(),
                        evidence_type="primary", embedding=[], topic=question["query"], session_id=sample_id))
            row["models"] = {k: getattr(v, "model_name", "unknown")
                             for k, v in modules.items() if k.endswith("_policy")}
            report_text = await run_research(question["query"], cfg, modules, metadata=metadata)
            row["status"] = modules.get("last_run", {}).get("status", "success")
            if not report_text.strip():
                row["status"] = "failed"
                row["error"] = "empty report"
        except asyncio.CancelledError:
            row["status"] = "cancelled"
            raise
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
            telemetry = modules.get("last_run", {})
            row.update({k: telemetry.get(k) for k in ("research_run_id", "trace_id", "trace_url")})
            row["usage"] = telemetry.get("usage", {})
            resource_error = infrastructure_error(row, telemetry)
            if resource_error:
                row["infrastructure_error"] = resource_error
            row["adversarial_rounds"] = getattr(modules.get("last_report"), "adversarial_rounds", 0)
            spans = telemetry.get("spans", [])
            m5 = [s for s in spans if s.get("name") in ("stage.adversarial", "red_agent.attack", "blue_agent.defend")]
            enabled = config.get("adversarial", {}).get("enabled", True)
            row["adversarial_status"] = ("not_requested" if not enabled else "not_started" if not m5
                else "failed" if any(s.get("status") != "success" for s in m5)
                else "completed" if row["adversarial_rounds"] else "incomplete")
            (directory / "trace.json").write_text(json.dumps(telemetry, ensure_ascii=False, indent=2), encoding="utf-8")
            (directory / "report.md").write_text(report_text, encoding="utf-8")
            # Do not score the wrapper title/query: that would leak reference keys.
            body = getattr(modules.get("last_report"), "content", report_text)
            # Body and reference evidence are complete audit artifacts, independent
            # of bounded trace previews and the human-readable report wrapper.
            (directory / "report_body.md").write_text(body, encoding="utf-8")
            before = getattr(modules.get("orchestrator"), "pre_adversarial_report", None)
            if before is not None:
                (directory / "report_before_adversarial.md").write_text(before.content, encoding="utf-8")
            evidence = normalize_evidence(question.get("evidence", []))
            (directory / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
            row["report_contract"] = report_contract(body, question.get("requirements"))
            row["constraint_pass"] = row["report_contract"]["constraint_pass"]
            row["report_preview"] = body[:2000]
            quality = score_report(body, question)
            row.update(quality)
            row["report_quality_score"] = quality["composite_score"]
            row["composite_score"] = quality["composite_score"] if row["status"] == "success" else 0.0
            row["report_path"] = str((directory / "report.md").relative_to(output_dir))
            row["trace_path"] = str((directory / "trace.json").relative_to(output_dir))
            # A cancelled sample must remain reviewable even as cancellation propagates.
            (directory / "result.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    if judge is not None and body.strip() and row["status"] != "cancelled" and not row.get("infrastructure_error"):
        judge_trace = None
        try:
            with research_trace({**metadata, "phase": "evaluation", "research_run_id_ref": row["research_run_id"]}) as judge_trace:
                with deadline_scope(seconds=120):
                    if hasattr(judge, "score_report"):
                        row["judge"] = await bounded(run_sync(judge.score_report, body, question["query"],
                            question.get("ground_truth"), question.get("requirements", {}), evidence))
                    else:
                        row["judge"] = await bounded(run_sync(judge.score_single, body, question["query"], question.get("ground_truth")))
        except Exception as exc:
            row["judge"] = {"error": f"{type(exc).__name__}: {exc}"}
        row["judge_status"] = "failed" if "error" in row["judge"] else "success"
        if row["judge_status"] == "success" and "report_pass" in row["judge"]:
            row["report_pass"] = row["status"] == "success" and row["judge"]["report_pass"]
        row["report_evaluator_version"] = row["judge"].get("evaluator_version", "legacy-four-dimension")
        row["judge_usage"] = judge_trace.snapshot()["usage"] if judge_trace else {}
        resource_error = infrastructure_error(row, judge_trace.snapshot() if judge_trace else {})
        if resource_error:
            row["infrastructure_error"] = resource_error
        if judge_trace:
            (directory / "judge_trace.json").write_text(json.dumps(judge_trace.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        row["judge_status"] = "not_requested_infrastructure_error" if row.get("infrastructure_error") else "not_requested"
    (directory / "result.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def summarize(rows: list[dict]) -> dict:
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    output = {}
    for variant in variants:
        group = [r for r in rows if r["variant"] == variant]
        succeeded = [r for r in group if r["status"] == "success"]
        judge_ok = [r for r in group if r.get("judge_status") == "success"]
        output[variant] = {
            "num_attempted": len(group), "status_counts": dict(Counter(r["status"] for r in group)),
            "completion_rate": len(succeeded) / len(group),
            "average_composite": sum(r["composite_score"] for r in group) / len(group),
            "average_elapsed_seconds": sum(r["elapsed_seconds"] for r in group) / len(group),
            "total_provider_tokens": sum(r.get("usage", {}).get("total_tokens", 0) for r in group),
            "total_judge_tokens": sum(r.get("judge_usage", {}).get("total_tokens", 0) for r in group),
            "judge_successes": len(judge_ok),
            "average_judge_score": sum(r["judge"]["average"] for r in judge_ok) / len(judge_ok) if judge_ok else None,
            "constraint_pass_rate": sum(r.get("constraint_pass", False) for r in group) / len(group),
            "report_acceptance_rate": sum(r.get("report_pass", False) for r in group) / len(group)
                if any("report_pass" in r for r in group) else None,
            "report_acceptance_evaluated": sum("report_pass" in r for r in group),
            "average_reference_fact_coverage": sum(r["judge"]["reference_fact_coverage"] for r in judge_ok
                if r["judge"].get("reference_fact_coverage") is not None) / max(1, sum(r["judge"].get("reference_fact_coverage") is not None for r in judge_ok))
                if any(r["judge"].get("reference_fact_coverage") is not None for r in judge_ok) else None,
        }
    comparisons = {}
    if variants:
        baseline = variants[0]
        for variant in variants[1:]:
            # Average repetitions within question before paired resampling.
            def per_question(name):
                groups = {}
                for row in rows:
                    if row["variant"] == name:
                        groups.setdefault(row["question_id"], []).append(row["composite_score"])
                return {k: sum(v) / len(v) for k, v in groups.items()}
            a, b = per_question(variant), per_question(baseline)
            common = sorted(a.keys() & b.keys())
            stats = bootstrap_ci_paired([a[k] - b[k] for k in common])
            # Existing helper's p_value is a bootstrap tail fraction, not a
            # calibrated hypothesis-test p-value. Name it accurately here.
            stats["bootstrap_nonpositive_fraction"] = stats.pop("p_value")
            stats["ci_positive"] = stats.pop("significant")
            comparisons[f"{variant}_minus_{baseline}"] = {
                **stats, "num_paired_questions": len(common), "exploratory": len(common) < 10,
                "interpretation": "Small samples verify the pipeline; do not establish a general quality gain." if len(common) < 10 else "Paired question-level bootstrap.",
            }
    report_comparisons = {}
    if variants:
        for variant in variants[1:]:
            paired = {}
            for name in (variants[0], variant):
                groups = {}
                for row in rows:
                    if row["variant"] == name and row.get("judge_status") == "success":
                        groups.setdefault(row["question_id"], []).append(row["judge"]["average"] / 10)
                paired[name] = {k: sum(v) / len(v) for k, v in groups.items()}
            common = sorted(paired[variants[0]].keys() & paired[variant].keys())
            if common:
                stats = bootstrap_ci_paired([paired[variant][k] - paired[variants[0]][k] for k in common])
                stats["bootstrap_nonpositive_fraction"] = stats.pop("p_value")
                stats["ci_positive"] = stats.pop("significant")
                report_comparisons[f"{variant}_minus_{variants[0]}"] = {**stats, "num_paired_questions": len(common),
                    "metric": "judge_quality_0_to_1", "exploratory": len(common) < 10,
                    "scope": "Only available Judge scores; completion and acceptance rates reported separately"}
    return {"variants": output, "comparisons": comparisons, "report_comparisons": report_comparisons}


async def run_experiment(config: dict, dataset: dict, output_dir: str | Path,
                         variants: dict[str, dict] | None = None, repetitions: int = 1,
                         judge=None, upload: bool = False, resume_from: str | Path | None = None) -> dict:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    concurrency = config.get("evaluation", {}).get("max_sample_concurrency", 1)
    if type(concurrency) is not int or not 1 <= concurrency <= 8:
        raise ValueError("evaluation.max_sample_concurrency must be an integer in 1..8")
    validate_dataset(dataset)
    dataset = copy.deepcopy(dataset)
    for question in dataset["examples"]:
        question.setdefault("evidence", dataset.get("documents", []))
    dataset["content_hash"] = hashlib.sha256(json.dumps(
        {k: v for k, v in dataset.items() if k != "content_hash"},
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if upload:
        from src.utils.env_config import get_env
        if not get_env("LANGSMITH_API_KEY"):
            raise ValueError("--langsmith requires LANGSMITH_API_KEY configured locally")
    variants = variants or {"full": {}, "no_compressor": {"compressor": {"enable_multilevel": False}}}
    previous = []
    if resume_from is not None:
        if upload:
            raise ValueError("Local checkpoint reuse cannot be combined with cloud upload")
        resume_from = Path(resume_from)
        prior_manifest = json.loads((resume_from / "manifest.json").read_text())
        if prior_manifest["status"] == "running" or prior_manifest["dataset_hash"] != dataset["content_hash"]:
            raise ValueError("Resume requires a stopped experiment with an identical dataset snapshot")
        previous = [json.loads(line) for line in (resume_from / "results.jsonl").read_text().splitlines()]
        for row in previous:
            if row["status"] == "success" and row["variant"] in variants:
                expected = config_fingerprint(merge_config(config, variants[row["variant"]]))
                if row.get("config_hash") != expected:
                    raise ValueError("Resume configuration differs; run a separate comparison instead")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    experiment_id = str(uuid.uuid4())
    revision = source_revision()
    manifest = {"experiment_id": experiment_id, "created_at": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset.get("name", "custom"), "dataset_version": dataset.get("version", "unversioned"),
                "dataset_hash": dataset.get("content_hash") or config_fingerprint(dataset),
                "scorer_version": SCORER_VERSION, "repetitions": repetitions, **revision,
                "report_evaluator_version": REPORT_EVALUATOR_VERSION if judge is not None and hasattr(judge, "score_report") else "not_requested_or_legacy",
                "config": safe_data(config, limit=1_000_000), "variants": variants,
                "langsmith": "requested" if upload else "disabled", "status": "running"}
    if resume_from is not None:
        manifest["resume_from"] = str(resume_from.resolve())
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    final_status = "completed"
    try:
        for variant, overrides in variants.items():
            cfg = merge_config(config, overrides)
            metadata = {k: manifest[k] for k in ("experiment_id", "dataset_name", "dataset_version", "dataset_hash", "git_commit", "source_hash", "scorer_version")}
            metadata.update({"variant": variant, "config_hash": config_fingerprint(cfg)})
            if upload:
                await _run_langsmith(dataset, cfg, output_dir, metadata, repetitions, judge, rows)
            else:
                semaphore = asyncio.Semaphore(concurrency)
                stopped = asyncio.Event()
                reused = set()
                for old in previous:
                    key = (old["question_id"], old.get("repetition", 0))
                    if old["variant"] != variant or old["status"] != "success" or key in reused:
                        continue
                    relative = Path(old["report_path"]).parent
                    shutil.copytree(resume_from / relative, output_dir / relative)
                    saved = {**old, "reused_from": str(resume_from.resolve()), "origin_sample_id": old["sample_id"]}
                    (output_dir / relative / "result.json").write_text(json.dumps(saved, ensure_ascii=False, indent=2))
                    rows.append(saved); _append_result(output_dir, saved); reused.add(key)
                async def execute(question, repetition):
                    async with semaphore:
                        if stopped.is_set():
                            return
                        local_judge = judge
                        if concurrency > 1 and hasattr(judge, "_sampling_kwargs"):
                            local_judge = copy.copy(judge)
                            local_judge._policy = None
                        row = await run_sample(question, cfg, output_dir, {**metadata, "repetition": repetition, "example_id": question["id"]}, local_judge)
                        rows.append(row)
                        _append_result(output_dir, row)
                        if row.get("infrastructure_error"):
                            stopped.set()
                            raise EvaluationInfrastructureError(row["infrastructure_error"])
                tasks = [asyncio.create_task(execute(question, repetition)) for repetition in range(repetitions)
                         for question in dataset["examples"] if (question["id"], repetition) not in reused]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
    except BaseException as exc:
        final_status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
        raise
    finally:
        # Recover a sample persisted by its cancellation finally block, so that
        # an interrupted experiment still counts that attempted sample.
        seen = {r["sample_id"] for r in rows}
        for result_path in sorted((output_dir / "samples").glob("*/result.json")):
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            if saved["sample_id"] not in seen:
                rows.append(saved)
                _append_result(output_dir, saved)
        summary = summarize(rows)
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["status"] = final_status
        (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        flush_traces(timeout=2.0)
    return {"manifest": manifest, "summary": summary, "details": rows}


def _append_result(output_dir: Path, row: dict):
    with (output_dir / "results.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


async def _run_langsmith(dataset, config, output_dir, metadata, repetitions, judge, rows):
    """SDK drives the actual target: model calls are never rerun just to upload."""
    from src.utils.env_config import get_env
    if not get_env("LANGSMITH_API_KEY"):
        raise ValueError("--langsmith requires LANGSMITH_API_KEY configured locally")
    from langsmith import Client, tracing_context
    from langsmith.utils import LangSmithNotFoundError
    client = Client(timeout_ms=10000)
    name = f"{metadata['dataset_name']}-{metadata['dataset_hash'][:12]}"
    try:
        remote = await asyncio.to_thread(client.read_dataset, dataset_name=name)
        existing = await asyncio.to_thread(lambda: list(client.list_examples(dataset_id=remote.id)))
        expected = {q["id"]: {"inputs": {"question_id": q["id"], "query": q["query"]},
                             "outputs": {"ground_truth": q.get("ground_truth", {}), "expected_topics": q.get("expected_topics", [])}}
                    for q in dataset["examples"]}
        actual = {e.inputs.get("question_id"): {"inputs": e.inputs, "outputs": e.outputs} for e in existing}
        if len(existing) != len(expected) or actual != expected:
            raise ValueError("Remote content-hash dataset differs from local snapshot; refusing to reuse it")
    except LangSmithNotFoundError:
        remote = await asyncio.to_thread(client.create_dataset, name, description="Immutable content-hash version")
        await asyncio.to_thread(client.create_examples, dataset_id=remote.id,
            inputs=[{"question_id": q["id"], "query": q["query"]} for q in dataset["examples"]],
            outputs=[{"ground_truth": q.get("ground_truth", {}), "expected_topics": q.get("expected_topics", [])} for q in dataset["examples"]])
    questions = {q["id"]: q for q in dataset["examples"]}
    counts = Counter()
    async def target(inputs: dict) -> dict:
        q = questions[inputs["question_id"]]
        repetition = counts[q["id"]]
        counts[q["id"]] += 1
        row = await run_sample(q, config, output_dir, {**metadata, "example_id": q["id"], "repetition": repetition}, judge)
        rows.append(row)
        _append_result(output_dir, row)
        return row
    def evaluator(outputs: dict) -> dict:
        scores = [{"key": "rule_composite", "score": outputs["composite_score"]},
                  {"key": "completed", "score": float(outputs["status"] == "success")}]
        if outputs.get("judge_status") == "success":
            scores.append({"key": "judge_quality", "score": outputs["judge"]["average"] / 10})
        if "constraint_pass" in outputs:
            scores.append({"key": "report_contract", "score": float(outputs["constraint_pass"])})
        if "report_pass" in outputs:
            scores.append({"key": "report_accepted", "score": float(outputs["report_pass"])})
        return {"results": scores}
    with tracing_context(enabled=True, client=client):
        result = await client.aevaluate(target, data=remote.name, evaluators=[evaluator],
            max_concurrency=1, num_repetitions=repetitions, metadata=metadata,
            experiment_prefix=f"deepr-{metadata['variant']}", blocking=True)
    await asyncio.to_thread(client.flush, timeout=2.0)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("langsmith_experiments", []).append(result.experiment_name)
    manifest["langsmith"] = "submitted"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def export_review_queue(experiment_dir: str | Path, threshold: float = 0.6) -> Path:
    """Export low-score/failure cases for human labeling; does not invent ground truth."""
    directory = Path(experiment_dir)
    cases = [json.loads(line) for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    dataset = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in dataset["examples"]}
    path = directory / "review_queue.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for row in cases:
            if (row["status"] != "success" or row["composite_score"] < threshold or row.get("judge_status") == "failed"
                    or not row.get("constraint_pass", True) or row.get("report_pass") is False
                    or (row.get("judge_status") == "success" and row["judge"]["average"] / 10 < threshold)):
                stream.write(json.dumps({**row, "question": questions[row["question_id"]],
                                         "human_label": None, "review_notes": "", "approved_for_regression": False}, ensure_ascii=False) + "\n")
    return path
