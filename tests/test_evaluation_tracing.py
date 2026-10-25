"""Offline regression for telemetry/evaluation. No real keys, models or production DB."""
from __future__ import annotations
import asyncio
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from src.utils.tracing import research_trace, trace_chain, trace_block, record_event, config_fingerprint, safe_data
from src.core.judge import LLMJudge
from src.orchestrator.schemas import ResearchReport
from src.models.vllm_policy import VLLMPolicy
from evaluation.experiments import run_experiment, load_dataset, score_report, summarize, export_review_queue

ROOT = Path(__file__).resolve().parents[1]


class TraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_children_and_thread_parentage(self):
        @trace_chain(name="leaf")
        def leaf(value):
            record_event("value", {"value": value})
            return value
        @trace_chain(name="worker")
        async def worker(value):
            await asyncio.sleep(0)
            return await asyncio.to_thread(leaf, value)
        with research_trace({"example_id": "q"}) as recorder:
            await asyncio.gather(worker(1), worker(2))
        spans = recorder.snapshot()["spans"]
        root = next(s for s in spans if s["name"] == "research.run")
        workers = [s for s in spans if s["name"] == "worker"]
        self.assertEqual(len(workers), 2)
        for worker_span in workers:
            self.assertEqual(worker_span["parent_span_id"], root["span_id"])
            child = next(s for s in spans if s["name"] == "leaf" and s["parent_span_id"] == worker_span["span_id"])
            self.assertEqual(child["outputs"], worker_span["outputs"])

    async def test_concurrent_roots_do_not_mix(self):
        async def run(value):
            with research_trace({"example_id": value}) as recorder:
                await asyncio.sleep(0)
                record_event("value", {"value": value})
            return recorder.snapshot()
        a, b = await asyncio.gather(run("a"), run("b"))
        self.assertNotEqual(a["research_run_id"], b["research_run_id"])
        self.assertEqual(a["events"][0]["data"]["value"], "a")
        self.assertEqual(b["events"][0]["data"]["value"], "b")

    async def test_cancellation_propagates_and_is_recorded(self):
        recorder = None
        async def wait():
            nonlocal recorder
            with research_trace() as recorder:
                with trace_block("waiting"):
                    await asyncio.sleep(100)
        task = asyncio.create_task(wait())
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(recorder.status, "cancelled")
        self.assertTrue(all(s["status"] == "cancelled" for s in recorder.snapshot()["spans"]))

    async def test_exporter_failure_does_not_repeat_business(self):
        import langsmith
        body = Mock(return_value=7)
        @trace_chain(name="business")
        def work():
            return body()
        with patch("src.utils.tracing.is_tracing_enabled", return_value=True), patch.object(langsmith, "trace", side_effect=RuntimeError("exporter down")):
            with research_trace() as recorder:
                self.assertEqual(work(), 7)
        body.assert_called_once()
        self.assertEqual(recorder.status, "success")
        self.assertTrue(recorder.export_errors)

    async def test_real_sdk_local_context_has_matching_trace_id(self):
        # Actual SDK context propagation, no network or cloud credentials.
        from langsmith import Client, tracing_context
        client = Client(api_key="unit-test", auto_batch_tracing=False)
        with patch.object(Client, "create_run") as create, patch.object(Client, "update_run") as update:
            with tracing_context(enabled="local", client=client):
                with research_trace() as recorder:
                    with trace_block("child"):
                        from langsmith.run_helpers import get_current_run_tree
                        child = get_current_run_tree()
                        self.assertEqual(str(child.trace_id), recorder.trace_id)
                        self.assertEqual(str(child.parent_run_id), recorder.run_id)
            create.assert_not_called()
            update.assert_not_called()
        self.assertEqual(recorder.trace_id, recorder.run_id)

    async def test_real_sdk_experiment_target_and_evaluator_contract(self):
        from datetime import datetime, timezone
        import uuid
        from langsmith import Client
        from langsmith.schemas import Example
        client = Client(api_key="unit-test", auto_batch_tracing=False)
        called = []
        async def target(inputs):
            called.append(inputs["x"])
            return {"score": inputs["x"]}
        def evaluator(outputs):
            return {"results": [{"key": "rule_composite", "score": outputs["score"]}]}
        example = Example(id=uuid.uuid4(), dataset_id=uuid.uuid4(),
                          created_at=datetime.now(timezone.utc), inputs={"x": .8})
        # upload_results=False currently still traces the prediction in SDK
        # 0.14.11, so explicitly stub transport methods as well.
        with patch.object(Client, "create_run"), patch.object(Client, "update_run"):
            result = await client.aevaluate(target, data=[example], evaluators=[evaluator],
                                           upload_results=False, max_concurrency=1)
            await result.wait()
            rows = [row async for row in result]
        self.assertEqual(called, [.8])
        feedback = rows[0]["evaluation_results"]["results"]
        self.assertEqual(feedback[0].key, "rule_composite")
        self.assertEqual(feedback[0].score, .8)

    async def test_provider_usage_is_not_character_estimate(self):
        policy = VLLMPolicy(base_url="http://unit.test/v1", api_key="unit")
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="a", tool_calls=[], reasoning_content=None), finish_reason="stop")],
                                   usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}))
        with research_trace() as recorder:
            parsed = policy._parse_response(response)
            response.usage = None
            missing = policy._parse_response(response)
        self.assertEqual(parsed["usage"]["total_tokens"], 13)
        self.assertEqual(missing["usage_source"], "unavailable")
        usage = recorder.snapshot()["usage"]
        self.assertEqual((usage["llm_calls"], usage["provider_usage_calls"], usage["total_tokens"]), (2, 1, 13))
        policy.client.close()

    def test_secret_redaction_and_hash(self):
        data = {"model": {"api_key": "do-not-export", "max_tokens": 300}, "policy": object(), "reasoning_content": "private"}
        safe = safe_data(data)
        self.assertNotIn("do-not-export", json.dumps(safe))
        self.assertEqual(safe["model"]["max_tokens"], 300)
        changed_key = copy.deepcopy(data); changed_key["model"]["api_key"] = "another"
        self.assertEqual(config_fingerprint(data), config_fingerprint(changed_key))
        changed_key["model"]["max_tokens"] = 400
        self.assertNotEqual(config_fingerprint(data), config_fingerprint(changed_key))
        self.assertNotEqual(config_fingerprint({"items": list(range(101))}),
                            config_fingerprint({"items": list(range(100)) + [999]}))


class JudgeTests(unittest.TestCase):
    def output(self, score=8):
        return {key: {"score": score, "reason": "ok"} for key in
                ("factual_accuracy", "logical_consistency", "citation_quality", "comprehensiveness", "overall")}

    def test_long_report_tail_is_reviewed_and_weighted(self):
        prompts = []
        def policy(messages):
            prompts.append(messages[-1]["content"])
            return {"content": json.dumps(self.output(8 if len(prompts) == 1 else 4))}
        judge = LLMJudge(chunk_chars=4000)
        judge._policy = policy
        result = judge.score_single("A" * 4000 + "TAIL_MARKER" * 100, "q")
        self.assertEqual(result["reviewed_chars"], 5100)
        self.assertEqual(result["num_chunks"], 2)
        self.assertIn("TAIL_MARKER", prompts[1])
        self.assertAlmostEqual(result["average"], (4000 * 8 + 1100 * 4) / 5100, places=3)

    def test_bad_judge_output_is_failure_not_zero_quality(self):
        judge = LLMJudge()
        judge._policy = lambda messages: {"content": json.dumps(self.output(99))}
        self.assertIn("error", judge.score_single("report", "q"))
        judge._policy = lambda messages: {"content": '{"overall":{"score":8}}'}
        self.assertIn("error", judge.score_single("report", "q"))

    def test_review_budget_is_explicit(self):
        judge = LLMJudge(chunk_chars=10, max_report_chars=20)
        judge._policy = Mock()
        self.assertIn("error", judge.score_single("x" * 21, "q"))
        judge._policy.assert_not_called()


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_synthesis_fallback_is_partial_not_full_success(self):
        from src.agents.summarizer import SummarizerAgent
        from src.orchestrator.schemas import AgentResult, AgentStatus, SubTask, TaskType
        class Policy:
            tools = None
            def __call__(self, messages):
                raise RuntimeError("model unavailable")
        agent = SummarizerAgent("test", Policy())
        task = SubTask("summary", TaskType.ANALYZE, "synthesize")
        result = await agent.run(task, {"query": "q", "results": [
            AgentResult("evidence", AgentStatus.SUCCESS, output="已经获得的有效材料与结论。" * 30)]})
        self.assertEqual(result.output.status, "partial")

    async def test_cancelled_sample_is_in_interrupted_experiment_denominator(self):
        entered = asyncio.Event()
        def initialize(config, session_id):
            class Orchestrator:
                async def run(self, query, config):
                    entered.set()
                    await asyncio.sleep(100)
            return {"orchestrator": Orchestrator(), "tools": []}
        dataset = {"examples": [{"id": "q", "query": "q"}]}
        with tempfile.TemporaryDirectory() as tmp, patch("evaluation.experiments.initialize_modules", side_effect=initialize):
            output = Path(tmp) / "interrupted"
            task = asyncio.create_task(run_experiment({"adversarial": {"enabled": False}}, dataset, output, {"full": {}}))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            manifest = json.loads((output / "manifest.json").read_text())
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(manifest["status"], "cancelled")
            self.assertEqual(summary["variants"]["full"]["status_counts"], {"cancelled": 1})
            self.assertEqual(summary["variants"]["full"]["average_composite"], 0)

    async def test_judge_failure_is_separate_from_research_quality(self):
        def initialize(config, session_id):
            class Orchestrator:
                async def run(self, query, config):
                    return ResearchReport(query=query, content="## 结论\nLRU 用于容量淘汰。")
            return {"orchestrator": Orchestrator(), "tools": []}
        judge = SimpleNamespace(score_single=lambda *args: {"error": "malformed judge response"})
        with tempfile.TemporaryDirectory() as tmp, patch("evaluation.experiments.initialize_modules", side_effect=initialize):
            result = await run_experiment({"adversarial": {"enabled": False}},
                {"examples": [{"id": "q", "query": "q", "ground_truth": {"LRU": "capacity"}}]},
                Path(tmp) / "run", {"full": {}}, judge=judge)
        row = result["details"][0]
        self.assertEqual(row["status"], "success")
        self.assertEqual(row["judge_status"], "failed")
        self.assertGreater(row["composite_score"], 0)
        self.assertIsNone(result["summary"]["variants"]["full"]["average_judge_score"])

    async def test_cloud_adapter_calls_target_once_and_persists_feedback_contract(self):
        from unittest.mock import AsyncMock
        from evaluation.experiments import _run_langsmith
        fake = Mock()
        fake.read_dataset.return_value = SimpleNamespace(name="dataset", id="dataset-id")
        fake.list_examples.return_value = [SimpleNamespace(inputs={"question_id": "q", "query": "q"},
            outputs={"ground_truth": {}, "expected_topics": []})]
        async def aevaluate(target, **kwargs):
            output = await target({"question_id": "q", "query": "q"})
            scores = kwargs["evaluators"][0](outputs=output)["results"]
            self.assertEqual({s["key"] for s in scores}, {"rule_composite", "completed"})
            self.assertEqual(kwargs["max_concurrency"], 1)
            return SimpleNamespace(experiment_name="unit-experiment")
        fake.aevaluate = aevaluate
        row = {"sample_id": "sample", "status": "success", "composite_score": .8,
               "elapsed_seconds": 1, "variant": "full", "question_id": "q"}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "manifest.json").write_text('{}')
            rows = []
            with patch("src.utils.env_config.get_env", return_value="unit-test"), \
                 patch("langsmith.Client", return_value=fake), \
                 patch("evaluation.experiments.run_sample", new=AsyncMock(return_value=row)) as target:
                await _run_langsmith({"examples": [{"id": "q", "query": "q"}]}, {}, directory,
                    {"dataset_name": "unit", "dataset_hash": "hash", "variant": "full"}, 1, None, rows)
            target.assert_awaited_once()
            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads((directory / "manifest.json").read_text())["langsmith"], "submitted")

    async def test_isolation_failure_denominator_artifacts_and_review(self):
        seen = []
        def initialize(config, session_id):
            seen.append((config["memory"]["db_path"], session_id))
            class Orchestrator:
                async def run(self, query, config):
                    if query == "bad":
                        raise RuntimeError("intentional model failure")
                    return ResearchReport(query=query, content="## 结论\nLRU 用于容量淘汰。", confidence=.8)
            return {"orchestrator": Orchestrator(), "tools": []}
        dataset = {"name": "unit", "version": "1", "examples": [
            {"id": "good", "query": "good", "ground_truth": {"LRU": "eviction"}, "expected_topics": ["LRU"]},
            {"id": "bad", "query": "bad", "ground_truth": {"LRU": "eviction"}}]}
        config = {"memory": {"db_path": "production-must-not-open.db"}, "adversarial": {"enabled": False}}
        with tempfile.TemporaryDirectory() as tmp, patch("evaluation.experiments.initialize_modules", side_effect=initialize):
            output = Path(tmp) / "experiment"
            result = await run_experiment(config, dataset, output)
            self.assertEqual(len(seen), 4)
            self.assertEqual(len(set(p for p, _ in seen)), 4)
            self.assertEqual(len(set(i for _, i in seen)), 4)
            self.assertTrue(all(p != config["memory"]["db_path"] for p, _ in seen))
            for variant in result["summary"]["variants"].values():
                self.assertEqual(variant["completion_rate"], .5)
            for row in result["details"]:
                self.assertTrue((output / row["trace_path"]).is_file())
                self.assertTrue(row["research_run_id"])
                if row["status"] == "failed":
                    self.assertEqual(row["composite_score"], 0)
            queue = export_review_queue(output)
            self.assertGreaterEqual(len(queue.read_text().splitlines()), 2)
        self.assertEqual(config["memory"]["db_path"], "production-must-not-open.db")

    async def test_scoring_uses_body_without_question_leakage(self):
        def initialize(config, session_id):
            class Orchestrator:
                async def run(self, query, config):
                    return ResearchReport(query=query, content="## 报告\n这里没有标准答案。")
            return {"orchestrator": Orchestrator(), "tools": []}
        dataset = {"examples": [{"id": "q", "query": "LRU", "ground_truth": {"LRU": "eviction"}}]}
        with tempfile.TemporaryDirectory() as tmp, patch("evaluation.experiments.initialize_modules", side_effect=initialize):
            result = await run_experiment({"adversarial": {"enabled": False}}, dataset, Path(tmp) / "run", {"full": {}})
        self.assertEqual(result["details"][0]["metrics"]["factual_accuracy"], 0)

    async def test_ablation_switches_reach_runtime_and_no_memory_opens_no_db(self):
        from src.core.runner import initialize_modules
        from src.models.model_router import ModelRouter
        policy = VLLMPolicy(base_url="http://unit.test/v1", api_key="unit")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "should-not-exist.db"
            cfg = {"model": {"backend": "deepseek"}, "memory": {"enabled": False, "db_path": str(path)},
                   "compressor": {"enable_multilevel": False}, "tools": {"web_search": {"mock_mode": True}}}
            with patch.object(ModelRouter, "create_backend", return_value=policy):
                modules = initialize_modules(cfg)
            self.assertIsNone(modules["orchestrator"].memory_store)
            self.assertIsNone(modules["orchestrator"].compressor)
            self.assertFalse(path.exists())
        policy.client.close()

    def test_canonical_metric_contributes_to_researchbench_composite(self):
        from evaluation.benchmarks.research_bench import ResearchBench
        bench = ResearchBench()
        q = bench.questions[0]
        text = " ".join(q["ground_truth"])
        with patch("evaluation.metrics.rule_based.RuleBasedMetrics.semantic_fact_accuracy", return_value=0):
            result = bench.evaluate_report(text, q["id"])
        self.assertEqual(result["metrics"]["factual_accuracy"], 1)
        self.assertAlmostEqual(result["composite_score"],
                               sum(result["metrics"][k] * w for k, w in {"factual_accuracy": .25, "logical_consistency": .2, "citation_coverage": .2, "bias": .2, "comprehensiveness": .15}.items()))

    def test_dataset_hash_changes_and_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            data = {"examples": [{"id": "q", "query": "original"}]}
            path.write_text(json.dumps(data))
            first = load_dataset(path)["content_hash"]
            data["examples"][0]["query"] = "changed"
            path.write_text(json.dumps(data))
            self.assertNotEqual(first, load_dataset(path)["content_hash"])
            data["examples"] *= 2
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_dataset(path)

    def test_paired_bootstrap_is_reproducible_and_small_sample_marked(self):
        rows = [{"variant": v, "question_id": q, "status": "success", "composite_score": score,
                 "elapsed_seconds": 1, "judge_status": "not_requested"}
                for v, q, score in (("full", "a", .5), ("full", "b", .6), ("off", "a", .7), ("off", "b", .7))]
        a = summarize(rows)
        self.assertEqual(a, summarize(rows))
        self.assertTrue(a["comparisons"]["off_minus_full"]["exploratory"])

    async def test_snapshot_tools_never_fetch_external_urls(self):
        from src.tools.snapshot import SnapshotBrowserTool, SnapshotSearchTool
        path = str(ROOT / "evaluation/data/research_regression_v1.json")
        search, browser = SnapshotSearchTool(path), SnapshotBrowserTool(path)
        first = await search.execute("anything")
        second = await search.execute("different")
        self.assertEqual(first["results"], second["results"])
        self.assertIn("Error:", await browser.execute("https://not-in-corpus.example"))


if __name__ == "__main__":
    unittest.main()
