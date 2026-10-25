"""Offline five-level recovery contracts: scripted policies, local HTTP, no paid calls."""
import asyncio
import copy
from dataclasses import asdict, replace
import json
import unittest
from unittest.mock import AsyncMock, patch

from aiohttp import web

from src.agents.researcher import ResearcherAgent
from src.agents.summarizer import SummarizerAgent
from src.orchestrator.agent_pool import AgentPool
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.schemas import (AgentResult, AgentStatus, OrchestratorState,
                                     ResearchReport, RunConfig, SubTask, TaskType)
from src.planner.dag import DAG
from src.planner.planner import Planner, PlanParseError
from src.utils.execution import call_policy, deadline_scope
from src.utils.recovery import (Failure, RecoveryConfig, RecoveryError, ResearchAbort,
                                ToolExecutor, classify_error, current_recovery,
                                normalize_result, public_message, recovery_scope,
                                retry_after_seconds, wait_retry)
from src.utils.schema_validator import validate_planner_output


FAST = RecoveryConfig(base_delay=0, max_delay=0, jitter=0)
TASK = SubTask("a", TaskType.SEARCH, "检索并归纳研究证据")
EVIDENCE = {"results": [{"title": "证据", "url": "https://evidence.test/a", "snippet": "有来源的研究发现"}]}


class Tool:
    name = "web_search"
    backend = "primary"

    def __init__(self, outputs=None, name="web_search", backend="primary"):
        self.outputs = list(outputs or [EVIDENCE])
        self.name, self.backend, self.calls = name, backend, 0

    def get_openai_tool_schema(self):
        return {"type": "function", "function": {"name": self.name, "parameters": {
            "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}

    async def execute(self, query):
        self.calls += 1
        value = self.outputs[min(self.calls - 1, len(self.outputs) - 1)]
        if isinstance(value, BaseException):
            raise value
        return copy.deepcopy(value)


class Policy:
    tools = None
    max_tokens = 128

    def __init__(self, outputs):
        self.outputs, self.calls, self.messages = list(outputs), 0, []

    def set_tools(self, tools):
        self.tools = tools

    async def acall(self, messages):
        self.messages.append(copy.deepcopy(messages))
        value = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        if isinstance(value, BaseException):
            raise value
        return copy.deepcopy(value)


def call(name="web_search", args=None, identifier="c1"):
    return {"id": identifier, "type": "function", "function": {
        "name": name, "arguments": json.dumps(args or {"query": "问题"}, ensure_ascii=False)}}


def message(*calls, content=""):
    return {"content": content, "tool_calls": list(calls)}


def task_data(task):
    return {**asdict(task), "task_type": task.task_type.value}


class FixedPlanner:
    _last_raw_json = "{}"

    def __init__(self, tasks):
        self.tasks, self.replans = tasks, 0

    def dag(self):
        dag = DAG()
        for task in self.tasks:
            dag.add_node(task.task_id)
            for dep in task.dependencies:
                dag.add_edge(dep, task.task_id)
        return dag

    async def generate_plan(self, *args):
        return self.dag()

    async def areplan(self, **kwargs):
        self.replans += 1
        return self.dag()

    def get_task_map_from_dag(self, *args):
        return {task.task_id: task for task in self.tasks}


class ClassificationTests(unittest.TestCase):
    def test_severity_depends_on_failure_scope(self):
        for status in (401, 402, 403):
            self.assertEqual(classify_error({"status_code": status, "error": "denied"}, model=True).level, 5)
            self.assertEqual(classify_error({"status_code": status, "error": "denied"}).level, 3)
        for status in (408, 425, 429, 500, 502, 503, 599):
            failure = classify_error({"status_code": status, "error": "unavailable"})
            self.assertEqual((failure.level, failure.retryable), (2, True))
        self.assertEqual(classify_error({"status_code": 404, "error": "missing"}).scope, "resource")

    def test_legacy_contracts_and_empty_are_distinct(self):
        for name, output in (("browser", "[Browser Error] Network error: timeout"),
                             ("calculator", "Error: division by zero"),
                             ("code_sandbox", {"success": False, "stderr": "SyntaxError: invalid syntax"})):
            self.assertEqual(normalize_result(name, output).status, "error")
        self.assertEqual(normalize_result("browser", "[Browser Warning] No content").status, "empty")
        self.assertEqual(normalize_result("web_search", {"results": []}).status, "empty")
        self.assertEqual(normalize_result("browser", "Article discusses Error: messages in applications").status, "success")
        self.assertEqual(normalize_result("web_search", {"results": [{"url": "x"}]}).failure.level, 2)
        self.assertEqual(normalize_result("browser", "<html>login</html>").failure.level, 2)
        self.assertEqual(normalize_result("browser", {"status_code": 403, "error": "denied"}).failure.scope, "resource")

    def test_bad_config_and_credential_redaction(self):
        for config in ({"max_attempts": 0}, {"max_attempts": True}, {"jitter": 2},
                       {"base_delay": float("nan")}, {"max_fallbacks": -1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                RecoveryConfig(**config)
        value = public_message('api_key=private1&token=private2 {"secret": "private3"} Bearer private4 sk-testsecret')
        for secret in ("private1", "private2", "private3", "private4", "sk-testsecret"):
            self.assertNotIn(secret, value)
        self.assertEqual(retry_after_seconds("3"), 3)
        self.assertIsNone(retry_after_seconds("nan"))
        self.assertIsNone(retry_after_seconds("bad date"))

    def test_plan_boundary_validation(self):
        a, b = task_data(TASK), task_data(replace(TASK, task_id="b"))
        for changes in ({"dependencies": None}, {"dependencies": [{}]},
                        {"dependencies": ["a"]}, {"timeout_seconds": 0},
                        {"timeout_seconds": float("inf")}, {"timeout_seconds": True}):
            with self.subTest(changes=changes):
                self.assertFalse(validate_planner_output({"sub_tasks": [{**a, **changes}, b]}).valid)
        self.assertFalse(validate_planner_output({"sub_tasks": [a, a]}).valid)
        self.assertFalse(validate_planner_output({"sub_tasks": [{**a, "dependencies": ["b"]}, {**b, "dependencies": ["a"]}]}).valid)
        for changes in ({"priority": "bad"}, {"context_keys": None}, {"search_hints": [1]}, {"task_type": {}}):
            self.assertFalse(validate_planner_output({"sub_tasks": [{**a, **changes}, b]}).valid)


class ToolRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_l1_rejects_before_io_and_has_one_repair(self):
        tool = Tool(); executor = ToolExecutor([tool], FAST)
        for args in ("not json", "[]", {}, {"query": 2}, {"query": " "}, {"query": "x", "extra": 3}):
            self.assertEqual((await executor.run(tool.name, args)).failure.level, 1)
        self.assertEqual((await executor.run("unknown", {})).failure.level, 1)
        self.assertEqual(tool.calls, 0)
        self.assertTrue(executor.repair_feedback())
        self.assertFalse(executor.repair_feedback())

    async def test_l2_transport_and_malformed_output_retry_then_succeed(self):
        for first in (ConnectionError("network unavailable"), TimeoutError("transport timeout"), {"results": "bad format"}):
            with self.subTest(first=first):
                tool = Tool([first, EVIDENCE]); executor = ToolExecutor([tool], FAST)
                result = await executor.run(tool.name, {"query": "x"})
                self.assertEqual((result.status, tool.calls), ("success", 2))
                self.assertEqual(executor.history[0]["error"]["level"], 2)

    async def test_l2_exhaustion_escalates_and_disables_tool(self):
        tool = Tool([ConnectionError("offline")]); executor = ToolExecutor([tool], FAST)
        result = await executor.run(tool.name, {"query": "x"})
        self.assertEqual((result.failure.level, result.failure.code, tool.calls), (3, "RETRIES_EXHAUSTED", 3))
        await executor.run(tool.name, {"query": "changed"})
        self.assertEqual(tool.calls, 3)
        self.assertNotIn(tool.name, executor.available())
        self.assertEqual(executor.history[-2]["error"]["level"], 3)

    async def test_empty_result_does_not_retry(self):
        tool = Tool([{"results": []}]); result = await ToolExecutor([tool], FAST).run(tool.name, {"query": "x"})
        self.assertEqual((result.status, tool.calls), ("empty", 1))

    async def test_retry_after_is_honored_and_deadline_skips_wait(self):
        failure = Failure(2, "RATE_LIMIT", "limited", retryable=True, retry_after=10)
        with patch("src.utils.recovery.asyncio.sleep", new_callable=AsyncMock) as sleep:
            self.assertTrue(await wait_retry(failure, 0, FAST))
            sleep.assert_awaited_once_with(10)
        with deadline_scope(seconds=.1), patch("src.utils.recovery.asyncio.sleep", new_callable=AsyncMock) as sleep:
            self.assertFalse(await wait_retry(failure, 0, FAST))
            sleep.assert_not_awaited()

    async def test_expired_deadline_and_cancellation_do_not_retry(self):
        tool = Tool()
        with deadline_scope(seconds=0), self.assertRaises(TimeoutError):
            await ToolExecutor([tool], FAST).run(tool.name, {"query": "x"})
        self.assertEqual(tool.calls, 0)
        started, drained = asyncio.Event(), asyncio.Event()
        class Blocking(Tool):
            async def execute(self, query):
                self.calls += 1; started.set()
                try:
                    await asyncio.sleep(10)
                finally:
                    drained.set()
        tool = Blocking(); task = asyncio.create_task(ToolExecutor([tool], FAST).run(tool.name, {"query": "x"}))
        await started.wait(); task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(drained.is_set()); self.assertEqual(tool.calls, 1)

    async def test_l3_backend_fallback_and_shared_circuit(self):
        with recovery_scope(FAST) as state:
            primary = Tool([{"error": "unauthorized", "status_code": 401}])
            fallback = Tool(backend="alternate"); primary.recovery_fallbacks = [fallback]
            result = await ToolExecutor([primary]).run(primary.name, {"query": "x"})
            self.assertEqual((result.status, result.backend), ("success", "alternate"))
            self.assertIn((primary.name, "primary"), state.circuits)
            another = Tool(); another.recovery_fallbacks = [Tool(backend="alternate")]
            result = await ToolExecutor([another]).run(another.name, {"query": "x"})
            self.assertEqual(result.status, "success"); self.assertEqual(another.calls, 0)

    async def test_resource_error_does_not_disable_whole_tool(self):
        tool = Tool([{"error": "not found", "status_code": 404}, "页面正文"], name="browser")
        executor = ToolExecutor([tool], FAST)
        self.assertEqual((await executor.run("browser", {"query": "missing"})).failure.scope, "resource")
        self.assertEqual((await executor.run("browser", {"query": "another"})).status, "success")
        self.assertEqual(tool.calls, 2)

    async def test_fallback_and_task_and_global_budgets(self):
        primary = Tool([{"error": "unauthorized", "status_code": 401}])
        primary.recovery_fallbacks = [Tool(backend="alternate")]
        executor = ToolExecutor([primary], replace(FAST, max_fallbacks=0))
        self.assertEqual((await executor.run(primary.name, {"query": "x"})).failure.level, 3)
        self.assertEqual(primary.recovery_fallbacks[0].calls, 0)
        executor = ToolExecutor([Tool()], replace(FAST, max_task_tool_calls=1))
        await executor.run("web_search", {"query": "x"})
        self.assertEqual((await executor.run("web_search", {"query": "y"})).failure.level, 4)
        with recovery_scope(replace(FAST, max_run_tool_calls=1)):
            await ToolExecutor([Tool()]).run("web_search", {"query": "x"})
            with self.assertRaises(ResearchAbort) as raised:
                await ToolExecutor([Tool()]).run("web_search", {"query": "y"})
            self.assertEqual(raised.exception.failure.level, 5)

    async def test_side_effect_is_not_retried_or_replayed(self):
        tool = Tool([ConnectionError("network")], name="mutation")
        tool.recovery_fallbacks = [Tool(name="mutation", backend="alternate")]
        executor = ToolExecutor([tool], FAST)
        first = await executor.run(tool.name, {"query": "write"}, "op1")
        second = await executor.run(tool.name, {"query": "write"}, "op1")
        self.assertIs(first, second); self.assertEqual(tool.calls, 1)
        self.assertEqual(tool.recovery_fallbacks[0].calls, 0)
        result = await executor.run(tool.name, {"query": "other write"}, "op1")
        self.assertEqual(result.failure.code, "CALL_ID_REUSED")

    async def test_cross_tool_fallbacks_are_bounded_but_other_capabilities_are_independent(self):
        failed = {"error": "not found", "status_code": 404}
        tools = [Tool([failed], name=name) for name in ("web_search", "arxiv_reader", "browser", "file_reader")]
        executor = ToolExecutor(tools, FAST)
        for tool in tools[:3]:
            self.assertEqual((await executor.run(tool.name, {"query": "x"})).failure.level, 3)
        result = await executor.run("file_reader", {"query": "x"})
        self.assertEqual(result.failure.level, 4); self.assertEqual(tools[-1].calls, 0)
        calculator = Tool(["2"], name="calculator")
        executor = ToolExecutor([Tool([failed]), calculator], replace(FAST, max_fallbacks=0))
        await executor.run("web_search", {"query": "x"})
        self.assertEqual((await executor.run("calculator", {"query": "1+1"})).status, "success")

    async def test_cancel_during_backoff_stops_further_attempts(self):
        tool = Tool([ConnectionError("offline")]); first = asyncio.Event()
        async def backoff(*args):
            first.set(); await asyncio.sleep(10)
        with patch("src.utils.recovery.wait_retry", side_effect=backoff):
            task = asyncio.create_task(ToolExecutor([tool], FAST).run(tool.name, {"query": "x"}))
            await first.wait(); task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(tool.calls, 1)

    async def test_real_tools_arguments_and_snapshot_never_escape_to_live(self):
        from src.core.runner import _create_tools_factory
        from src.tools.arxiv_reader import ArxivReaderTool
        from src.tools.notepad import NotepadTool
        from src.tools.calculator import CalculatorTool
        from pathlib import Path
        import tempfile
        arxiv = ArxivReaderTool(use_mock=True); notepad = NotepadTool()
        executor = ToolExecutor([arxiv, notepad, CalculatorTool()], FAST)
        for name, args in (("arxiv_reader", {}), ("arxiv_reader", {"query": ""}),
                           ("notepad", {"action": "write"}),
                           ("notepad", {"action": "clear", "content": "invalid"})):
            self.assertEqual((await executor.run(name, args)).failure.level, 1)
        self.assertEqual(executor.calls, 0)
        self.assertEqual((await executor.run("calculator", {"expression": "[x for x in [1, 2]]"})).failure.level, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshot.json"; path.write_text(json.dumps({"documents": []}))
            with patch("src.tools.web_search.WebSearchTool", side_effect=AssertionError("live fallback")):
                tools = _create_tools_factory({"tools": {"snapshot_path": str(path),
                    "fallback_backends": {"web_search": ["bing"]}}})
            self.assertTrue(all(not getattr(t, "recovery_fallbacks", []) for t in tools))
            executor = ToolExecutor(tools, FAST)
            self.assertEqual((await executor.run("web_search", {"query": "x", "top_n": 21})).failure.level, 1)
            self.assertEqual(executor.calls, 0)

    async def test_cleanup_failure_does_not_skip_other_transports(self):
        from src.utils.recovery import close_tools
        primary, fallback = Tool(), Tool(backend="alternate")
        primary.recovery_fallbacks = [fallback]
        primary.close_session = AsyncMock(side_effect=RuntimeError("close failed"))
        fallback.close_session = AsyncMock()
        self.assertFalse(await close_tools([primary]))
        fallback.close_session.assert_awaited_once()

    async def test_actual_http_status_and_non_json_retry_after(self):
        from src.tools.web_search import WebSearchTool
        from src.tools.browser import BrowserTool
        count = 0
        async def handler(request):
            nonlocal count
            count += 1
            if count == 1:
                return web.Response(status=429, text="busy", headers={"Retry-After": "0"})
            return web.json_response({"organic_results": [{"title": "t", "link": "https://evidence.test/a", "snippet": "real body"}]})
        async def missing(request):
            return web.Response(status=404, text="not available")
        app = web.Application(); app.router.add_get("/search", handler); app.router.add_get("/missing", missing)
        runner = web.AppRunner(app); await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0); await site.start()
        port = site._server.sockets[0].getsockname()[1]
        tool = WebSearchTool(backend="serpapi"); tool.serpapi_key = "local-test-key"
        tool.serpapi_endpoint = f"http://127.0.0.1:{port}/search"
        try:
            executor = ToolExecutor([tool], FAST)
            result = await executor.run("web_search", {"query": "x"})
            self.assertEqual((result.status, count), ("success", 2))
            self.assertEqual(executor.history[0]["error"]["retry_after"], 0)
            outcome = await ToolExecutor([BrowserTool()], FAST).run("browser", {"url": f"http://127.0.0.1:{port}/missing"})
            self.assertEqual((outcome.failure.level, outcome.failure.scope), (3, "resource"))
        finally:
            await tool.close_session(); await runner.cleanup()


class ModelAndWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_retries_once_owner_and_auth_aborts_immediately(self):
        policy = Policy([ConnectionError("offline"), message(content="answer")])
        with recovery_scope(FAST):
            self.assertEqual((await call_policy(policy, []))["content"], "answer")
        self.assertEqual(policy.calls, 2)
        policy = Policy([RecoveryError(Failure(5, "LLM_AUTH_OR_QUOTA_ERROR", "HTTP 402", "backend"))])
        with recovery_scope(FAST), self.assertRaises(ResearchAbort):
            await call_policy(policy, [])
        self.assertEqual(policy.calls, 1)

    async def test_invalid_model_envelope_is_retried(self):
        policy = Policy([{"tool_calls": "bad"}, message(content="answer")])
        with recovery_scope(FAST):
            await call_policy(policy, [])
        self.assertEqual(policy.calls, 2)

    async def test_model_shared_budget_and_scope_reset(self):
        policy = Policy([message(content="answer")])
        with recovery_scope(replace(FAST, max_run_model_calls=1)) as state:
            await call_policy(policy, [])
            with self.assertRaises(ResearchAbort):
                await call_policy(policy, [])
            self.assertEqual(state.model_calls, 1)
        self.assertIsNone(current_recovery())
        with recovery_scope(FAST) as other:
            self.assertEqual((other.model_calls, other.tool_calls, other.circuits), (0, 0, set()))

    async def test_estimated_token_budget_rejects_before_model_io(self):
        policy = Policy([message(content="answer")])
        with recovery_scope(replace(FAST, max_estimated_tokens=1)), self.assertRaises(ResearchAbort):
            await call_policy(policy, [{"role": "user", "content": "q"}])
        self.assertEqual(policy.calls, 0)

    async def test_native_model_timeout_retries_but_invalid_content_or_duplicate_ids_do_not_escape(self):
        for invalid in (TimeoutError("transport timeout"), {"content": []}, message(call(), call())):
            policy = Policy([invalid, message(content="answer")])
            with recovery_scope(FAST):
                self.assertEqual((await call_policy(policy, []))["content"], "answer")
            self.assertEqual(policy.calls, 2)

    async def test_missing_call_id_gets_consistent_protocol_identity(self):
        tc = call(); tc.pop("id")
        policy = Policy([message(tc), message(content="根据来源得到结论。")])
        result = await ResearcherAgent("w", policy, [Tool()], recovery_config=FAST).run(TASK, {})
        self.assertEqual(result.status, AgentStatus.SUCCESS)
        assistant, tool_message = policy.messages[-1][-2:]
        self.assertEqual(assistant["tool_calls"][0]["id"], tool_message["tool_call_id"])

    async def test_browser_success_is_evidence_and_not_false_empty(self):
        tool = Tool(["带有可用信息的网页正文"], name="browser")
        policy = Policy([message(call("browser")), message(content="根据来源，结论有证据支持。")])
        result = await ResearcherAgent("worker", policy, [tool], recovery_config=FAST).run(TASK, {})
        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertTrue(result.acceptance["passed"])

    async def test_fake_complete_output_without_evidence_fails_acceptance(self):
        policy = Policy([message(content="## 结论\n模型声称这是完整的结论。")])
        result = await ResearcherAgent("worker", policy, [Tool()], recovery_config=FAST).run(TASK, {})
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("no_usable_evidence", result.acceptance["gaps"])
        self.assertTrue(result.recovery["needs_replan"])

    async def test_argument_repair_and_batch_protocol_order(self):
        bad = call(args={"query": 3}); good = call(identifier="c2")
        policy = Policy([message(bad, good), message(content="根据来源，已获得证据。")])
        tool = Tool(); result = await ResearcherAgent("w", policy, [tool], recovery_config=FAST).run(TASK, {})
        self.assertEqual((result.status, result.recovery["repairs"], tool.calls), (AgentStatus.SUCCESS, 1, 1))
        roles = [m["role"] for m in policy.messages[-1]]
        self.assertEqual(roles[-4:], ["assistant", "tool", "tool", "user"])

    async def test_exhausted_argument_repair_and_partial_evidence(self):
        bad = call(args={"query": 3})
        policy = Policy([message(call()), message(bad), message(bad)])
        result = await ResearcherAgent("w", policy, [Tool()], recovery_config=FAST).run(TASK, {})
        self.assertEqual(result.status, AgentStatus.PARTIAL)
        self.assertIn("有来源的研究发现", result.output)
        self.assertIn("argument_repair_budget_exhausted", result.acceptance["gaps"])

    async def test_explicit_acceptance_failure_stays_partial(self):
        policy = Policy([message(call()), message(content="已有来源的初步发现。")])
        result = await ResearcherAgent("w", policy, [Tool()], recovery_config=FAST).run(TASK,
            {"task_acceptance": {"a": {"required_terms": ["关键条件"], "min_chars": 100}}})
        self.assertEqual(result.status, AgentStatus.PARTIAL)
        self.assertIn("missing_required_term:关键条件", result.acceptance["gaps"])

    async def test_turn_limit_preserves_evidence_as_partial(self):
        result = await ResearcherAgent("w", Policy([message(call())]), [Tool()], max_turns=1,
                                       recovery_config=FAST).run(TASK, {})
        self.assertEqual(result.status, AgentStatus.PARTIAL)
        self.assertIn("有来源的研究发现", result.output)
        self.assertIn("max_turns_exhausted", result.acceptance["gaps"])

    async def test_analyze_accepts_upstream_evidence_without_duplicate_search(self):
        task = replace(TASK, task_type=TaskType.ANALYZE, dependencies=["up"])
        policy = Policy([message(content="根据上游结果，比较两种研究方案。")]); tool = Tool()
        result = await ResearcherAgent("w", policy, [tool], recovery_config=FAST).run(task,
            {"dep:up": AgentResult("up", AgentStatus.SUCCESS, "上游证据")})
        self.assertEqual(result.status, AgentStatus.SUCCESS); self.assertEqual(tool.calls, 0)

    async def test_summarizer_retains_partial_and_lists_gaps(self):
        policy = Policy([message(content="## 关键发现\n依据现有结果整理。\n## 结论\n尚需补充验证。")])
        partial = AgentResult("a", AgentStatus.PARTIAL, "现有的部分证据", acceptance={"passed": False, "gaps": ["missing_condition"]})
        result = await SummarizerAgent("s", policy).run(TASK, {"query": "q", "results": [partial]})
        self.assertEqual(result.output.status, "partial")
        self.assertIn("现有的部分证据", str(policy.messages))
        self.assertIn("missing_condition", result.output.content)


class PlanningAndTerminationTests(unittest.IsolatedAsyncioTestCase):
    def orchestrator(self, tasks, factory=None, tools=None, config=FAST):
        pool = AgentPool(factory or (lambda: Policy([message(content="根据来源得出结论。")])) , tools or (lambda: []), recovery_config=config)
        return Orchestrator(FixedPlanner(tasks), pool), pool

    async def test_any_one_rejected_task_uses_existing_replan_and_respects_limit(self):
        o, _ = self.orchestrator([TASK]); o._config = RunConfig(max_replan_rounds=1)
        o._results = [AgentResult(str(i), AgentStatus.SUCCESS, "good") for i in range(5)]
        o._results.append(AgentResult("a", AgentStatus.PARTIAL, "partial", acceptance={"passed": False}))
        with recovery_scope(FAST):
            self.assertEqual(await o._do_collecting(), OrchestratorState.REPLANNING)
            self.assertEqual(await o._do_collecting(), OrchestratorState.SYNTHESIZING)
        self.assertEqual(o._replan_count, 1)

    async def test_exhausted_replan_without_evidence_stops_before_synthesis(self):
        o, pool = self.orchestrator([TASK], lambda: Policy([RuntimeError("controlled failure")]))
        report = await o.run("q", RunConfig(max_replan_rounds=1, enable_adversarial=True))
        self.assertEqual((report.status, report.recovery["code"]), ("failed", "NO_USABLE_EVIDENCE"))
        self.assertEqual(o.planner.replans, 1)
        self.assertNotIn("synthesize", pool.get_stats())

    async def test_unchanged_tasks_reused_and_changed_descendants_invalidated(self):
        a = TASK; b = replace(TASK, task_id="b", dependencies=["a"]); c = replace(TASK, task_id="c")
        o, pool = self.orchestrator([a, b, c])
        o._task_map = {t.task_id: t for t in (a, b, c)}
        o._results = [AgentResult(t.task_id, AgentStatus.SUCCESS, "old evidence", acceptance={"passed": True}) for t in (a, b, c)]
        for result in o._results:
            o._memory_store[f"result:{result.task_id}"] = result
            o._memory_store[f"compressed:{result.task_id}"] = "old compressed"
        o.planner.tasks = [replace(a, description="修改后的研究任务内容"), b, c]
        self.assertEqual(await o._do_replanning(), OrchestratorState.DISPATCHING)
        self.assertEqual(set(o._reusable_results), {"c"})
        self.assertNotIn("result:a", o._memory_store); self.assertNotIn("compressed:b", o._memory_store)
        await o._do_dispatching()
        self.assertEqual(next(r for r in o._results if r.task_id == "c").output, "old evidence")
        self.assertEqual(pool.get_stats()["search"]["created"], 1)

    async def test_planner_restores_accepted_definitions_for_dependencies(self):
        a = TASK; b = replace(TASK, task_id="b", task_type=TaskType.ANALYZE, dependencies=["a"])
        policy = Policy([message(content=json.dumps({"sub_tasks": [task_data(b)]}))])
        planner = Planner(policy, search_calls=0)
        planner._last_raw_json = json.dumps({"sub_tasks": [task_data(a), task_data(b)]})
        dag = await planner.areplan("q", [b], [AgentResult("a", AgentStatus.SUCCESS, "accepted", confidence=.1, acceptance={"passed": True})], "gap")
        self.assertEqual(len(dag), 2)
        self.assertEqual(set(planner.get_task_map_from_dag(dag, planner._last_raw_json)), {"a", "b"})
        self.assertEqual(policy.calls, 1)

    async def test_planner_and_schema_helper_do_not_multiply_transport_retries(self):
        from src.utils.schema_validator import retry_with_schema_feedback
        policy = Policy([ConnectionError("offline")]); planner = Planner(policy, search_calls=0)
        with recovery_scope(FAST), self.assertRaises(PlanParseError):
            await planner.generate_plan("q")
        self.assertEqual(policy.calls, 3)
        policy = Policy([ConnectionError("offline")])
        with recovery_scope(FAST):
            result = await retry_with_schema_feedback(lambda msgs: call_policy(policy, msgs), [], lambda value: None)
        self.assertIsNone(result); self.assertEqual(policy.calls, 3)

    async def test_single_task_plan_and_schema_fallback_cannot_bypass_validation(self):
        policy = Policy([message(content=json.dumps({"sub_tasks": [task_data(replace(TASK, timeout_seconds=.5))]}))])
        planner = Planner(policy, search_calls=0, max_sub_tasks=1)
        dag = await planner.generate_plan("q")
        self.assertEqual(len(dag), 1)
        self.assertEqual(planner.get_task_map_from_dag(dag, planner._last_raw_json)["a"].timeout_seconds, .5)
        planner = Planner(Policy([message(content='{"sub_tasks": [{"task_id": "x"}]}')]), search_calls=0)
        with recovery_scope(FAST), self.assertRaises(PlanParseError):
            await planner.generate_plan("q")
        self.assertIsNone(planner._try_parse_json("[]"))

    async def test_blue_search_failure_does_not_claim_fix_or_call_writer(self):
        from src.adversarial.blue_agent import BlueAgent
        from src.adversarial.verdict import Issue, Severity, Dimension, FixType
        blue = BlueAgent(Policy([message()]), [Tool([ConnectionError("offline")])])
        blue._call_fix_with_schema_retry = AsyncMock()
        issue = Issue(Severity.MAJOR, Dimension.COVERAGE, "补充证据", fix_type=FixType.SUPPLEMENTARY)
        with recovery_scope(FAST):
            result = await blue._do_supplementary_search(ResearchReport("q", "原始报告"), issue)
        self.assertFalse(result.success); blue._call_fix_with_schema_retry.assert_not_awaited()

    async def test_planner_search_and_blue_search_use_same_retries(self):
        from src.adversarial.blue_agent import BlueAgent
        from src.adversarial.verdict import Issue, Severity, Dimension, FixType
        tool = Tool([ConnectionError("offline"), EVIDENCE])
        planner = Planner(Policy([message()]), tools=[tool])
        with recovery_scope(FAST):
            entry = await planner._execute_search_tool(call())
        self.assertFalse(entry["error"]); self.assertEqual(tool.calls, 2)
        tool = Tool([ConnectionError("offline"), EVIDENCE]); blue = BlueAgent(Policy([message()]), [tool])
        blue._call_fix_with_schema_retry = AsyncMock(return_value=("修复后的内容", []))
        issue = Issue(Severity.MAJOR, Dimension.COVERAGE, "补充证据", fix_type=FixType.SUPPLEMENTARY)
        with recovery_scope(FAST):
            result = await blue._do_supplementary_search(ResearchReport("q", "报告"), issue)
        self.assertTrue(result.success); self.assertEqual(tool.calls, 2)

    async def test_l5_fatal_auth_cancels_sibling_and_drains_pool(self):
        started, cancelled = asyncio.Event(), asyncio.Event()
        class Blocking(Policy):
            async def acall(self, messages):
                started.set()
                try:
                    await asyncio.sleep(10)
                finally:
                    cancelled.set()
        class Fatal(Policy):
            async def acall(self, messages):
                await started.wait()
                raise RecoveryError(Failure(5, "LLM_AUTH_OR_QUOTA_ERROR", "HTTP 402", "backend"))
        policies = iter([Blocking([message()]), Fatal([message()])])
        o, pool = self.orchestrator([TASK, replace(TASK, task_id="b")], lambda: next(policies))
        report = await o.run("q", RunConfig(enable_adversarial=False))
        self.assertEqual((report.status, report.recovery["level"]), ("failed", 5))
        self.assertTrue(cancelled.is_set()); self.assertFalse(pool._leased)
        self.assertEqual(report.recovery["model_calls"], 2)

    async def test_l5_global_budget_preserves_completed_sibling(self):
        o, pool = self.orchestrator([TASK, replace(TASK, task_id="b")], config=replace(FAST, max_run_model_calls=1))
        report = await o.run("q", RunConfig(max_concurrent=1, enable_adversarial=False))
        self.assertEqual((report.status, report.recovery["code"]), ("partial", "MODEL_BUDGET_EXHAUSTED"))
        self.assertIn("根据来源得出结论", report.content); self.assertFalse(pool._leased)

    async def test_timeout_preserves_tool_evidence_before_task_finishes(self):
        class SlowFinish(Policy):
            async def acall(self, messages):
                if self.calls:
                    await asyncio.sleep(10)
                return await super().acall(messages)
        o, pool = self.orchestrator([TASK], lambda: SlowFinish([message(call())]), lambda: [Tool()])
        report = await o.run("q", RunConfig(global_timeout_seconds=.05, enable_adversarial=False))
        self.assertEqual(report.status, "timeout")
        self.assertIn("有来源的研究发现", report.content); self.assertFalse(pool._leased)


if __name__ == "__main__":
    unittest.main()
