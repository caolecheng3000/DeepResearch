"""Offline regression suite: python -m unittest discover -s tests -p test_runtime_safety.py."""
import asyncio
import copy
import json
import sqlite3
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
from openai import AsyncOpenAI as RealAsyncOpenAI

from src.memory.long_term import LongTermMemory, MemoryEntry
from src.memory.memory_store import SharedMemoryStore
from src.models.vllm_policy import VLLMPolicy, OpenAICompatibleDict
from src.orchestrator.agent_pool import AgentPool
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.schemas import AgentResult, AgentStatus, RunConfig, SubTask, TaskType, OrchestratorState
from src.planner.dag import DAG
from src.tools.notepad import NotepadTool
from src.utils.execution import bounded, call_policy, deadline_scope, remaining, run_sync


class Embedder:
    dim = 3

    def encode(self, text):
        return [1., 0., 0.]


def entry(key="same", confidence=.6, vector=None):
    return MemoryEntry(key, "这是一条用于并发一致性测试的完整研究结论，包含足够的上下文和可核查的信息。",
                       "test", confidence, "worker", 123., "primary", vector or [1., 0., 0.], "topic")


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.temp.name) / "memory.db")

    def tearDown(self):
        self.temp.cleanup()

    def store(self, session="a"):
        return SharedMemoryStore(self.path, Embedder(), session)

    def test_migrate_legacy_database_preserves_rows(self):
        with sqlite3.connect(self.path) as conn:
            conn.execute("""CREATE TABLE entries (entry_id TEXT PRIMARY KEY,claim TEXT NOT NULL,
                source TEXT NOT NULL,confidence REAL NOT NULL,agent_id TEXT NOT NULL,timestamp REAL NOT NULL,
                evidence_type TEXT NOT NULL,embedding_json TEXT NOT NULL,topic TEXT NOT NULL,metadata_json TEXT NOT NULL)""")
            conn.execute("INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?,?)",
                         ("legacy", "old", "source", .5, "a", 1., "primary", "[1,0,0]", "topic", "{}"))
        lt = LongTermMemory(self.path)
        self.assertEqual(lt.get_entry("legacy", "").claim, "old")
        lt.insert_entry(entry())
        other = entry(); other.session_id = "b"
        lt.insert_entry(other)
        self.assertEqual(lt.count_entries(), 3)
        # Migration is idempotent and preserves data across reopen.
        self.assertEqual(LongTermMemory(self.path).count_entries(), 3)
        with sqlite3.connect(self.path) as conn:
            pk = {row[1]: row[5] for row in conn.execute("PRAGMA table_info(entries)")}
        self.assertEqual((pk["session_id"], pk["entry_id"]), (1, 2))

    def test_same_id_across_sessions_and_scoped_dedup(self):
        a, b = self.store("a"), self.store("b")
        original = entry(confidence=.4)
        a.put(original)
        b.put(entry(confidence=.7))
        a.put(entry("new", confidence=.9))
        self.assertEqual(original.session_id, "")  # Input objects are not mutated.
        self.assertEqual(a.lt.get_entry("same", "a").confidence, .9)
        self.assertEqual(b.lt.get_entry("same", "b").confidence, .7)
        self.assertEqual((len(a), len(b)), (1, 1))
        self.assertEqual(b.query_by_similarity("q")[0][0].session_id, "b")

    def test_default_sessions_are_private(self):
        a, b = self.store(""), self.store("")
        self.assertNotEqual(a.session_id, b.session_id)
        a.put(entry())
        self.assertEqual(len(b), 0)

    def test_concurrent_dedup_across_store_instances(self):
        stores = [self.store() for _ in range(8)]
        barrier = threading.Barrier(8)
        def write(i):
            barrier.wait()
            return stores[i].put(entry(str(i), confidence=.3 + i * .08))
        with ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(pool.map(write, range(8)))
        self.assertEqual(len(set(ids)), 1)
        self.assertEqual(len(stores[0]), 1)
        latest = stores[0].query_by_similarity("q")[0][0]
        self.assertAlmostEqual(latest.confidence, .86)
        self.assertEqual(len(stores[0]._entry_ids), 1)

    def test_same_id_upsert_does_not_duplicate_index(self):
        a = self.store()
        a.put(entry(vector=[1., 0., 0.]))
        a.put(entry(vector=[0., 1., 0.]))
        self.assertEqual(a._entry_ids, ["same"])
        self.assertEqual(len(a), 1)

    def test_put_failure_does_not_publish_index(self):
        a = self.store(); a.put(entry())
        snapshot = copy.deepcopy(a._entries_cache)
        with patch.object(a.lt, "write_entry", side_effect=RuntimeError("disk failure")):
            with self.assertRaises(RuntimeError):
                a.put(entry("new", .9))
        self.assertEqual(a._entries_cache, snapshot)
        self.assertEqual(a.lt.get_entry("same", "a").confidence, .6)

    def test_compaction_rollback_and_session_boundary(self):
        a, b = self.store("a"), self.store("b")
        a.put(entry()); b.put(entry())
        originals = a.list_entries()
        with patch.object(a.lt, "write_entry", side_effect=RuntimeError("failed after delete")):
            with self.assertRaises(RuntimeError):
                a.replace_entries(originals, entry("summary"))
        self.assertIsNotNone(a.lt.get_entry("same", "a"))
        self.assertEqual(a._entry_ids, ["same"])
        self.assertTrue(a.replace_entries(originals, entry("summary")))
        self.assertEqual(a._entry_ids, ["summary"])
        self.assertIsNotNone(b.lt.get_entry("same", "b"))

    def test_compaction_rejects_changed_source_and_bad_summary(self):
        a = self.store(); a.put(entry())
        originals = a.list_entries()
        a.put(entry(confidence=.9))
        self.assertFalse(a.replace_entries(originals, entry("summary")))
        self.assertFalse(a.replace_entries(a.list_entries(), entry("summary", .1)))
        with self.assertRaises(ValueError):
            a.replace_entries(a.list_entries(), entry("summary", vector=[1, 2]))
        self.assertEqual(len(a), 1)

    def test_expired_transaction_rolls_back(self):
        a = self.store()
        with self.assertRaises(asyncio.TimeoutError):
            with deadline_scope(seconds=.01):
                with a.lt.transaction() as conn:
                    value = entry(); value.session_id = "a"
                    a.lt.write_entry(conn, value)
                    time.sleep(.02)
        self.assertEqual(len(a), 0)


class Policy:
    tools = None
    was_truncated = False

    def __init__(self, delay=0., error=False):
        self.delay, self.error = delay, error
        self.calls = 0
        self.cancelled = False

    async def acall(self, messages):
        self.calls += 1
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.error:
            raise RuntimeError("controlled failure")
        return OpenAICompatibleDict(content="证据与结论。" * 100, tool_calls=[])


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_nested_deadline_does_not_extend_parent(self):
        p = Policy(delay=1)
        with deadline_scope(seconds=.04):
            before = remaining()
            with deadline_scope(seconds=10):
                self.assertLessEqual(remaining(), before)
                with self.assertRaises(asyncio.TimeoutError):
                    await call_policy(p, [])
        self.assertTrue(p.cancelled)
        self.assertTrue(p.was_truncated)
        self.assertIsNone(remaining())

    async def test_expired_call_never_starts(self):
        p = Policy()
        with deadline_scope(seconds=0):
            with self.assertRaises(asyncio.TimeoutError):
                await call_policy(p, [])
        self.assertEqual(p.calls, 0)

    async def test_native_http_request_is_cancelled(self):
        cancelled = asyncio.Event()
        started = asyncio.Event()
        async def handler(request):
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise
        def client(**kwargs):
            return RealAsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        p = VLLMPolicy(base_url="http://unit.test/v1", api_key="test")
        try:
            with patch("src.models.vllm_policy.AsyncOpenAI", side_effect=client):
                task = asyncio.create_task(call_policy(p, [{"role": "user", "content": "q"}]))
                await asyncio.wait_for(started.wait(), 3)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertTrue(cancelled.is_set())
        finally:
            p.client.close()

    async def test_policy_clone_isolates_mutable_fields_and_parsing(self):
        p = VLLMPolicy(base_url="http://unit.test/v1", api_key="test", tools=[{"a": [1]}])
        try:
            a, b = p.clone(), p.clone()
            a.tools[0]["a"].append(2); a.max_tokens = 12; a.was_truncated = True
            self.assertEqual(b.tools, [{"a": [1]}])
            self.assertEqual(b.max_tokens, 1024)
            self.assertFalse(b.was_truncated)
            async def handler(request):
                return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 1,
                    "model": "test", "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"}}]})
            def client(**kwargs):
                return RealAsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
            with patch("src.models.vllm_policy.AsyncOpenAI", side_effect=client):
                result = await b.acall([("user", "q")])
            self.assertEqual(result.content, "ok")
        finally:
            p.client.close()

    async def test_cancelled_sync_memory_preparation_cannot_commit_late(self):
        started = threading.Event()
        class SlowEmbedder(Embedder):
            def encode(self, text):
                started.set(); time.sleep(.12)
                return super().encode(text)
        with tempfile.TemporaryDirectory() as temp:
            store = SharedMemoryStore(str(Path(temp) / "db"), SlowEmbedder(), "a")
            value = entry(); value.embedding = []
            task = asyncio.create_task(run_sync(store.put, value))
            while not started.is_set():
                await asyncio.sleep(.001)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await asyncio.sleep(.16)
            self.assertEqual(len(store), 0)

    async def test_pool_separates_worker_and_reused_task_state(self):
        pool = AgentPool(Policy, lambda: [NotepadTool()])
        a = await pool.get_agent(TaskType.ANALYZE)
        b = await pool.get_agent(TaskType.ANALYZE)
        self.assertIsNot(a.policy, b.policy)
        self.assertIsNot(a.tools[0], b.tools[0])
        old_policy, old_tool = a.policy, a.tools[0]
        await pool.release_agent(a)
        c = await pool.get_agent(TaskType.ANALYZE)
        self.assertIs(c, a)
        self.assertIsNot(c.policy, old_policy)
        self.assertIsNot(c.tools[0], old_tool)
        await pool.release_agent(c, reusable=False)
        await pool.release_agent(b, reusable=False)
        self.assertEqual(pool.get_stats()["analyze"]["active"], 0)
        self.assertNotIn("search", pool.get_stats())
        self.assertFalse(pool._leased)

    async def test_tool_cancellation_reaches_implementation(self):
        cancelled = asyncio.Event()
        class Tool:
            name = "blocking"
            async def execute(self):
                try:
                    await asyncio.sleep(10)
                finally:
                    cancelled.set()
        from src.agents.researcher import ResearcherAgent
        agent = ResearcherAgent("test", Policy(), [Tool()])
        with deadline_scope(seconds=.02):
            with self.assertRaises(asyncio.TimeoutError):
                await agent._execute_tool("blocking", {})
        self.assertTrue(cancelled.is_set())

    async def test_notepad_does_not_leak_between_tasks(self):
        pool = AgentPool(Policy, lambda: [NotepadTool()])
        a, b = await pool.get_agent(TaskType.SEARCH), await pool.get_agent(TaskType.SEARCH)
        await a.tools[0].execute("write", content="private")
        self.assertEqual(len(a.tools[0]._notes), 1)
        self.assertEqual(b.tools[0]._notes, [])
        await pool.release_agent(a)
        c = await pool.get_agent(TaskType.SEARCH)
        self.assertEqual(c.tools[0]._notes, [])
        await pool.release_agent(c)
        await pool.release_agent(b)

    async def test_web_search_sessions_are_instance_owned(self):
        from src.tools.web_search import WebSearchTool
        a, b = WebSearchTool(api_key="test"), WebSearchTool(api_key="test")
        first, second = a._get_session(), b._get_session()
        self.assertIsNot(first, second)
        await a.close_session()
        self.assertFalse(second.closed)
        await b.close_session()

    async def test_code_process_is_reaped_on_timeout_and_cancel(self):
        from src.tools.code_sandbox import CodeSandboxTool
        processes = []
        real_create = asyncio.create_subprocess_exec
        async def create(*args, **kwargs):
            proc = await real_create(*args, **kwargs)
            processes.append(proc)
            return proc
        tool = CodeSandboxTool(use_mock=False)
        # A deliberately tiny budget exercises the actual child-process cleanup.
        with patch("src.tools.code_sandbox.asyncio.create_subprocess_exec", side_effect=create):
            result = await tool.execute("2 ** 1000000000", timeout=.01)
            self.assertFalse(result["success"])
            self.assertIsNotNone(processes[-1].returncode)
            task = asyncio.create_task(tool.execute("2 ** 1000000000", timeout=10))
            while len(processes) < 2:
                await asyncio.sleep(.001)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertIsNotNone(processes[-1].returncode)
        result = await tool.execute("1 + 1", timeout=2)
        self.assertEqual(result["return_value"], 2)

    async def test_async_compression_cancels_llm(self):
        from src.compressor.compressor import ContextCompressor
        policy = Policy(delay=10)
        compressor = ContextCompressor(policy, Embedder(), budget=100, output_reserve=20)
        with patch.object(compressor, "_l1_filter", side_effect=lambda texts, *args: texts), \
             patch.object(compressor, "_l2_extract", side_effect=lambda texts, *args: texts):
            with deadline_scope(seconds=.04):
                with self.assertRaises(asyncio.TimeoutError):
                    await bounded(compressor.acompress(["材料与证据。" * 200], "q", level=3))
        self.assertTrue(policy.cancelled)

    async def test_runner_factories_isolate_roles_workers_and_tools(self):
        from src.core.runner import initialize_modules
        from src.models.model_router import ModelRouter
        prototype = VLLMPolicy(base_url="http://unit.test/v1", api_key="test")
        try:
            with tempfile.TemporaryDirectory() as temp:
                config = {"tools": {"web_search": {"mock_mode": True}},
                          "memory": {"db_path": str(Path(temp) / "db")}}
                with patch.object(ModelRouter, "create_backend", return_value=prototype):
                    modules = initialize_modules(config)
                    other = initialize_modules(config)
                self.assertNotEqual(modules["session_id"], other["session_id"])
                policies = [modules[f"{r}_policy"] for r in
                            ("planner", "solver", "summarizer", "red_agent", "blue_agent", "compressor")]
                self.assertEqual(len({id(p) for p in policies}), 6)
                pool = modules["agent_pool"]
                a = await pool.get_agent(TaskType.SEARCH)
                b = await pool.get_agent(TaskType.ANALYZE)
                self.assertIsNot(a.policy, b.policy)
                self.assertTrue(all(x is not y for x, y in zip(a.tools, b.tools)))
                await pool.release_agent(a)
                await pool.release_agent(b)
        finally:
            prototype.client.close()

    async def test_failed_factory_does_not_leak_pool_active_count(self):
        pool = AgentPool(Policy, lambda: [])
        agent = await pool.get_agent(TaskType.ANALYZE)
        await pool.release_agent(agent)
        with patch.object(pool, "policy_factory", side_effect=RuntimeError("factory failed")):
            with self.assertRaises(RuntimeError):
                await pool.get_agent(TaskType.ANALYZE)
        self.assertEqual(pool.get_stats()["analyze"]["active"], 0)
        self.assertFalse(pool._leased)


class Planner:
    _last_raw_json = "{}"

    def __init__(self, tasks, delay=0):
        self.tasks, self.delay = tasks, delay
        self.dag = DAG()
        for t in tasks:
            self.dag.add_node(t.task_id)
            for dep in t.dependencies:
                self.dag.add_edge(dep, t.task_id)

    async def generate_plan(self, query, memory_context):
        await asyncio.sleep(self.delay)
        return self.dag

    def get_task_map_from_dag(self, *args):
        return {t.task_id: t for t in self.tasks}


class OrchestratorTests(unittest.IsolatedAsyncioTestCase):
    def make(self, tasks, policy_factory=Policy, planner_delay=0, **kwargs):
        pool = AgentPool(policy_factory, lambda: [])
        return Orchestrator(Planner(tasks, planner_delay), pool, **kwargs), pool

    async def test_failed_dependency_skips_transitive_chain(self):
        tasks = [SubTask("a", TaskType.SEARCH, "collect"),
                 SubTask("b", TaskType.ANALYZE, "analyze", ["a"]),
                 SubTask("c", TaskType.ANALYZE, "analyze", ["b"])]
        o, pool = self.make(tasks, lambda: Policy(error=True))
        o._dag, o._task_map = o.planner.dag, {t.task_id: t for t in tasks}
        await o._do_dispatching()
        self.assertEqual([r.status for r in o._results], [AgentStatus.FAILED, AgentStatus.SKIPPED, AgentStatus.SKIPPED])
        self.assertEqual(pool.get_stats()["search"]["created"], 1)
        self.assertNotIn("analyze", pool.get_stats())

    async def test_missing_dependency_skips_task(self):
        task = SubTask("b", TaskType.ANALYZE, "analyze", ["missing"])
        o, _ = self.make([task])
        o._dag, o._task_map = o.planner.dag, {task.task_id: task}
        await o._do_dispatching()
        self.assertEqual(o._results[-1].status, AgentStatus.SKIPPED)

    async def test_local_timeout_discards_agent_and_skips_child(self):
        tasks = [SubTask("a", TaskType.SEARCH, "collect", timeout_seconds=.025),
                 SubTask("b", TaskType.ANALYZE, "analyze", ["a"])]
        o, pool = self.make(tasks, lambda: Policy(delay=2))
        o._dag, o._task_map = o.planner.dag, {t.task_id: t for t in tasks}
        await o._do_dispatching()
        self.assertEqual([r.status for r in o._results], [AgentStatus.TIMEOUT, AgentStatus.SKIPPED])
        self.assertEqual(pool.get_stats()["search"]["idle"], 0)
        self.assertFalse(pool._leased)

    async def test_planning_deadline_returns_without_new_calls(self):
        o, pool = self.make([], planner_delay=2)
        start = time.monotonic()
        report = await o.run("q", RunConfig(global_timeout_seconds=.04))
        self.assertLess(time.monotonic() - start, .5)
        self.assertIn("总时间上限", report.content)
        self.assertFalse(pool.get_stats())

    async def test_global_deadline_preserves_completed_sibling(self):
        policies = []
        def factory():
            p = Policy(delay=0 if not policies else 2); policies.append(p); return p
        tasks = [SubTask("a", TaskType.SEARCH, "collect"), SubTask("b", TaskType.SEARCH, "collect")]
        o, pool = self.make(tasks, factory)
        report = await o.run("q", RunConfig(global_timeout_seconds=.07, enable_adversarial=False))
        self.assertIn("## a", report.content)
        self.assertTrue(policies[-1].cancelled)
        self.assertEqual(pool.get_stats()["search"]["active"], 0)
        self.assertFalse(pool._leased)

    async def test_synthesis_deadline_drains_lease(self):
        o, pool = self.make([SubTask("a", TaskType.SEARCH, "collect")])
        pool.summarizer_policy_factory = lambda: Policy(delay=2)
        report = await o.run("q", RunConfig(global_timeout_seconds=.07, enable_adversarial=False))
        self.assertIn("总时间上限", report.content)
        self.assertFalse(pool._leased)
        self.assertEqual(pool.get_stats()["synthesize"]["active"], 0)

    async def test_external_cancel_propagates_and_drains_workers(self):
        o, pool = self.make([SubTask("a", TaskType.SEARCH, "collect")], lambda: Policy(delay=2))
        task = asyncio.create_task(o.run("q", RunConfig(global_timeout_seconds=10)))
        while not pool._leased:
            await asyncio.sleep(.001)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(o._running)
        self.assertFalse(pool._leased)
        self.assertEqual(pool.get_stats()["search"]["active"], 0)

    async def test_same_orchestrator_rejects_overlapping_runs(self):
        o, _ = self.make([], planner_delay=2)
        task = asyncio.create_task(o.run("first", RunConfig(global_timeout_seconds=.06)))
        await asyncio.sleep(.01)
        with self.assertRaises(RuntimeError):
            await o.run("second")
        await task

    async def test_zero_budget_cannot_reuse_previous_report(self):
        o, _ = self.make([])
        from src.orchestrator.schemas import ResearchReport
        o._memory_store["final_report"] = ResearchReport(query="old", content="old")
        report = await o.run("new", RunConfig(global_timeout_seconds=0))
        self.assertEqual(report.query, "new")
        self.assertNotIn("old", report.content)

    async def test_adversarial_deadline_preserves_synthesized_report(self):
        class Loop:
            cancelled = False
            async def run(self, report):
                try:
                    await asyncio.sleep(10)
                finally:
                    self.cancelled = True
        loop = Loop()
        o, pool = self.make([SubTask("a", TaskType.SEARCH, "collect")], adversarial_loop=loop)
        report = await o.run("q", RunConfig(global_timeout_seconds=.08, enable_adversarial=True))
        self.assertTrue(loop.cancelled)
        self.assertIn("证据", report.content)
        self.assertFalse(pool._leased)

    async def test_successful_runs_namespace_memory_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            # Orthogonal embeddings avoid semantic dedup masking ID collisions.
            class DifferentEmbedder(Embedder):
                def encode(self, text):
                    return [0., 1., 0.] if "second" in text else [1., 0., 0.]
            store = SharedMemoryStore(str(Path(temp) / "db"), DifferentEmbedder(), "session")
            o, pool = self.make([SubTask("a", TaskType.SEARCH, "collect")], memory_store=store)
            first = await o.run("first", RunConfig(global_timeout_seconds=5, enable_adversarial=False))
            first_id = o._run_id
            self.assertIn("证据", first.content)
            self.assertTrue(any(e.entry_id.startswith(first_id) for e in store.list_entries()))
            await o.run("second", RunConfig(global_timeout_seconds=5, enable_adversarial=False))
            self.assertNotEqual(first_id, o._run_id)
            self.assertFalse(pool._leased)


if __name__ == "__main__":
    unittest.main()
