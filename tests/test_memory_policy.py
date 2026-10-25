"""Offline memory semantics, retention and transaction integration regressions."""
import copy
import sqlite3
import tempfile
import threading
import unittest
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from src.memory.long_term import MemoryEntry
from src.memory.memory_store import MemoryCapacityError, SharedMemoryStore


class Embedder:
    dim = 3

    def encode(self, text):
        return [1., 0., 0.]


def entry(key, claim=None, vector=None, **metadata):
    return MemoryEntry(key, claim or f"研究对象{key}的资料描述与其他对象有所区别，这里保存足够完整的原始结论与来源说明。",
                       "https://example.test/source", .6, "worker", 1000., "primary",
                       vector or [1., 0., 0.], "topic", metadata)


POSITIVE = "研究资料表明，该产品支持离线部署，因此可以在本地服务器环境运行并完成相关研究任务。"
NEGATIVE = "研究资料表明，该产品不支持离线部署，因此可以在本地服务器环境运行并完成相关研究任务。"


class MemoryPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.temp.name) / "memory.db")

    def tearDown(self):
        self.temp.cleanup()

    def store(self, session="a", **kwargs):
        return SharedMemoryStore(self.path, Embedder(), session, **kwargs)

    def test_opposite_claims_stay_disputed_even_with_higher_confidence(self):
        store = self.store()
        old = entry("old", POSITIVE)
        store.put(old)
        incoming = entry("new", NEGATIVE); incoming.confidence = .99
        store.put(incoming)
        self.assertEqual(len(store), 2)
        self.assertTrue(all(e.metadata["_memory"]["status"] == "disputed" for e in store.list_entries()))
        self.assertIn("未裁决", store.get_context_for_query("q"))
        self.assertNotIn("_memory", incoming.metadata)
        self.assertEqual(store.get_history()[0]["reason"], "conflict_unresolved")

    def test_source_authority_resolves_and_archives_full_old_version(self):
        store = self.store()
        store.put(entry("old", POSITIVE, source_trust=.4))
        new = entry("new", NEGATIVE, source_trust=.9)
        store.put(new)
        self.assertEqual([e.entry_id for e in store.list_entries()], ["new"])
        history = store.get_history()[0]
        self.assertEqual(history["reason"], "conflict_superseded")
        self.assertEqual(history["entry"].claim, POSITIVE)
        self.assertEqual(history["entry"].source, "https://example.test/source")
        self.assertEqual(history["entry"].embedding, [1., 0., 0.])

    def test_weaker_incoming_is_archived_and_existing_id_returned(self):
        store = self.store(); store.put(entry("old", POSITIVE, source_trust=.9))
        result = store.put(entry("new", NEGATIVE, source_trust=.2))
        self.assertEqual(result, "old")
        self.assertEqual(len(store), 1)
        self.assertEqual(store.get_history()[0]["entry"].claim, NEGATIVE)
        self.assertEqual(store.get_history()[0]["reason"], "conflict_rejected")

    def test_primary_evidence_beats_inference_without_using_confidence(self):
        store = self.store()
        old = entry("old", POSITIVE); old.evidence_type = "inference"; old.confidence = .99
        store.put(old)
        store.put(entry("new", NEGATIVE))
        self.assertEqual(store.list_entries()[0].entry_id, "new")

    def test_structured_fact_conflict_bypasses_embedding_candidate_threshold(self):
        store = self.store()
        store.put(entry("old", fact={"subject": "product", "predicate": "offline", "value": True}))
        store.put(entry("new", vector=[0., 1., 0.], fact={"subject": "product", "predicate": "offline", "value": False}))
        self.assertEqual(len(store), 2)
        self.assertTrue(all(e.metadata["_memory"]["status"] == "disputed" for e in store.list_entries()))

    def test_fact_scope_keeps_versions_and_conditions_separate(self):
        store = self.store()
        store.put(entry("v1", fact={"subject": "product", "predicate": "offline", "value": False, "scope": {"version": 1}}))
        store.put(entry("v2", fact={"subject": "product", "predicate": "offline", "value": True, "scope": {"version": 2}}))
        self.assertEqual(len(store), 2)
        self.assertTrue(all("_memory" not in e.metadata for e in store.list_entries()))

    def test_explicit_fact_time_updates_and_late_stale_fact_cannot_replace_it(self):
        store = self.store()
        def version(key, value, when):
            return entry(key, fact={"subject": "product", "predicate": "offline", "value": value, "valid_from": when})
        store.put(version("v1", False, "2025-01-01"))
        store.put(version("v2", True, "2026-01-01"))
        self.assertEqual([e.entry_id for e in store.list_entries()], ["v2"])
        self.assertEqual(store.put(version("stale", False, "2024-01-01")), "v2")
        self.assertEqual(len(store), 1)

    def test_different_numbers_in_matching_frame_are_disputed(self):
        store = self.store()
        claim = "当前资料中的同一测试环境下，该系统每秒处理100个请求，测试条件与部署方式保持一致。"
        store.put(entry("a", claim))
        store.put(entry("b", claim.replace("100", "200")))
        self.assertEqual(len(store), 2)
        self.assertTrue(all(e.metadata["_memory"]["status"] == "disputed" for e in store.list_entries()))

    def test_high_similarity_unknown_relation_does_not_delete_content(self):
        store = self.store()
        store.put(entry("a")); store.put(entry("b"))
        self.assertEqual(len(store), 2)  # Same fake vector does not prove equivalent claims.
        self.assertFalse(store.get_history())

    def test_entity_identifier_digits_do_not_become_numeric_conflicts(self):
        store = self.store()
        store.put(entry("product1")); store.put(entry("product2"))
        self.assertFalse(store.get_conflicts())
        self.assertFalse(store.get_history())

    def test_equivalent_fact_prefers_better_source_before_confidence(self):
        store = self.store()
        old = entry("old", POSITIVE, source_trust=.2); old.confidence = .99
        store.put(old)
        new = entry("new", POSITIVE, source_trust=.9); new.confidence = .4
        store.put(new)
        self.assertEqual(len(store), 1)
        self.assertEqual(store.list_entries()[0].metadata["source_trust"], .9)

    def test_duplicate_threshold_is_configurable(self):
        claim = POSITIVE
        strict = self.store("strict", similarity_threshold_dup=.95)
        loose = self.store("loose", similarity_threshold_dup=.8)
        for store in (strict, loose):
            store.put(entry("a", claim))
            store.put(entry("b", claim, vector=[.9, .435889894, 0.]))
        self.assertEqual(len(strict), 2)
        self.assertEqual(len(loose), 1)

    def test_conflict_threshold_is_configurable_for_free_text(self):
        strict = self.store("strict", similarity_threshold_conflict=.9)
        loose = self.store("loose", similarity_threshold_conflict=.7)
        for store in (strict, loose):
            store.put(entry("a", POSITIVE))
            store.put(entry("b", NEGATIVE, vector=[.8, .6, 0.]))
        self.assertTrue(all("_memory" not in e.metadata for e in strict.list_entries()))
        self.assertTrue(all(e.metadata["_memory"]["status"] == "disputed" for e in loose.list_entries()))

    def test_same_id_opposite_claim_is_not_an_implicit_overwrite(self):
        store = self.store(); store.put(entry("same", POSITIVE))
        new_id = store.put(entry("same", NEGATIVE))
        self.assertNotEqual(new_id, "same")
        self.assertEqual(len(store), 2)

    def test_same_id_same_text_different_structured_value_retains_both(self):
        store = self.store()
        store.put(entry("same", POSITIVE, fact={"subject": "p", "predicate": "offline", "value": True}))
        returned = store.put(entry("same", POSITIVE, fact={"subject": "p", "predicate": "offline", "value": False}))
        self.assertNotEqual(returned, "same")
        self.assertEqual(len(store.get_conflicts()), 2)
        context = store.get_context_for_query("q")
        self.assertIn("offline = true", context)
        self.assertIn("offline = false", context)

    def test_duplicate_reinforcement_preserves_pin(self):
        store = self.store(max_entries=2)
        store.put(entry("old", POSITIVE, pinned=True))
        new = entry("new", POSITIVE); new.confidence = .9
        store.put(new)
        self.assertTrue(store.list_entries()[0].metadata["pinned"])

    def test_automatic_resolution_clears_old_dispute_annotation(self):
        store = self.store()
        store.put(entry("a", POSITIVE)); store.put(entry("b", NEGATIVE))
        store.put(entry("verified", POSITIVE, source_trust=.9))
        self.assertEqual(len(store), 1)
        self.assertFalse(store.get_conflicts())

    def test_manual_resolution_requires_linked_dispute_and_records_review(self):
        store = self.store(); store.put(entry("a", POSITIVE)); store.put(entry("b", NEGATIVE))
        store.resolve_conflict("b", ["a"], reason="核对了原始来源及版本")
        self.assertEqual(len(store), 1)
        self.assertEqual(store.list_entries()[0].metadata["_memory"]["status"], "active")
        self.assertEqual(store.get_history()[0]["details"]["reason"], "核对了原始来源及版本")
        store.put(entry("unrelated"))
        with self.assertRaises(ValueError):
            store.resolve_conflict("b", ["unrelated"], reason="not a dispute")

    def test_unresolved_disputes_cannot_be_compacted_into_one_summary(self):
        store = self.store(); store.put(entry("a", POSITIVE)); store.put(entry("b", NEGATIVE))
        self.assertFalse(store.replace_entries(store.list_entries(), entry("summary")))
        self.assertEqual(len(store), 2)

    def test_lru_access_is_persistent_across_instances_and_sessions(self):
        store = self.store(max_entries=2)
        with patch("src.memory.memory_store.time.time", return_value=1000):
            store.put(entry("a"))
        with patch("src.memory.memory_store.time.time", return_value=1001):
            store.put(entry("b", vector=[0., 1., 0.]))
        with patch("src.memory.memory_store.time.time", return_value=1002):
            store.query_by_similarity("a", top_k=1)
        other = self.store(max_entries=2)
        with patch("src.memory.memory_store.time.time", return_value=1003):
            other.put(entry("c", vector=[0., 0., 1.]))
        self.assertEqual({e.entry_id for e in other.list_entries()}, {"a", "c"})
        self.assertEqual(other.get_history()[0]["entry_id"], "b")
        with sqlite3.connect(self.path) as conn:
            self.assertEqual(conn.execute("SELECT access_count FROM memory_access WHERE session_id='a' AND entry_id='a'").fetchone()[0], 1)
        isolated = self.store("other", max_entries=1)
        isolated.put(entry("x")); isolated.put(entry("y"))
        self.assertEqual(len(other), 2)

    def test_all_pinned_capacity_rejects_and_rolls_back_new_write(self):
        store = self.store(max_entries=1)
        store.put(entry("pinned", pinned=True))
        with self.assertRaises(MemoryCapacityError):
            store.put(entry("new"))
        self.assertEqual([e.entry_id for e in store.list_entries()], ["pinned"])
        self.assertEqual(store._entry_ids, ["pinned"])
        self.assertFalse(store.get_history())

    def test_archived_dispute_can_be_explicitly_resolved(self):
        store = self.store(max_entries=1)
        store.put(entry("a", POSITIVE)); store.put(entry("b", NEGATIVE))
        self.assertEqual(len(store.get_conflicts()), 1)
        store.resolve_conflict("b", ["a"], reason="已核验来源，选择当前版本")
        self.assertFalse(store.get_conflicts())
        self.assertTrue(store.get_history()[0]["details"]["peer_already_archived"])

    def test_periodic_maintenance_uses_configured_write_interval(self):
        store = self.store(evict_interval=2)
        with sqlite3.connect(self.path) as conn:
            conn.execute("INSERT INTO memory_access VALUES ('a','orphan',1,0)")
        store.put(entry("a"))
        with sqlite3.connect(self.path) as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM memory_access WHERE entry_id='orphan'").fetchone()[0], 1)
        store.put(entry("b"))
        with sqlite3.connect(self.path) as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM memory_access WHERE entry_id='orphan'").fetchone()[0], 0)

    def test_idle_ttl_and_explicit_expiry_are_enforced_on_query(self):
        with patch("src.memory.memory_store.time.time", return_value=1000):
            store = self.store(ttl_seconds=10)
            store.put(entry("a"))
            explicit = entry("b", expires_at=1005.)
            store.put(explicit)
        with patch("src.memory.memory_store.time.time", return_value=1006):
            results = store.query_by_similarity("q")
        self.assertEqual([e.entry_id for e, _ in results], ["a"])
        with patch("src.memory.memory_store.time.time", return_value=1015):
            self.assertEqual(len(store.query_by_similarity("q")), 1)
        with patch("src.memory.memory_store.time.time", return_value=1026):
            self.assertEqual(store.query_by_similarity("q"), [])
        self.assertEqual(len(store), 0)

    def test_history_is_bounded_and_preserves_compaction_sources(self):
        store = self.store(max_entries=1, max_history_entries=2)
        for i in range(4):
            store.put(entry(str(i)))
        self.assertEqual(len(store.get_history(100)), 2)
        self.assertTrue(store.replace_entries(store.list_entries(), entry("summary")))
        history = store.get_history(100)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["reason"], "compacted")
        self.assertEqual(history[0]["entry"].entry_id, "3")

    def test_archive_failure_rolls_back_eviction_and_index_publication(self):
        store = self.store(max_entries=1); store.put(entry("old"))
        snapshot = copy.deepcopy(store._entries_cache)
        with patch.object(store, "_archive", side_effect=RuntimeError("history disk failure")):
            with self.assertRaises(RuntimeError):
                store.put(entry("new"))
        self.assertEqual(store._entries_cache, snapshot)
        self.assertEqual([e.entry_id for e in store.list_entries()], ["old"])

    def test_concurrent_capacity_writes_never_exceed_bound(self):
        stores = [self.store(max_entries=3) for _ in range(8)]
        barrier = threading.Barrier(8)
        def write(index):
            barrier.wait()
            stores[index].put(entry(str(index)))
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write, range(8)))
        self.assertEqual(len(stores[0]), 3)
        self.assertEqual(len(stores[0].get_history(100)), 5)

    def test_disabling_vector_index_preserves_retrieval_results(self):
        store = self.store(enable_vector_index=False)
        store.put(entry("a")); store.put(entry("b", vector=[0., 1., 0.]))
        self.assertIsNone(store._embeddings)
        self.assertEqual([e.entry_id for e, _ in store.query_by_similarity("q")], ["a"])

    def test_reopen_enforces_smaller_capacity_and_keeps_history(self):
        store = self.store()
        for i in range(4):
            store.put(entry(str(i)))
        reopened = self.store(max_entries=2)
        self.assertEqual(len(reopened), 2)
        self.assertEqual(len(reopened.get_history()), 2)

    def test_invalid_options_and_facts_fail_without_writes(self):
        for kwargs in ({"max_entries": 0}, {"evict_interval": 0}, {"max_history_entries": -1},
                       {"similarity_threshold_conflict": .99}, {"ttl_seconds": -1},
                       {"enable_vector_index": "false"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.store(**kwargs)
        store = self.store()
        for metadata in ({"fact": {"subject": "a"}}, {"source_trust": 2}, {"expires_at": "yesterday"},
                         {"fact": {"subject": "p", "predicate": "v", "value": 1, "valid_from": "2099-01-01"}}):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                store.put(entry("bad", **metadata))
        self.assertEqual(len(store), 0)


class MemoryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_runner_passes_all_memory_settings(self):
        from src.core.runner import initialize_modules
        from src.models.model_router import ModelRouter
        from src.models.vllm_policy import VLLMPolicy
        policy = VLLMPolicy(base_url="http://unit.test/v1", api_key="test")
        try:
            with tempfile.TemporaryDirectory() as temp:
                settings = {"db_path": str(Path(temp) / "db"), "max_entries": 31,
                            "similarity_threshold_dup": .88, "similarity_threshold_conflict": .61,
                            "evict_interval": 7, "enable_vector_index": False,
                            "ttl_seconds": 7200, "max_history_entries": 17}
                config = {"memory": settings, "tools": {"web_search": {"mock_mode": True}}}
                with patch.object(ModelRouter, "create_backend", return_value=policy):
                    modules = initialize_modules(config)
                store = modules["memory_store"]
                for key, value in settings.items():
                    if key != "db_path":
                        self.assertEqual(getattr(store, key), value)
                self.assertIsNone(store._embeddings)
        finally:
            policy.client.close()

    async def test_orchestrator_uses_configured_capacity_and_excludes_disputed_pinned(self):
        from src.orchestrator.orchestrator import Orchestrator
        texts_seen = []
        async def summarize(texts, **kwargs):
            texts_seen.extend(texts)
            return "整理后的研究结论，保留来源、条件与数字，用于后续研究背景。" * 10
        with tempfile.TemporaryDirectory() as temp:
            store = SharedMemoryStore(str(Path(temp) / "db"), Embedder(), "a", max_entries=28)
            for i in range(22):
                store.put(entry(str(i)))
            for key in ("pinned1", "pinned2"):
                store.put(entry(key, pinned=True))
            store.put(entry("positive", POSITIVE)); store.put(entry("negative", NEGATIVE))
            orchestrator = Orchestrator(None, None, memory_store=store,
                                        compressor=SimpleNamespace(summarizer=SimpleNamespace(asummarize_documents=summarize)))
            await orchestrator._maybe_compress_old_memories()
            self.assertEqual(len(texts_seen), 11)
            self.assertNotIn(POSITIVE, texts_seen)
            self.assertNotIn(NEGATIVE, texts_seen)
            self.assertTrue(all("pinned" not in text for text in texts_seen))
            self.assertEqual(len(store), 16)
            self.assertEqual(len(store.get_conflicts()), 2)
            self.assertEqual(sum(h["reason"] == "compacted" for h in store.get_history(100)), 11)


if __name__ == "__main__":
    unittest.main()
