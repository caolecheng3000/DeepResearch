"""
Shared Memory Store 模块：跨 Agent 统一读写接口

设计决策：
1. 持久层用 LongTermMemory（SQLite），内存层维护 numpy 向量索引，读写平衡
2. 可配置相似度候选、事实/否定/数值关系判别，争议保留与人工裁决
3. 会话内写入使用 SQLite 写事务，摘要替换支持源版本检查和回滚
4. get_context_for_query 将相关记忆组装为文本，控制总 token 在预算内
5. 写事务内强制会话容量，持久化 LRU/过期清理和有界历史归档
"""

from __future__ import annotations

import copy
import json
import math
import uuid
import logging
import re
import threading
import time
from typing import Optional

import numpy as np

from src.memory.embedder import Embedder
from src.memory.long_term import LongTermMemory, MemoryEntry
from src.memory.policy import authority, canonical, fact, prefer, relation, timestamp
from src.utils.tracing import trace_retriever, record_event

logger = logging.getLogger(__name__)

class MemoryCapacityError(RuntimeError):
    """Capacity cannot be respected without removing a protected/pinned entry."""

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的 cosine similarity。"""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SharedMemoryStore:
    """
    跨 Agent 共享记忆存储。

    提供 put/query 和事务化摘要替换接口，
    底层通过 LongTermMemory 做 SQLite 持久化，内存中维护 numpy 向量索引加速相似度查询。
    """

    def __init__(
        self,
        db_path: str = "memory.db",
        embedder: Optional[Embedder] = None,
        session_id: str = "",
        *,
        max_entries: int = 10000,
        similarity_threshold_dup: float = 0.92,
        similarity_threshold_conflict: float = 0.65,
        evict_interval: int = 100,
        enable_vector_index: bool = True,
        ttl_seconds: float | None = None,
        max_history_entries: int = 10000,
    ) -> None:
        """
        初始化共享记忆存储。

        Args:
            db_path: SQLite 数据库路径
            embedder: 向量化器，None 时自动创建
            session_id: 会话 ID；未提供时生成独立会话，不自动加载其他会话
        """
        for name, value, minimum in (("max_entries", max_entries, 1),
                                     ("evict_interval", evict_interval, 1),
                                     ("max_history_entries", max_history_entries, 0)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in (similarity_threshold_conflict, similarity_threshold_dup))
                or not 0 <= similarity_threshold_conflict <= similarity_threshold_dup <= 1):
            raise ValueError("Expected 0 <= conflict threshold <= duplicate threshold <= 1")
        if not isinstance(enable_vector_index, bool):
            raise ValueError("enable_vector_index must be boolean")
        if ttl_seconds is not None and (isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float))
                                        or not math.isfinite(ttl_seconds) or ttl_seconds <= 0):
            raise ValueError("ttl_seconds must be positive or None")
        self.max_entries = max_entries
        self.similarity_threshold_dup = similarity_threshold_dup
        self.similarity_threshold_conflict = similarity_threshold_conflict
        self.evict_interval = evict_interval
        self.enable_vector_index = enable_vector_index
        self.ttl_seconds = ttl_seconds
        self.max_history_entries = max_history_entries
        self.lt = LongTermMemory(db_path=db_path)
        self.embedder = embedder or Embedder()
        self._lock = threading.RLock()
        self.session_id = session_id or uuid.uuid4().hex

        # 内存向量索引
        self._entry_ids: list[str] = []
        self._embeddings: np.ndarray | None = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._entries_cache: dict[str, MemoryEntry] = {}

        # 加载已有数据到内存索引（按 session 过滤）
        self.maintain()

    # ------------------------------------------------------------------
    # 内部索引管理
    # ------------------------------------------------------------------

    def _index_snapshot(self, entries):
        ids = [e.entry_id for e in entries]
        cache = {e.entry_id: copy.deepcopy(e) for e in entries}
        if not self.enable_vector_index:
            for entry in entries:
                vec = np.asarray(entry.embedding, dtype=np.float32)
                if vec.shape != (self.embedder.dim,) or not np.isfinite(vec).all():
                    raise ValueError("Invalid memory embedding dimensions or values")
            return ids, None, cache
        mat = np.array([e.embedding for e in entries], dtype=np.float32) if entries else np.zeros((0, self.embedder.dim), dtype=np.float32)
        if mat.shape != (len(entries), self.embedder.dim) or not np.isfinite(mat).all():
            raise ValueError("Invalid memory embedding dimensions or values")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        return ids, mat / norms, cache

    def _install_index(self, snapshot):
        self._entry_ids, self._embeddings, self._entries_cache = snapshot

    def _rebuild_index(self):
        with self._lock:
            self._install_index(self._index_snapshot(self.lt.get_all_entries(session_id=self.session_id)))

    def list_entries(self):
        return self.lt.get_all_entries(session_id=self.session_id)

    def get_conflicts(self):
        """Return active disputed entries, with peer IDs and detection reasons."""
        self.maintain()
        return [e for e in self.list_entries() if e.metadata.get("_memory", {}).get("status") == "disputed"]

    def _prepare_entry(self, entry):
        entry = copy.deepcopy(entry)
        entry.session_id = self.session_id
        if not math.isfinite(entry.confidence) or not 0 <= entry.confidence <= 1:
            raise ValueError("confidence must be finite and between 0 and 1")
        if not math.isfinite(entry.timestamp):
            raise ValueError("timestamp must be finite")
        entry.metadata.pop("_memory", None)  # Store-owned conflict annotations.
        if "pinned" in entry.metadata and not isinstance(entry.metadata["pinned"], bool):
            raise ValueError("metadata.pinned must be boolean")
        structured = fact(entry)
        if structured is not None and structured[2] is not None and structured[2] > time.time():
            raise ValueError("fact.valid_from cannot be in the future for current memory")
        authority(entry)
        expires = timestamp(entry.metadata.get("expires_at"))
        if expires is not None and expires <= time.time():
            raise ValueError("Cannot insert an already expired memory")
        if not entry.embedding:
            entry.embedding = self.embedder.encode(entry.claim)
        # Validate before entering a transaction or deleting anything.
        self._index_snapshot([entry])
        return entry

    def _read_entries(self, conn):
        return [MemoryEntry.from_row(row) for row in conn.execute(
            "SELECT * FROM entries WHERE session_id=?", (self.session_id,))]

    def _access(self, conn):
        return {row["entry_id"]: (row["last_access"], row["access_count"]) for row in conn.execute(
            "SELECT * FROM memory_access WHERE session_id=?", (self.session_id,))}

    def _archive(self, conn, entry, reason, details, now):
        conn.execute("""INSERT INTO memory_history
            (history_id,session_id,entry_id,archived_at,reason,entry_json,details_json)
            VALUES (?,?,?,?,?,?,?)""", (uuid.uuid4().hex, self.session_id, entry.entry_id,
            now, reason, json.dumps(entry.to_dict(), ensure_ascii=False),
            json.dumps(details, ensure_ascii=False)))

    def _prune_history(self, conn):
        conn.execute("""DELETE FROM memory_history WHERE history_id IN
            (SELECT history_id FROM memory_history WHERE session_id=?
             ORDER BY archived_at DESC,rowid DESC LIMIT -1 OFFSET ?)""",
            (self.session_id, self.max_history_entries))

    def _expired(self, entry, access, now):
        expires = timestamp(entry.metadata.get("expires_at"))
        if expires is not None and now >= expires:
            return True
        last_access = access.get(entry.entry_id, (entry.timestamp, 0))[0]
        return self.ttl_seconds is not None and now - last_access >= self.ttl_seconds

    def _limit(self, conn, entries, events, now, *, expire=False, protect=()):
        access = self._access(conn)
        if expire:
            expired = [e for e in entries if self._expired(e, access, now) and e.entry_id not in protect]
            expired_ids = {e.entry_id for e in expired}
            events.extend((e, "expired", {}) for e in expired)
            entries = [e for e in entries if e.entry_id not in expired_ids]
        overflow = len(entries) - self.max_entries
        if overflow > 0:
            candidates = [e for e in entries if not e.metadata.get("pinned", False) and e.entry_id not in protect]
            candidates.sort(key=lambda e: (access.get(e.entry_id, (e.timestamp, 0))[0], e.timestamp, e.entry_id))
            if len(candidates) < overflow:
                raise MemoryCapacityError("Memory capacity exhausted by pinned/protected entries")
            victims = candidates[:overflow]
            events.extend((e, "lru_evicted", {}) for e in victims)
            removed = {e.entry_id for e in victims}
            entries = [e for e in entries if e.entry_id not in removed]
        return entries

    def _persist(self, conn, before, after, events, now, *, touch=None):
        # Validate/stage index BEFORE any change is published or committed.
        snapshot = self._index_snapshot(after)
        previous = {e.entry_id: e for e in before}
        current_ids = {e.entry_id for e in after}
        for key in previous.keys() - current_ids:
            conn.execute("DELETE FROM entries WHERE session_id=? AND entry_id=?", (self.session_id, key))
            conn.execute("DELETE FROM memory_access WHERE session_id=? AND entry_id=?", (self.session_id, key))
        for item in after:
            if item.entry_id not in previous or item.to_dict() != previous[item.entry_id].to_dict():
                self.lt.write_entry(conn, item)
        if touch is not None:
            conn.execute("""INSERT INTO memory_access(session_id,entry_id,last_access,access_count)
                VALUES (?,?,?,0) ON CONFLICT(session_id,entry_id) DO UPDATE SET last_access=excluded.last_access""",
                (self.session_id, touch, now))
        for item, reason, details in events:
            self._archive(conn, item, reason, details, now)
        self._prune_history(conn)
        return snapshot

    def _tick(self, conn):
        conn.execute("""INSERT INTO memory_maintenance(session_id,write_count) VALUES (?,1)
            ON CONFLICT(session_id) DO UPDATE SET write_count=write_count+1""", (self.session_id,))
        count = conn.execute("SELECT write_count FROM memory_maintenance WHERE session_id=?", (self.session_id,)).fetchone()[0]
        periodic = count % self.evict_interval == 0
        if periodic:
            # Also clean state left by explicit administrative DB deletions.
            conn.execute("""DELETE FROM memory_access WHERE session_id=? AND NOT EXISTS
                (SELECT 1 FROM entries WHERE entries.session_id=memory_access.session_id
                 AND entries.entry_id=memory_access.entry_id)""", (self.session_id,))
        return periodic

    @staticmethod
    def _mark_disputed(entry, other, reason):
        note = entry.metadata.setdefault("_memory", {})
        note["status"] = "disputed"
        note["conflicts"] = list(dict.fromkeys(note.get("conflicts", []) + [other.entry_id]))[-32:]
        note["reason"] = reason

    def _plan_put(self, entries, incoming):
        original = copy.deepcopy(incoming)
        same_id = next((e for e in entries if e.entry_id == incoming.entry_id), None)
        if same_id is not None and (canonical(same_id.claim) != canonical(incoming.claim)
                                    or fact(same_id) != fact(incoming)):
            incoming.entry_id = f"{incoming.entry_id}:{uuid.uuid4().hex}"
        equivalent = []
        for old in entries:
            sim = _cosine_similarity(np.asarray(old.embedding), np.asarray(incoming.embedding))
            kind, _ = relation(old, incoming, sim, self.similarity_threshold_conflict)
            if kind == "equivalent" and (sim > self.similarity_threshold_dup or (fact(old) and fact(incoming))):
                equivalent.append(old)
        events = []
        candidate = incoming
        if equivalent:
            best = max(equivalent, key=lambda e: (authority(e), e.confidence, e.timestamp, e.entry_id))
            if (authority(best), best.confidence, best.timestamp) >= (authority(incoming), incoming.confidence, incoming.timestamp):
                candidate = copy.deepcopy(best)
                events.append((original, "duplicate_ignored", {"retained": best.entry_id}))
            else:
                candidate.entry_id = best.entry_id
                candidate.timestamp = max(candidate.timestamp, best.timestamp)
                events.append((best, "duplicate_replaced", {"retained": best.entry_id}))
            events.extend((e, "duplicate_merged", {"retained": candidate.entry_id})
                          for e in equivalent if e.entry_id != candidate.entry_id)
            if incoming.metadata.get("pinned", False) or any(e.metadata.get("pinned", False) for e in equivalent):
                candidate.metadata["pinned"] = True
        equivalent_ids = {e.entry_id for e in equivalent}
        conflicts = []
        for old in entries:
            if old.entry_id in equivalent_ids or old.entry_id == candidate.entry_id:
                continue
            sim = _cosine_similarity(np.asarray(old.embedding), np.asarray(candidate.embedding))
            kind, reason = relation(candidate, old, sim, self.similarity_threshold_conflict)
            if kind == "conflict":
                decision, why = prefer(candidate, old)
                conflicts.append((old, decision, reason, why))
        blockers = [e for e, decision, _, _ in conflicts if decision < 0]
        if blockers:
            winner = max(blockers, key=lambda e: (authority(e), e.timestamp, e.entry_id))
            return entries, winner.entry_id, [(original, "conflict_rejected", {"retained": winner.entry_id})]
        removed = equivalent_ids | {candidate.entry_id}
        changes = {}
        for old, decision, reason, why in conflicts:
            if decision > 0:
                removed.add(old.entry_id)
                events.append((old, "conflict_superseded", {"retained": candidate.entry_id, "reason": why}))
            else:
                revised = copy.deepcopy(old)
                self._mark_disputed(candidate, revised, reason)
                self._mark_disputed(revised, candidate, reason)
                changes[old.entry_id] = revised
                events.append((original, "conflict_unresolved",
                               {"retained": candidate.entry_id, "other": old.entry_id, "reason": why}))
        if same_id is not None and candidate.entry_id == same_id.entry_id and not equivalent:
            events.append((same_id, "entry_updated", {}))
        updated = [changes.get(e.entry_id, e) for e in entries if e.entry_id not in removed] + [candidate]
        resolved = {old.entry_id for old, decision, _, _ in conflicts if decision > 0}
        for index, item in enumerate(updated):
            note = item.metadata.get("_memory", {})
            if resolved.intersection(note.get("conflicts", [])):
                revised = copy.deepcopy(item)
                note = revised.metadata["_memory"]
                note["conflicts"] = [key for key in note["conflicts"] if key not in resolved]
                if not note["conflicts"]:
                    note["status"] = "active"
                updated[index] = revised
        return updated, candidate.entry_id, events

    @trace_retriever(name="memory.maintain")
    def maintain(self):
        """Enforce current-session capacity/expiry, also when opening an old store."""
        with self._lock:
            with self.lt.transaction() as conn:
                before = self._read_entries(conn)
                events = []
                now = time.time()
                after = self._limit(conn, before, events, now, expire=True)
                snapshot = self._persist(conn, before, after, events, now)
            self._install_index(snapshot)
        return len(before) - len(after)

    def get_history(self, limit=20):
        """Bounded full snapshots of replacements, disputes, expiry and LRU removals."""
        conn = self.lt._connect()
        try:
            return [{"history_id": r["history_id"], "entry_id": r["entry_id"],
                     "reason": r["reason"], "archived_at": r["archived_at"],
                     "entry": MemoryEntry.from_row(json.loads(r["entry_json"])),
                     "details": json.loads(r["details_json"])}
                    for r in conn.execute("""SELECT * FROM memory_history WHERE session_id=?
                        ORDER BY archived_at DESC,rowid DESC LIMIT ?""", (self.session_id, max(0, limit)))]
        finally:
            conn.close()

    @trace_retriever(name="memory.resolve_conflict")
    def resolve_conflict(self, winner_id, loser_ids, *, reason):
        """Review linked disputes, including peers already removed to bounded history."""
        loser_ids = set(loser_ids)
        if not reason or not reason.strip() or not loser_ids or winner_id in loser_ids:
            raise ValueError("Provide a winner, distinct losers and review reason")
        with self._lock:
            with self.lt.transaction() as conn:
                before = self._read_entries(conn)
                current = {e.entry_id: e for e in before}
                if winner_id not in current:
                    raise ValueError("Resolution winner is missing or belongs to another session")
                winner = current[winner_id]
                for key in loser_ids:
                    if key not in winner.metadata.get("_memory", {}).get("conflicts", []) and (
                            key not in current or winner_id not in current[key].metadata.get("_memory", {}).get("conflicts", [])):
                        raise ValueError("Entries do not have a recorded dispute")
                after = copy.deepcopy([e for e in before if e.entry_id not in loser_ids])
                for item in after:
                    note = item.metadata.get("_memory", {})
                    note["conflicts"] = [key for key in note.get("conflicts", []) if key not in loser_ids]
                    if not note["conflicts"]:
                        note["status"] = "active"
                events = [(current.get(key, winner), "conflict_resolved",
                           {"retained": winner_id, "rejected": key, "reason": reason,
                            "peer_already_archived": key not in current}) for key in loser_ids]
                now = time.time()
                after = self._limit(conn, after, events, now, protect=(winner_id,))
                snapshot = self._persist(conn, before, after, events, now, touch=winner_id)
            self._install_index(snapshot)
        return winner_id

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    # 垃圾内容检测模式（正则/关键词，写入前过滤）
    _JUNK_PATTERNS = [
        re.compile(r"i'm ready to help", re.I),
        re.compile(r"您好|你好|hello|hi there", re.I),
        re.compile(r"error\s*:\s*error code:\s*\d+", re.I),
        re.compile(r"^\s*error\s*:\s*", re.I),
        re.compile(r"could you please|请问您想要|您想让我", re.I),
    ]

    def _is_junk(self, entry: MemoryEntry) -> bool:
        """启发式判断 entry 是否为低质量/垃圾内容，应拒绝入库。"""
        claim = entry.claim or ""
        # 1. 过短
        if len(claim.strip()) < 30:
            return True
        # 2. confidence 过低
        if entry.confidence < 0.3:
            return True
        # 3. 命中垃圾模式
        for pat in self._JUNK_PATTERNS:
            if pat.search(claim):
                return True
        return False

    @trace_retriever(name="memory.put", tags=["m4", "memory"])
    def put(self, entry: MemoryEntry) -> str:
        """Resolve/retain relations and enforce capacity atomically; return retained ID."""
        if self._is_junk(entry):
            return entry.entry_id
        entry = self._prepare_entry(entry)
        with self._lock:
            with self.lt.transaction() as conn:
                before = self._read_entries(conn)
                now = time.time()
                events = []
                # Expired facts must not suppress a fresh incoming fact.
                live = self._limit(conn, before, events, now, expire=True)
                after, retained, decisions = self._plan_put(live, entry)
                events.extend(decisions)
                periodic = self._tick(conn)
                after = self._limit(conn, after, events, now, expire=periodic, protect=(retained,))
                snapshot = self._persist(conn, before, after, events, now, touch=retained)
            self._install_index(snapshot)
            record_event("memory_write", {"candidates": len(before), "retained_id": retained,
                         "active_entries": len(after), "decisions": [
                             {"entry_id": e.entry_id, "action": action, "details": details}
                             for e, action, details in events]})
        return retained

    @trace_retriever(name="memory.compact")
    def replace_entries(self, originals: list[MemoryEntry], summary: MemoryEntry) -> bool:
        """Atomically compact this session; abort if any source changed since summarizing."""
        if not originals or self._is_junk(summary):
            return False
        if any(e.session_id != self.session_id for e in originals):
            raise ValueError("Cannot compact entries from another session")
        summary = self._prepare_entry(summary)
        with self._lock:
            with self.lt.transaction() as conn:
                entries = self._read_entries(conn)
                current = {e.entry_id: e for e in entries}
                if summary.entry_id in current:
                    return False
                if any(e.entry_id not in current or current[e.entry_id].to_dict() != e.to_dict()
                       for e in originals):
                    return False
                # Keep unresolved disagreements as individual evidence for review.
                if any(e.metadata.get("_memory", {}).get("status") == "disputed" or e.metadata.get("pinned", False)
                       for e in originals):
                    return False
                removed = {e.entry_id for e in originals}
                now = time.time()
                events = [(e, "compacted", {"summary": summary.entry_id}) for e in originals]
                after = [e for e in entries if e.entry_id not in removed] + [summary]
                periodic = self._tick(conn)
                after = self._limit(conn, after, events, now, expire=periodic, protect=(summary.entry_id,))
                snapshot = self._persist(conn, entries, after, events, now, touch=summary.entry_id)
            self._install_index(snapshot)
        return True

    @trace_retriever(name="memory.query", tags=["m4", "memory"])
    def query_by_similarity(self, query: str, top_k: int = 5, min_sim: float = 0.50):
        vec = np.asarray(self.embedder.encode(query), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if vec.shape != (self.embedder.dim,) or not np.isfinite(vec).all():
            raise ValueError("Invalid query embedding")
        if norm < 1e-9 or top_k <= 0:
            return []
        with self._lock:
            with self.lt.transaction() as conn:
                before = self._read_entries(conn)
                events = []
                now = time.time()
                after = self._limit(conn, before, events, now, expire=True)
                snapshot = self._persist(conn, before, after, events, now)
                ids, matrix, cache = snapshot
                if matrix is None:
                    sims = np.asarray([_cosine_similarity(np.asarray(cache[key].embedding), vec) for key in ids])
                else:
                    sims = matrix.dot(vec / norm)
                indices = np.argsort(sims)[::-1][:max(0, top_k)]
                results = [(copy.deepcopy(cache[ids[int(i)]]), float(sims[i]))
                           for i in indices if float(sims[i]) >= min_sim]
                for item, _ in results:
                    conn.execute("""INSERT INTO memory_access(session_id,entry_id,last_access,access_count)
                        VALUES (?,?,?,1) ON CONFLICT(session_id,entry_id) DO UPDATE SET
                        last_access=excluded.last_access,access_count=access_count+1""",
                        (self.session_id, item.entry_id, now))
            self._install_index(snapshot)
            record_event("memory_query", {"candidates": len(after), "returned": len(results)})
            return results

    def get_context_for_query(self, query: str, max_tokens: int = 4000) -> str:
        """
        为 Agent 组装与 query 相关的记忆上下文文本。

        策略：
        1. 语义相似度搜索 top-10（门槛 min_sim=0.55，避免无关记忆污染）
        2. 加入时间衰减：太旧的记忆降低权重
        3. 按综合得分排序，逐条拼接直到接近 max_tokens
        4. 返回格式化的文本块

        Args:
            query: 当前查询
            max_tokens: token 预算上限

        Returns:
            组装好的上下文文本（空字符串表示无相关记忆）
        """
        import time

        entries_with_sim = self.query_by_similarity(query, top_k=10, min_sim=0.55)
        if not entries_with_sim:
            return ""

        now = time.time()

        def _score(entry: MemoryEntry, sim: float) -> float:
            """综合得分 = 相似度 × confidence × 时间衰减。"""
            days_old = max((now - entry.timestamp) / 86400.0, 0.0)
            recency = np.exp(-days_old / 30.0)  # 时间常数 30 天，半衰期约 20.8 天
            return sim * entry.confidence * recency

        # 按综合得分降序排序
        entries_with_sim.sort(key=lambda x: _score(x[0], x[1]), reverse=True)

        max_chars = int(max_tokens * 3.5)  # token → 字符经验换算
        parts: list[str] = []
        current_chars = 0

        header = "## 相关背景知识\n"
        current_chars += len(header)
        parts.append(header)

        for entry, sim in entries_with_sim:
            note = entry.metadata.get("_memory", {})
            warning = "【存在未裁决的冲突证据，请核验来源与条件】" if note.get("status") == "disputed" else ""
            structured = entry.metadata.get("fact")
            fact_text = ""
            if isinstance(structured, dict) and all(k in structured for k in ("subject", "predicate", "value")):
                fact_text = (f"  事实: {structured['subject']} / {structured['predicate']} = "
                             f"{json.dumps(structured['value'], ensure_ascii=False)} | "
                             f"条件: {json.dumps(structured.get('scope', {}), ensure_ascii=False)} | "
                             f"生效时间: {structured.get('valid_from', '未声明')}\n")
            block = (
                f"- {warning}[{entry.topic}] {entry.claim}\n"
                f"{fact_text}"
                f"  来源: {entry.source} | 置信度: {entry.confidence:.2f} | "
                f"证据类型: {entry.evidence_type} | 相关度: {sim:.2f}\n"
            )
            if current_chars + len(block) > max_chars:
                break
            parts.append(block)
            current_chars += len(block)

        return "".join(parts)

    def __len__(self) -> int:
        return self.lt.count_entries(session_id=self.session_id)
