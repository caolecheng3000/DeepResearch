"""
Shared Memory Store 模块：跨 Agent 统一读写接口

设计决策：
1. 持久层用 LongTermMemory（SQLite），内存层维护 numpy 向量索引，读写平衡
2. 写前自动去重（cosine > 0.92），低质量内容过滤
3. 淘汰策略：按综合评分从低到高删除
4. get_context_for_query 将相关记忆组装为文本，控制总 token 在预算内
5. 所有内存索引操作受 threading.Lock 保护，保证线程安全
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Optional

import numpy as np

from src.memory.embedder import Embedder
from src.memory.long_term import LongTermMemory, MemoryEntry
from src.utils.tracing import trace_retriever

logger = logging.getLogger(__name__)

# 去重相似度阈值
_DEDUP_THRESHOLD = 0.92

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

    提供 put/query/conflict_resolution/evict 等高级语义接口，
    底层通过 LongTermMemory 做 SQLite 持久化，内存中维护 numpy 向量索引加速相似度查询。
    """

    def __init__(
        self,
        db_path: str = "memory.db",
        embedder: Optional[Embedder] = None,
        session_id: str = "",
    ) -> None:
        """
        初始化共享记忆存储。

        Args:
            db_path: SQLite 数据库路径
            embedder: 向量化器，None 时自动创建
            session_id: 会话 ID，空字符串表示加载所有历史数据（不隔离）
        """
        self.lt = LongTermMemory(db_path=db_path)
        self.embedder = embedder or Embedder()
        self._lock = threading.RLock()
        self.session_id = session_id

        # 内存向量索引
        self._entry_ids: list[str] = []
        self._embeddings: np.ndarray = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._entries_cache: dict[str, MemoryEntry] = {}

        # 加载已有数据到内存索引（按 session 过滤）
        self._rebuild_index()

    # ------------------------------------------------------------------
    # 内部索引管理
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> None:
        """从 SQLite 重建内存向量索引。若指定了 session_id，只加载该会话数据。"""
        entries = self.lt.get_all_entries(session_id=self.session_id or None)
        with self._lock:
            self._entry_ids = [e.entry_id for e in entries]
            self._entries_cache = {e.entry_id: e for e in entries}
            if entries:
                mat = np.array([e.embedding for e in entries], dtype=np.float32)
                # 归一化（防御性）
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms < 1e-9] = 1.0
                self._embeddings = mat / norms
            else:
                self._embeddings = np.zeros((0, self.embedder.dim), dtype=np.float32)
        scope = f"session={self.session_id}" if self.session_id else "all sessions"
        logger.info(f"Memory index rebuilt: {len(entries)} entries loaded ({scope}).")

    def _add_to_index(self, entry: MemoryEntry) -> None:
        """将单条 entry 追加到内存索引。"""
        vec = np.array(entry.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec = vec / norm
        with self._lock:
            self._entry_ids.append(entry.entry_id)
            self._entries_cache[entry.entry_id] = entry
            if self._embeddings.shape[0] == 0:
                self._embeddings = vec.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, vec.reshape(1, -1)])

    def _remove_from_index(self, entry_id: str) -> None:
        """从内存索引删除 entry。"""
        with self._lock:
            if entry_id not in self._entry_ids:
                return
            idx = self._entry_ids.index(entry_id)
            self._entry_ids.pop(idx)
            self._entries_cache.pop(entry_id, None)
            if self._embeddings.shape[0] > 0:
                self._embeddings = np.delete(self._embeddings, idx, axis=0)

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
        """
        写入记忆条目。

        流程：
        1. 质量过滤：低质量/垃圾内容不入库
        2. 若 embedding 为空，自动生成
        3. 去重检测：cosine > 0.92 → 合并（保留 confidence 更高的）
        4. 持久化到 SQLite 并更新内存索引

        Args:
            entry: 记忆条目

        Returns:
            entry_id（如果是合并，返回被合并的已有 entry_id）
        """
        # 1. 质量过滤
        if self._is_junk(entry):
            logger.info(f"[M4] Junk entry rejected (conf={entry.confidence:.2f}, len={len(entry.claim)}): {entry.claim[:60]}...")
            return entry.entry_id

        # 2. 确保有 embedding
        if not entry.embedding:
            entry.embedding = self.embedder.encode(entry.claim)

        # 去重检测
        duplicate_id = self._find_duplicate(entry)
        if duplicate_id:
            existing = self.lt.get_entry(duplicate_id)
            if existing and entry.confidence > existing.confidence:
                # 用新 entry 更新，但保留旧 ID
                entry.entry_id = duplicate_id
                entry.timestamp = max(entry.timestamp, existing.timestamp)
                self.lt.insert_entry(entry)
                self._remove_from_index(duplicate_id)
                self._add_to_index(entry)
                logger.info(f"Merged entry {duplicate_id} with higher confidence.")
            else:
                logger.info(f"Duplicate detected, kept existing {duplicate_id}.")
            return duplicate_id

        # 写入 session_id 并持久化
        entry.session_id = self.session_id
        self.lt.insert_entry(entry)
        self._add_to_index(entry)

        return entry.entry_id

    def _find_duplicate(self, entry: MemoryEntry) -> Optional[str]:
        """
        在内存索引中搜索与 entry  cosine similarity > 0.92 的已有条目。

        Returns:
            最相似的已有 entry_id，无则 None
        """
        if self._embeddings.shape[0] == 0:
            return None
        vec = np.array(entry.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            return None
        vec = vec / norm
        with self._lock:
            sims = self._embeddings.dot(vec)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim > _DEDUP_THRESHOLD:
            return self._entry_ids[best_idx]
        return None

    @trace_retriever(name="memory.query", tags=["m4", "memory"])
    def query_by_similarity(
        self, query: str, top_k: int = 5, min_sim: float = 0.50
    ) -> list[tuple[MemoryEntry, float]]:
        """
        按 query 语义相似度搜索记忆，带相关性门槛过滤。

        Args:
            query: 查询文本
            top_k: 返回条数上限
            min_sim: 最小相似度门槛，低于此值的记忆视为不相关

        Returns:
            (MemoryEntry, similarity) 列表（按相似度降序）
        """
        if self._embeddings.shape[0] == 0:
            return []
        q_vec = np.array(self.embedder.encode(query), dtype=np.float32)
        norm = float(np.linalg.norm(q_vec))
        if norm < 1e-9:
            return []
        q_vec = q_vec / norm
        with self._lock:
            sims = self._embeddings.dot(q_vec)
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            sim = float(sims[int(idx)])
            if sim < min_sim:
                continue
            entry_id = self._entry_ids[int(idx)]
            entry = self._entries_cache.get(entry_id)
            if entry:
                results.append((entry, sim))
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
            recency = np.exp(-days_old / 30.0)  # 30 天半衰期
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
            block = (
                f"- [{entry.topic}] {entry.claim}\n"
                f"  来源: {entry.source} | 置信度: {entry.confidence:.2f} | "
                f"证据类型: {entry.evidence_type} | 相关度: {sim:.2f}\n"
            )
            if current_chars + len(block) > max_chars:
                break
            parts.append(block)
            current_chars += len(block)

        return "".join(parts)

    def __len__(self) -> int:
        return self.lt.count_entries(session_id=self.session_id or None)
