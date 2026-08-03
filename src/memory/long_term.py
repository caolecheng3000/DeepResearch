"""
Long-term Memory 模块：SQLite 持久化 + 结构化查询

设计决策：
1. SQLite 零运维、文件级持久化、ACID 事务
2. embedding 以 JSON 文本存储，加载时反序列化
3. 所有写操作通过 threading.Lock 保证线程安全
4. metadata 使用 JSON 字段，保持 schema 灵活性
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """记忆条目数据模型。"""

    entry_id: str
    claim: str              # 核心信息（一句话）
    source: str             # 来源 URL/标题
    confidence: float       # 0-1
    agent_id: str           # 写入的 agent
    timestamp: float
    evidence_type: str      # "primary" | "secondary" | "inference"
    embedding: list[float]  # 语义向量
    topic: str
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""    # 所属会话 ID，空字符串表示全局/未分类

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict，embedding 和 metadata 转为 JSON 字符串。"""
        return {
            "entry_id": self.entry_id,
            "claim": self.claim,
            "source": self.source,
            "confidence": self.confidence,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "evidence_type": self.evidence_type,
            "embedding_json": json.dumps(self.embedding, ensure_ascii=False),
            "topic": self.topic,
            "metadata_json": json.dumps(self.metadata, ensure_ascii=False),
            "session_id": self.session_id,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> MemoryEntry:
        """从 SQLite Row 反序列化。"""
        return cls(
            entry_id=row["entry_id"],
            claim=row["claim"],
            source=row["source"],
            confidence=row["confidence"],
            agent_id=row["agent_id"],
            timestamp=row["timestamp"],
            evidence_type=row["evidence_type"],
            embedding=json.loads(row["embedding_json"]),
            topic=row["topic"],
            metadata=json.loads(row["metadata_json"]),
            session_id=row["session_id"] if "session_id" in row.keys() else "",
        )


class LongTermMemory:
    """
    长期记忆 SQLite 封装。

    提供 entries 表的 CRUD 及基础查询。
    上层 SharedMemoryStore 负责向量相似度计算和高级语义操作。
    """

    def __init__(self, db_path: str = "memory.db") -> None:
        """
        初始化并建表。

        Args:
            db_path: SQLite 数据库文件路径
        """
        self.db_path = db_path
        self._lock = threading.RLock()
        # 自动创建父目录（SQLite 不会在目录不存在时自动创建）
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()
        self._migrate_add_session_id()

    def _connect(self) -> sqlite3.Connection:
        """创建新连接（SQLite 连接不是线程安全的，每次操作新建连接）。"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        """初始化表结构（如果不存在）。"""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS entries (
                        entry_id TEXT PRIMARY KEY,
                        claim TEXT NOT NULL,
                        source TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        agent_id TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        evidence_type TEXT NOT NULL,
                        embedding_json TEXT NOT NULL,
                        topic TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        session_id TEXT NOT NULL DEFAULT ''
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_entries_topic ON entries(topic)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_entries_agent ON entries(agent_id)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_entries_timestamp ON entries(timestamp)
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def _migrate_add_session_id(self) -> None:
        """为已存在的旧表添加 session_id 列（向后兼容）。"""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("PRAGMA table_info(entries)")
                columns = [row["name"] for row in cur.fetchall()]
                if "session_id" not in columns:
                    conn.execute("ALTER TABLE entries ADD COLUMN session_id TEXT NOT NULL DEFAULT ''")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_entries_session ON entries(session_id)")
                    conn.commit()
                    logger.info("[LongTermMemory] 已迁移：为 entries 表添加 session_id 列")
            except Exception as e:
                logger.warning(f"[LongTermMemory] session_id 迁移失败（可能已存在）: {e}")
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Entries 操作
    # ------------------------------------------------------------------

    def insert_entry(self, entry: MemoryEntry) -> None:
        """插入或替换 entry（REPLACE 语义）。"""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO entries
                    (entry_id, claim, source, confidence, agent_id, timestamp,
                     evidence_type, embedding_json, topic, metadata_json, session_id)
                    VALUES
                    (:entry_id, :claim, :source, :confidence, :agent_id, :timestamp,
                     :evidence_type, :embedding_json, :topic, :metadata_json, :session_id)
                    """,
                    entry.to_dict(),
                )
                conn.commit()
            finally:
                conn.close()

    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """按 ID 查询单条 entry。"""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT * FROM entries WHERE entry_id = ?", (entry_id,)
                )
                row = cur.fetchone()
                return MemoryEntry.from_row(row) if row else None
            finally:
                conn.close()

    def get_all_entries(self, session_id: Optional[str] = None) -> list[MemoryEntry]:
        """加载 entries。若指定 session_id，只加载该会话的数据。"""
        with self._lock:
            conn = self._connect()
            try:
                if session_id is not None:
                    cur = conn.execute("SELECT * FROM entries WHERE session_id = ?", (session_id,))
                else:
                    cur = conn.execute("SELECT * FROM entries")
                return [MemoryEntry.from_row(r) for r in cur.fetchall()]
            finally:
                conn.close()

    def count_entries(self, session_id: Optional[str] = None) -> int:
        """返回 entries 数量。可指定 session_id 过滤。"""
        with self._lock:
            conn = self._connect()
            try:
                if session_id is not None:
                    cur = conn.execute("SELECT COUNT(*) FROM entries WHERE session_id = ?", (session_id,))
                else:
                    cur = conn.execute("SELECT COUNT(*) FROM entries")
                return cur.fetchone()[0]
            finally:
                conn.close()

