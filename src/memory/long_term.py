"""
Long-term Memory 模块：SQLite 持久化 + 结构化查询

设计决策：
1. SQLite 零运维、文件级持久化、ACID 事务
2. embedding 以 JSON 文本存储，加载时反序列化
3. 实例内用 RLock，跨实例复合写入通过 SQLite BEGIN IMMEDIATE 事务协调
4. metadata 使用 JSON 字段，保持 schema 灵活性
"""

from __future__ import annotations

from contextlib import contextmanager
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
    """Session-scoped SQLite storage. Schema migration and writes are transactional."""

    def __init__(self, db_path: str = "memory.db") -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        from ..utils.execution import check_deadline, remaining
        check_deadline()
        budget = remaining()
        conn = sqlite3.connect(self.db_path, timeout=30 if budget is None else min(30, budget))
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def transaction(self):
        """Serialize read/check/write across store instances and processes."""
        from ..utils.execution import check_deadline
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                check_deadline()
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            finally:
                conn.close()

    @staticmethod
    def _create_table(conn, name="entries"):
        # name is internal, never supplied by a caller.
        conn.execute(f"""CREATE TABLE {name} (
            entry_id TEXT NOT NULL, claim TEXT NOT NULL, source TEXT NOT NULL,
            confidence REAL NOT NULL, agent_id TEXT NOT NULL, timestamp REAL NOT NULL,
            evidence_type TEXT NOT NULL, embedding_json TEXT NOT NULL,
            topic TEXT NOT NULL, metadata_json TEXT NOT NULL,
            session_id TEXT NOT NULL DEFAULT '', PRIMARY KEY (session_id, entry_id)
        )""")

    def _ensure_tables(self):
        with self.transaction() as conn:
            info = conn.execute("PRAGMA table_info(entries)").fetchall()
            if not info:
                self._create_table(conn)
            else:
                columns = {row["name"] for row in info}
                if "session_id" not in columns:
                    conn.execute("ALTER TABLE entries ADD COLUMN session_id TEXT NOT NULL DEFAULT ''")
                    info = conn.execute("PRAGMA table_info(entries)").fetchall()
                primary = [r["name"] for r in sorted(info, key=lambda r: r["pk"]) if r["pk"]]
                if primary != ["session_id", "entry_id"]:
                    self._create_table(conn, "entries_session_migration")
                    names = ("entry_id,claim,source,confidence,agent_id,timestamp,"
                             "evidence_type,embedding_json,topic,metadata_json,session_id")
                    conn.execute(f"INSERT INTO entries_session_migration ({names}) SELECT {names} FROM entries")
                    conn.execute("DROP TABLE entries")
                    conn.execute("ALTER TABLE entries_session_migration RENAME TO entries")
            for col in ("topic", "agent_id", "timestamp", "session_id"):
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_entries_{col} ON entries({col})")
            # Side tables preserve the public MemoryEntry and legacy entries schema.
            conn.execute("""CREATE TABLE IF NOT EXISTS memory_access (
                session_id TEXT NOT NULL, entry_id TEXT NOT NULL,
                last_access REAL NOT NULL, access_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(session_id, entry_id))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS memory_history (
                history_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                entry_id TEXT NOT NULL, archived_at REAL NOT NULL,
                reason TEXT NOT NULL, entry_json TEXT NOT NULL, details_json TEXT NOT NULL)""")
            conn.execute("""CREATE INDEX IF NOT EXISTS idx_memory_history_session
                ON memory_history(session_id, archived_at)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS memory_maintenance (
                session_id TEXT PRIMARY KEY, write_count INTEGER NOT NULL DEFAULT 0)""")

    @staticmethod
    def write_entry(conn, entry: MemoryEntry):
        conn.execute("""INSERT INTO entries
            (entry_id,claim,source,confidence,agent_id,timestamp,evidence_type,
             embedding_json,topic,metadata_json,session_id)
            VALUES (:entry_id,:claim,:source,:confidence,:agent_id,:timestamp,
                    :evidence_type,:embedding_json,:topic,:metadata_json,:session_id)
            ON CONFLICT(session_id, entry_id) DO UPDATE SET
                claim=excluded.claim,source=excluded.source,confidence=excluded.confidence,
                agent_id=excluded.agent_id,timestamp=excluded.timestamp,
                evidence_type=excluded.evidence_type,embedding_json=excluded.embedding_json,
                topic=excluded.topic,metadata_json=excluded.metadata_json
        """, entry.to_dict())

    def insert_entry(self, entry: MemoryEntry) -> None:
        with self.transaction() as conn:
            self.write_entry(conn, entry)

    def get_entry(self, entry_id: str, session_id: str = "") -> Optional[MemoryEntry]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT * FROM entries WHERE session_id=? AND entry_id=?",
                                   (session_id, entry_id)).fetchone()
                return MemoryEntry.from_row(row) if row else None
            finally:
                conn.close()

    def get_all_entries(self, session_id: Optional[str] = None) -> list[MemoryEntry]:
        """None is an explicit administrative all-session read; stores pass a scope."""
        with self._lock:
            conn = self._connect()
            try:
                rows = (conn.execute("SELECT * FROM entries") if session_id is None else
                        conn.execute("SELECT * FROM entries WHERE session_id=?", (session_id,)))
                return [MemoryEntry.from_row(r) for r in rows]
            finally:
                conn.close()

    def count_entries(self, session_id: Optional[str] = None) -> int:
        with self._lock:
            conn = self._connect()
            try:
                row = (conn.execute("SELECT COUNT(*) FROM entries") if session_id is None else
                       conn.execute("SELECT COUNT(*) FROM entries WHERE session_id=?", (session_id,)))
                return row.fetchone()[0]
            finally:
                conn.close()
