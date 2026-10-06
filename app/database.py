"""SQLite metadata: memory mirror, conversations, change audit trail, task queue.

Markdown is the human-readable canonical memory; SQLite is the structured
index and audit log. Raw JSONL remains the authority for recovery.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .schemas import MemoryEntry
from .util import now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL DEFAULT 'default',
    entry_key TEXT NOT NULL,
    category TEXT NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    source_conversation_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    UNIQUE(project_id, entry_key)
);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL DEFAULT 'default',
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT,
    task_id INTEGER,
    created_at TEXT NOT NULL,
    category TEXT,
    operation TEXT,
    entry_key TEXT,
    title TEXT,
    proposed_content TEXT,
    confidence REAL,
    reason TEXT,
    status TEXT NOT NULL CHECK (status IN ('pending','approved','rejected','failed')),
    status_detail TEXT
);

CREATE TABLE IF NOT EXISTS memory_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT,
    project_id TEXT NOT NULL DEFAULT 'default',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending','processing','done','failed','skipped')),
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL DEFAULT 0,
    last_error TEXT,
    duration_seconds REAL
);

CREATE TABLE IF NOT EXISTS session_summaries (
    conversation_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL DEFAULT 'default',
    summary TEXT NOT NULL,
    covered_turns INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_digests (
    hash TEXT PRIMARY KEY,
    tool TEXT,
    digest TEXT,
    original_tokens INTEGER NOT NULL,
    digest_tokens INTEGER NOT NULL,
    usable INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON memory_tasks(status, next_attempt_at);
CREATE INDEX IF NOT EXISTS idx_changes_conv ON memory_changes(conversation_id);
"""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self.connect() as c:
            c.executescript(SCHEMA)
            cols = {r["name"] for r in c.execute("PRAGMA table_info(memory_tasks)")}
            if "priority" not in cols:  # migration from 0.2.0
                c.execute("ALTER TABLE memory_tasks ADD COLUMN priority INTEGER NOT NULL DEFAULT 0")
                c.execute("ALTER TABLE memory_tasks ADD COLUMN kind TEXT NOT NULL DEFAULT 'extract'")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=30)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA foreign_keys=ON")
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # ---------------------------------------------------------------- memories
    def upsert_memory(self, e: MemoryEntry, project_id: str = "default") -> None:
        with self.connect() as c:
            c.execute(
                """INSERT INTO memories (project_id, entry_key, category, title, content,
                       source_conversation_id, created_at, updated_at, active)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(project_id, entry_key) DO UPDATE SET
                       category=excluded.category, title=excluded.title, content=excluded.content,
                       source_conversation_id=excluded.source_conversation_id,
                       updated_at=excluded.updated_at, active=excluded.active""",
                (project_id, e.entry_id, e.category.value, e.title, e.content, e.source or None,
                 e.created_at or now_iso(), e.updated_at or now_iso(), 1 if e.active else 0))

    def replace_memories(self, entries: list[MemoryEntry], project_id: str = "default") -> None:
        with self.connect() as c:
            c.execute("DELETE FROM memories WHERE project_id=?", (project_id,))
        for e in entries:
            self.upsert_memory(e, project_id)

    def list_memories(self, project_id: str = "default", active_only: bool = False) -> list[dict]:
        q = "SELECT * FROM memories WHERE project_id=?" + (" AND active=1" if active_only else "")
        with self.connect() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY category, entry_key", (project_id,))]

    # ----------------------------------------------------------- conversations
    def touch_conversation(self, conversation_id: str, project_id: str = "default") -> None:
        ts = now_iso()
        with self.connect() as c:
            c.execute(
                """INSERT INTO conversations (id, project_id, started_at, updated_at) VALUES (?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at""",
                (conversation_id, project_id, ts, ts))

    # ----------------------------------------------------------------- changes
    def record_change(self, *, conversation_id: str | None, task_id: int | None, category: str | None,
                      operation: str | None, entry_key: str | None, title: str | None,
                      content: str | None, confidence: float | None, reason: str | None,
                      status: str, detail: str | None = None) -> int:
        with self.connect() as c:
            cur = c.execute(
                """INSERT INTO memory_changes (conversation_id, task_id, created_at, category, operation,
                       entry_key, title, proposed_content, confidence, reason, status, status_detail)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (conversation_id, task_id, now_iso(), category, operation, entry_key, title, content,
                 confidence, reason, status, detail))
            return int(cur.lastrowid)

    def set_change_status(self, change_id: int, status: str, detail: str | None = None,
                          entry_key: str | None = None) -> None:
        with self.connect() as c:
            c.execute("UPDATE memory_changes SET status=?, status_detail=?, entry_key=COALESCE(?, entry_key) "
                      "WHERE id=?", (status, detail, entry_key, change_id))

    def list_changes(self, limit: int = 50) -> list[dict]:
        with self.connect() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM memory_changes ORDER BY id DESC LIMIT ?", (limit,))]

    # ------------------------------------------------------------------- tasks
    def enqueue_task(self, payload: dict[str, Any], status: str = "pending",
                     detail: str | None = None, kind: str = "extract", priority: int = 0) -> int:
        ts = now_iso()
        with self.connect() as c:
            cur = c.execute(
                """INSERT INTO memory_tasks (conversation_id, project_id, created_at, updated_at,
                       payload, status, last_error, kind, priority) VALUES (?,?,?,?,?,?,?,?,?)""",
                (payload.get("conversation_id"), payload.get("project_id", "default"), ts, ts,
                 json.dumps(payload, ensure_ascii=False), status, detail, kind, priority))
            return int(cur.lastrowid)

    def claim_next_task(self, now: float) -> dict | None:
        with self.connect() as c:
            row = c.execute(
                "SELECT * FROM memory_tasks WHERE status='pending' AND next_attempt_at<=? "
                "ORDER BY priority DESC, id LIMIT 1", (now,)).fetchone()
            if row is None:
                return None
            c.execute("UPDATE memory_tasks SET status='processing', attempts=attempts+1, updated_at=? "
                      "WHERE id=?", (now_iso(), row["id"]))
            d = dict(row)
            d["attempts"] += 1
            d["payload"] = json.loads(d["payload"])
            return d

    def finish_task(self, task_id: int, status: str, error: str | None = None,
                    duration: float | None = None, next_attempt_at: float = 0) -> None:
        with self.connect() as c:
            c.execute("UPDATE memory_tasks SET status=?, last_error=?, duration_seconds=?, "
                      "next_attempt_at=?, updated_at=? WHERE id=?",
                      (status, error, duration, next_attempt_at, now_iso(), task_id))

    def reset_stale_tasks(self) -> int:
        """Crash recovery: anything left 'processing' goes back to the queue."""
        with self.connect() as c:
            return c.execute("UPDATE memory_tasks SET status='pending' WHERE status='processing'").rowcount

    def task_counts(self) -> dict[str, int]:
        with self.connect() as c:
            return {r["status"]: r["n"] for r in c.execute(
                "SELECT status, COUNT(*) AS n FROM memory_tasks GROUP BY status")}

    def list_tasks(self, limit: int = 20, status: str | None = None) -> list[dict]:
        q = "SELECT id, kind, conversation_id, created_at, updated_at, status, attempts, last_error, " \
            "duration_seconds FROM memory_tasks"
        args: tuple = ()
        if status:
            q += " WHERE status=?"
            args = (status,)
        with self.connect() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY id DESC LIMIT ?", args + (limit,))]

    # ------------------------------------------------------ session summaries
    def get_summary(self, conversation_id: str) -> dict | None:
        with self.connect() as c:
            r = c.execute("SELECT * FROM session_summaries WHERE conversation_id=?",
                          (conversation_id,)).fetchone()
            return dict(r) if r else None

    def set_summary(self, conversation_id: str, summary: str, covered_turns: int,
                    project_id: str = "default") -> bool:
        """Only moves forward: a late retry for an older turn never overwrites a newer summary."""
        with self.connect() as c:
            cur = c.execute(
                """INSERT INTO session_summaries (conversation_id, project_id, summary, covered_turns, updated_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(conversation_id) DO UPDATE SET summary=excluded.summary,
                       covered_turns=excluded.covered_turns, updated_at=excluded.updated_at
                   WHERE excluded.covered_turns > session_summaries.covered_turns""",
                (conversation_id, project_id, summary, covered_turns, now_iso()))
            return cur.rowcount > 0

    def list_summaries(self, limit: int = 20) -> list[dict]:
        with self.connect() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM session_summaries ORDER BY updated_at DESC LIMIT ?", (limit,))]

    # ------------------------------------------------------------ tool digests
    def get_digests(self, hashes: list[str]) -> dict[str, dict]:
        if not hashes:
            return {}
        out: dict[str, dict] = {}
        with self.connect() as c:
            for i in range(0, len(hashes), 500):
                chunk = hashes[i:i + 500]
                q = f"SELECT * FROM tool_digests WHERE hash IN ({','.join('?' * len(chunk))})"
                out.update({r["hash"]: dict(r) for r in c.execute(q, chunk)})
        return out

    def set_digest(self, hash_: str, tool: str | None, digest: str, original_tokens: int,
                   digest_tokens: int, usable: bool) -> None:
        with self.connect() as c:
            c.execute("""INSERT OR REPLACE INTO tool_digests
                         (hash, tool, digest, original_tokens, digest_tokens, usable, created_at)
                         VALUES (?,?,?,?,?,?,?)""",
                      (hash_, tool, digest, original_tokens, digest_tokens, 1 if usable else 0, now_iso()))

    def digest_stats(self) -> dict:
        with self.connect() as c:
            r = c.execute("SELECT COUNT(*) n, SUM(usable) usable, SUM(original_tokens) orig, "
                          "SUM(CASE WHEN usable=1 THEN digest_tokens END) dig FROM tool_digests").fetchone()
            return {"digests": r["n"] or 0, "usable": r["usable"] or 0,
                    "original_tokens": r["orig"] or 0, "digest_tokens": r["dig"] or 0}

    def backup_to(self, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            src = sqlite3.connect(self.path)
            dst = sqlite3.connect(dest)
            try:
                src.backup(dst)
            finally:
                dst.close()
                src.close()
