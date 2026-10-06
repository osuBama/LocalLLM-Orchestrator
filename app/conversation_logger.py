"""Append-only raw conversation history (JSONL, one file per day).

This is the recovery source of truth: memory can always be rebuilt from it.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from .util import now_iso


class ConversationLogger:
    def __init__(self, directory: Path, enabled: bool = True):
        self.directory = Path(directory)
        self.enabled = enabled
        self._lock = threading.Lock()

    def _file(self) -> Path:
        return self.directory / f"{datetime.now():%Y-%m-%d}.jsonl"

    def _append(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        record.setdefault("timestamp", now_iso())
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            with open(self._file(), "a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())

    def message(self, conversation_id: str, role: str, content: str, *,
                project_id: str = "default", **extra: Any) -> None:
        self._append({"conversation_id": conversation_id, "project_id": project_id,
                      "type": "message", "role": role, "content": content, **extra})

    def tool_call(self, conversation_id: str, tool: str, arguments: Any, *,
                  project_id: str = "default") -> None:
        self._append({"conversation_id": conversation_id, "project_id": project_id,
                      "type": "tool_call", "tool": tool, "arguments": arguments})

    def tool_result(self, conversation_id: str, tool: str, result: Any, *,
                    project_id: str = "default") -> None:
        self._append({"conversation_id": conversation_id, "project_id": project_id,
                      "type": "tool_result", "tool": tool, "result": result})

    def system_event(self, conversation_id: str | None, event: str, *,
                     project_id: str = "default", **details: Any) -> None:
        self._append({"conversation_id": conversation_id, "project_id": project_id,
                      "type": "system_event", "event": event, **details})

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """All records in chronological order (file name, then line order)."""
        if not self.directory.exists():
            return
        for path in sorted(self.directory.glob("*.jsonl")):
            with open(path, encoding="utf-8") as f:
                for n, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        # A torn final line must not block recovery of everything else.
                        yield {"type": "system_event", "event": "corrupt_log_line",
                               "file": path.name, "line": n}

    def iter_interactions(self) -> Iterator[dict[str, Any]]:
        """Reassemble user -> (tools) -> assistant turns for memory replay."""
        pending: dict[str, dict[str, Any]] = {}
        for r in self.iter_records():
            cid = r.get("conversation_id")
            if not cid:
                continue
            t = r.get("type", "message")
            if t == "message" and r.get("role") == "user":
                pending[cid] = {"conversation_id": cid, "project_id": r.get("project_id", "default"),
                                "timestamp": r.get("timestamp", ""), "user_message": r.get("content", ""),
                                "tool_events": []}
            elif t in ("tool_call", "tool_result") and cid in pending:
                pending[cid]["tool_events"].append(
                    {k: r.get(k) for k in ("type", "tool", "arguments", "result") if k in r})
            elif t == "message" and r.get("role") == "assistant" and cid in pending:
                turn = pending.pop(cid)
                turn["assistant_response"] = r.get("content", "")
                yield turn
