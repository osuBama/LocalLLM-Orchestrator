"""Small shared helpers."""
from __future__ import annotations

import math
from datetime import datetime


def now_iso() -> str:
    """Local time with UTC offset, e.g. 2026-09-30T13:00:00+01:00."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def stamp() -> str:
    """Filesystem-safe timestamp, e.g. 2026-09-30_13-00-00."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def estimate_tokens(text: str) -> int:
    """Approximate token count (Phase 1: ~3.5 chars/token, rounded up).

    Deliberately conservative for English; code and non-Latin text tokenize
    worse, so budgets err on the small side.
    """
    if not text:
        return 0
    return math.ceil(len(text) / 3.5)


def truncate_tokens(text: str, max_tokens: int, marker: str = " …[truncated]") -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    max_chars = max(0, int(max_tokens * 3.5) - len(marker))
    return text[:max_chars].rstrip() + marker


def head_tail(text: str, max_tokens: int) -> str:
    """Keep the beginning and the end of a long text (errors usually live at the end)."""
    if estimate_tokens(text) <= max_tokens:
        return text
    budget = int(max_tokens * 3.5)
    head = budget * 2 // 3
    tail = budget - head
    return text[:head].rstrip() + "\n…[middle omitted]…\n" + text[-tail:].lstrip()
