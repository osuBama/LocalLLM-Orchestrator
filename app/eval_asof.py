"""Reconstruct memory as it was at a past moment, for leak-free evaluation.

Replaying an old session against TODAY's memory flatters the memory system: memory
may already contain facts that were only learned later. Every write to a memory file
first copies the previous version to memory/history/<STEM>.<time>.<n>.md, so:

  * the state at time T is the FIRST snapshot taken after T (it holds the file as it
    was just before that write), or the current file if nothing was written since T;
  * entries created after T are removed in any case (belt and braces, and covers
    hand edits that left no snapshot);
  * if snapshot retention has already pruned writes between T and the oldest
    remaining snapshot, the result is marked approximate instead of pretending.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .markdown_store import MarkdownDocument, MarkdownStore, all_stores

_SNAP_RE = re.compile(r"\.(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.\d+\.md$")


def parse_ts(value: str | None) -> datetime | None:
    """ISO timestamp -> aware datetime (naive values are taken as local time)."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.astimezone()


def _snapshots(store: MarkdownStore) -> list[tuple[datetime, Path]]:
    out = []
    if not store.history_dir.exists():
        return out
    for p in store.history_dir.glob(f"{store.path.stem}.*.md"):
        m = _SNAP_RE.search(p.name)
        if m:
            # Snapshot names use local wall-clock time.
            out.append((datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S").astimezone(), p))
    return sorted(out)


def memory_as_of(memory_dir: Path, when: datetime, *, history_dir: Path | None = None,
                 versions_to_keep: int = 50) -> tuple[dict[str, str], bool]:
    """({file name: Markdown text as of `when`}, approximate?)."""
    stores = all_stores(Path(memory_dir), history_dir=history_dir, create_backups=False,
                        versions_to_keep=versions_to_keep)
    texts: dict[str, str] = {}
    approximate = False
    for store in stores.values():
        snaps = _snapshots(store)
        after = next(((t, p) for t, p in snaps if t > when), None)
        if after is not None:
            text = after[1].read_text(encoding="utf-8")
            # Retention full and T older than everything kept: writes may be missing.
            if after == snaps[0] and len(snaps) >= versions_to_keep:
                approximate = True
        elif store.path.exists():
            text = store.path.read_text(encoding="utf-8")
        else:
            text = ""
        doc = store.parse(text) if text.strip() else MarkdownDocument(store.heading, [], [])
        kept = []
        for e in doc.entries:
            created = parse_ts(e.created_at)
            if created is not None and created > when:
                continue                       # did not exist yet
            if created is None and e.created_at:
                approximate = True
            kept.append(e)
        doc.entries = kept
        texts[store.filename] = store.render(doc)
    return texts, approximate


def empty_memory(memory_dir: Path) -> dict[str, str]:
    stores = all_stores(Path(memory_dir), create_backups=False)
    out = {}
    for s in stores.values():
        doc = s.load()
        doc.entries = []
        out[s.filename] = s.render(doc)
    return out
