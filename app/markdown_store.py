"""Human-readable Markdown memory, owned by application code.

File layout (one file per category):

    # LESSONS

    <!-- preamble: preserved verbatim -->

    ## Active

    ### L-001 — HTTP 404 debugging
    <!-- meta: status=active created=... updated=... source=abc123 -->
    A successful HTTP connection does not prove the endpoint exists.

    ## Inactive

    ### L-000 — ...

The LLM never writes these files. It proposes changes; this module applies
them deterministically: read -> backup -> modify -> write temp -> re-parse
and verify -> atomic replace.
"""
from __future__ import annotations

import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from .schemas import CATEGORY_INFO, Category, MemoryEntry
from .util import now_iso

DEFAULT_PREAMBLE = (
    "<!-- Managed by the local AI orchestrator. You may edit entry text by hand; "
    "keep the ### headings and meta lines intact. -->"
)

_ENTRY_RE = re.compile(r"^###\s+((?:S|O|C|D|L|DS|E)-\d{3,6})\s+(?:—|-)\s+(.+?)\s*$")
_META_RE = re.compile(r"^<!--\s*meta:(.*?)-->\s*$")
_INACTIVE_LABELS = {"inactive", "superseded", "completed / inactive", "completed"}

_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _locks_guard:
        return _locks.setdefault(key, threading.RLock())


class MarkdownFormatError(Exception):
    """The file contains content this store does not manage; refuse to rewrite it."""


@dataclass
class MarkdownDocument:
    heading: str
    preamble: list[str] = field(default_factory=list)
    entries: list[MemoryEntry] = field(default_factory=list)


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.replace("\r", " ").replace("\n", " ")).strip()


def normalize_content(content: str) -> str:
    """Make content safe to embed: no fake headings, no fake meta comments."""
    lines = []
    for line in content.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n"):
        line = line.rstrip()
        if re.match(r"^\s{0,3}#", line):
            line = "\\" + line.lstrip()
        lines.append(line.replace("<!--", "&lt;!--"))
    # Collapse runs of blank lines.
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _meta_value(v: str) -> str:
    return re.sub(r"\s+", "_", v.strip()) if v else ""


class MarkdownStore:
    def __init__(self, memory_dir: Path, category: Category, *,
                 history_dir: Path | None = None, create_backups: bool = True,
                 versions_to_keep: int = 50):
        self.category = category
        self.filename, self.prefix, self.heading, self.inactive_label = CATEGORY_INFO[category]
        self.path = Path(memory_dir) / self.filename
        self.history_dir = Path(history_dir) if history_dir else Path(memory_dir) / "history"
        self.create_backups = create_backups
        self.versions_to_keep = versions_to_keep
        self._lock = _lock_for(self.path)

    # ------------------------------------------------------------------ read
    def ensure_exists(self) -> None:
        with self._lock:
            if not self.path.exists():
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._atomic_write(MarkdownDocument(self.heading, [DEFAULT_PREAMBLE], []), backup=False)

    def load(self) -> MarkdownDocument:
        if not self.path.exists():
            return MarkdownDocument(self.heading, [DEFAULT_PREAMBLE], [])
        return self.parse(self.path.read_text(encoding="utf-8"))

    def parse(self, text: str) -> MarkdownDocument:
        lines = text.replace("\r\n", "\n").split("\n")
        i = 0
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            return MarkdownDocument(self.heading, [DEFAULT_PREAMBLE], [])
        if not lines[i].startswith("# "):
            raise MarkdownFormatError(f"{self.filename}: first line must be '# {self.heading}'")
        heading = lines[i][2:].strip() or self.heading
        i += 1

        preamble: list[str] = []
        while i < len(lines) and not lines[i].startswith("## "):
            preamble.append(lines[i])
            i += 1
        while preamble and not preamble[0].strip():
            preamble.pop(0)
        while preamble and not preamble[-1].strip():
            preamble.pop()
        for p in preamble:
            if p.startswith("### "):
                raise MarkdownFormatError(f"{self.filename}: entry heading outside a section: {p!r}")

        entries: list[MemoryEntry] = []
        seen: set[str] = set()
        section_active: bool | None = None
        current: MemoryEntry | None = None
        body: list[str] = []

        def flush():
            nonlocal current, body
            if current is not None:
                current.content = "\n".join(body).strip()
                entries.append(current)
            current, body = None, []

        while i < len(lines):
            line = lines[i]
            if line.startswith("## "):
                flush()
                label = line[3:].strip().lower()
                if label == "active":
                    section_active = True
                elif label in _INACTIVE_LABELS or label == self.inactive_label.lower():
                    section_active = False
                else:
                    raise MarkdownFormatError(
                        f"{self.filename}: unmanaged section '{line.strip()}' "
                        "(only '## Active' and '## " + self.inactive_label + "' are allowed)")
                i += 1
                continue
            m = _ENTRY_RE.match(line)
            if m:
                flush()
                if section_active is None:
                    raise MarkdownFormatError(f"{self.filename}: entry outside a section")
                entry_id, title = m.group(1), m.group(2)
                if not entry_id.startswith(self.prefix + "-") or entry_id.split("-")[0] != self.prefix:
                    raise MarkdownFormatError(f"{self.filename}: id {entry_id} does not belong here")
                if entry_id in seen:
                    raise MarkdownFormatError(f"{self.filename}: duplicate id {entry_id}")
                seen.add(entry_id)
                current = MemoryEntry(entry_id, self.category, title, "", section_active)
                if i + 1 < len(lines):
                    mm = _META_RE.match(lines[i + 1].strip())
                    if mm:
                        meta = dict(kv.split("=", 1) for kv in mm.group(1).split() if "=" in kv)
                        if "status" in meta:
                            current.active = meta["status"] == "active"
                        current.created_at = meta.get("created", "")
                        current.updated_at = meta.get("updated", "")
                        current.source = meta.get("source", "")
                        i += 1
                i += 1
                continue
            if line.startswith("### "):
                raise MarkdownFormatError(f"{self.filename}: malformed entry heading {line!r}")
            if current is None:
                if line.strip():
                    raise MarkdownFormatError(
                        f"{self.filename}: text outside any entry: {line.strip()[:60]!r}")
            else:
                body.append(line)
            i += 1
        flush()
        return MarkdownDocument(heading, preamble, entries)

    def render(self, doc: MarkdownDocument) -> str:
        out = [f"# {doc.heading}", ""]
        if doc.preamble:
            out += doc.preamble + [""]
        for label, active in (("Active", True), (self.inactive_label, False)):
            out += [f"## {label}", ""]
            for e in doc.entries:
                if e.active != active:
                    continue
                out.append(f"### {e.entry_id} — {e.title}")
                meta = [f"status={'active' if e.active else 'inactive'}"]
                if e.created_at:
                    meta.append(f"created={_meta_value(e.created_at)}")
                if e.updated_at:
                    meta.append(f"updated={_meta_value(e.updated_at)}")
                if e.source:
                    meta.append(f"source={_meta_value(e.source)}")
                out.append(f"<!-- meta: {' '.join(meta)} -->")
                out.append(e.content)
                out.append("")
        return "\n".join(out).rstrip() + "\n"

    def entries(self, active_only: bool = False) -> list[MemoryEntry]:
        es = self.load().entries
        return [e for e in es if e.active] if active_only else es

    def get(self, entry_id: str) -> MemoryEntry | None:
        return next((e for e in self.load().entries if e.entry_id == entry_id), None)

    def find_active_by_title(self, title: str) -> MemoryEntry | None:
        t = normalize_title(title).lower()
        return next((e for e in self.load().entries
                     if e.active and e.title.lower() == t), None)

    def token_size(self) -> int:
        from .util import estimate_tokens
        return estimate_tokens(self.path.read_text(encoding="utf-8")) if self.path.exists() else 0

    # ----------------------------------------------------------------- write
    def add(self, title: str, content: str, source: str = "") -> MemoryEntry:
        with self._lock:
            doc = self.load()
            ts = now_iso()
            entry = MemoryEntry(self._next_id(doc), self.category, normalize_title(title),
                                normalize_content(content), True, ts, ts, source)
            doc.entries.append(entry)
            self._atomic_write(doc)
            return entry

    def update(self, entry_id: str, *, title: str | None = None, content: str | None = None,
               source: str = "") -> MemoryEntry:
        with self._lock:
            doc = self.load()
            idx = self._index(doc, entry_id)
            old = doc.entries[idx]
            new = replace(old,
                          title=normalize_title(title) if title else old.title,
                          content=normalize_content(content) if content is not None else old.content,
                          active=True,
                          updated_at=now_iso(),
                          source=source or old.source)
            doc.entries[idx] = new
            self._atomic_write(doc)
            return new

    def deactivate(self, entry_id: str, source: str = "") -> MemoryEntry:
        with self._lock:
            doc = self.load()
            idx = self._index(doc, entry_id)
            new = replace(doc.entries[idx], active=False, updated_at=now_iso(),
                          source=source or doc.entries[idx].source)
            doc.entries[idx] = new
            self._atomic_write(doc)
            return new

    def replace_all(self, entries: list[MemoryEntry]) -> None:
        """Used by rebuild/reset. Still atomic and backed up."""
        with self._lock:
            doc = self.load() if self.path.exists() else MarkdownDocument(self.heading, [DEFAULT_PREAMBLE])
            doc.entries = list(entries)
            self._atomic_write(doc)

    # -------------------------------------------------------------- internals
    def _index(self, doc: MarkdownDocument, entry_id: str) -> int:
        for i, e in enumerate(doc.entries):
            if e.entry_id == entry_id:
                return i
        raise KeyError(f"{entry_id} not found in {self.filename}")

    def _next_id(self, doc: MarkdownDocument) -> str:
        nums = [int(e.entry_id.split("-")[1]) for e in doc.entries]
        return f"{self.prefix}-{(max(nums) + 1) if nums else 1:03d}"

    def _backup_current(self) -> None:
        if not (self.create_backups and self.path.exists()):
            return
        self.history_dir.mkdir(parents=True, exist_ok=True)
        stem = self.path.stem
        dest = self.history_dir / f"{stem}.{time.strftime('%Y-%m-%d_%H-%M-%S')}.{time.time_ns() % 1_000_000:06d}.md"
        shutil.copy2(self.path, dest)
        versions = sorted(self.history_dir.glob(f"{stem}.*.md"))
        for old in versions[:-self.versions_to_keep] if self.versions_to_keep > 0 else []:
            try:
                old.unlink()
            except OSError:
                pass

    def _atomic_write(self, doc: MarkdownDocument, backup: bool = True) -> None:
        text = self.render(doc)
        # Validate the rendered document before touching disk.
        check = self.parse(text)
        expected = [(e.entry_id, e.title, e.content, e.active) for e in doc.entries]
        got = [(e.entry_id, e.title, e.content, e.active) for e in check.entries]
        if sorted(expected) != sorted(got):
            raise MarkdownFormatError(f"{self.filename}: rendered document failed round-trip validation")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + f".tmp{os.getpid()}")
        try:
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            # Verify what actually landed on disk.
            if tmp.read_text(encoding="utf-8") != text:
                raise OSError(f"{tmp}: verification read-back mismatch")
            if backup:
                self._backup_current()
            os.replace(tmp, self.path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass


def all_stores(memory_dir: Path, *, history_dir: Path | None = None, create_backups: bool = True,
               versions_to_keep: int = 50) -> dict[Category, MarkdownStore]:
    return {c: MarkdownStore(memory_dir, c, history_dir=history_dir, create_backups=create_backups,
                             versions_to_keep=versions_to_keep) for c in Category}
