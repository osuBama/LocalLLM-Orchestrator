"""Applies validated memory changes. The only code path that writes memory.

Order per change: record 'pending' in SQLite -> apply to Markdown (atomic,
backed up) -> mirror into SQLite -> mark 'approved'. If Markdown fails the
change is marked 'failed' and the file is untouched.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .database import Database
from .markdown_store import MarkdownFormatError, MarkdownStore, all_stores
from .memory_validator import ValidatedChange, ValidationResult
from .schemas import Category, MemoryEntry, Operation
from .util import estimate_tokens, stamp

log = logging.getLogger("memory")


@dataclass
class AppliedChange:
    change_id: int
    status: str
    entry_id: str | None
    operation: str
    category: str
    detail: str = ""


class MemoryManager:
    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db
        self.project_id = config.memory.project_id
        self.stores: dict[Category, MarkdownStore] = all_stores(
            config.memory_dir, history_dir=config.history_dir,
            create_backups=config.memory.create_backups,
            versions_to_keep=config.memory.history_versions_per_file)

    def ensure_files(self) -> None:
        for s in self.stores.values():
            s.ensure_exists()

    def existing(self) -> dict[Category, list[MemoryEntry]]:
        return {c: s.entries() for c, s in self.stores.items()}

    # ------------------------------------------------------------- apply
    def apply(self, result: ValidationResult, *, conversation_id: str | None,
              task_id: int | None = None) -> list[AppliedChange]:
        out: list[AppliedChange] = []
        for rej in result.rejected:
            raw = rej.raw if isinstance(rej.raw, dict) else {}
            cid = self.db.record_change(
                conversation_id=conversation_id, task_id=task_id,
                category=str(raw.get("category"))[:40] if raw.get("category") else None,
                operation=str(raw.get("operation"))[:40] if raw.get("operation") else None,
                entry_key=None, title=str(raw.get("title", ""))[:200] or None,
                content=str(raw.get("content", ""))[:2000] or None,
                confidence=raw.get("confidence") if isinstance(raw.get("confidence"), (int, float)) else None,
                reason=str(raw.get("reason", ""))[:500] or None,
                status="rejected", detail=rej.reason)
            log.info("memory change rejected", extra={"conversation_id": conversation_id,
                                                      "detail": rej.reason, "change_id": cid})
            out.append(AppliedChange(cid, "rejected", None, str(raw.get("operation")),
                                     str(raw.get("category")), rej.reason))
        for vc in result.accepted:
            out.append(self._apply_one(vc, conversation_id, task_id))
        return out

    def _apply_one(self, vc: ValidatedChange, conversation_id: str | None,
                   task_id: int | None) -> AppliedChange:
        cid = self.db.record_change(
            conversation_id=conversation_id, task_id=task_id, category=vc.category.value,
            operation=vc.operation.value, entry_key=vc.target_id, title=vc.title,
            content=vc.content, confidence=vc.confidence, reason=vc.reason,
            status="pending", detail=vc.note or None)
        store = self.stores[vc.category]
        source = conversation_id or ""
        try:
            if vc.operation == Operation.add:
                entry = store.add(vc.title, vc.content, source)
            elif vc.operation == Operation.update:
                entry = store.update(vc.target_id, title=vc.title, content=vc.content, source=source)
            else:
                entry = store.deactivate(vc.target_id, source=source)
        except (MarkdownFormatError, KeyError, OSError) as e:
            self.db.set_change_status(cid, "failed", f"{type(e).__name__}: {e}")
            log.error("memory change failed", extra={"conversation_id": conversation_id,
                                                     "change_id": cid, "detail": str(e)})
            return AppliedChange(cid, "failed", vc.target_id, vc.operation.value,
                                 vc.category.value, str(e))
        try:
            self.db.upsert_memory(entry, self.project_id)
        except Exception as e:  # SQLite trouble must not undo a good Markdown write.
            log.error("sqlite mirror failed", extra={"change_id": cid, "detail": str(e)})
        self.db.set_change_status(cid, "approved", vc.note or None, entry.entry_id)
        log.info("memory change applied", extra={
            "conversation_id": conversation_id, "change_id": cid, "entry_id": entry.entry_id,
            "operation": vc.operation.value, "category": vc.category.value})
        return AppliedChange(cid, "approved", entry.entry_id, vc.operation.value, vc.category.value,
                             vc.note)

    # ------------------------------------------------------ rebuild / sync
    def sync_db_from_markdown(self) -> int:
        """Markdown is canonical: rebuild the SQLite mirror from it."""
        entries = [e for s in self.stores.values() for e in s.entries()]
        self.db.replace_memories(entries, self.project_id)
        return len(entries)

    def reset_markdown(self) -> None:
        """Start from empty memory files (after a snapshot). Used by replay rebuilds."""
        for s in self.stores.values():
            s.replace_all([])

    # ----------------------------------------------------------- validate
    def validate_files(self) -> list[dict]:
        report = []
        limit = self.config.memory.max_memory_file_tokens
        for cat, s in self.stores.items():
            item = {"file": s.filename, "ok": True, "entries": 0, "active": 0, "tokens": 0,
                    "warnings": []}
            try:
                entries = s.entries()
                item["entries"] = len(entries)
                item["active"] = sum(1 for e in entries if e.active)
                item["tokens"] = s.token_size()
                if item["tokens"] > limit:
                    item["warnings"].append(f"file is ~{item['tokens']} tokens (> {limit}); "
                                            "consider consolidating")
            except MarkdownFormatError as e:
                item["ok"] = False
                item["error"] = str(e)
            report.append(item)
        return report

    # ------------------------------------------------------------- backup
    def snapshot(self, label: str = "") -> Path:
        """backups/YYYY-MM-DD_HH-MM-SS[_label]/ with memory Markdown, SQLite and config."""
        name = stamp() + (f"_{label}" if label else "")
        dest = self.config.backups_dir / name
        n = 1
        while dest.exists():
            dest = self.config.backups_dir / f"{name}_{n}"
            n += 1
        (dest / "memory").mkdir(parents=True)
        for s in self.stores.values():
            if s.path.exists():
                shutil.copy2(s.path, dest / "memory" / s.filename)
        self.db.backup_to(dest / "memory.db")
        if self.config.source_path and Path(self.config.source_path).exists():
            shutil.copy2(self.config.source_path, dest / "config.yaml")
        log.info("backup created", extra={"detail": str(dest)})
        return dest

    def restore(self, backup_dir: Path) -> dict:
        """Put the memory Markdown from a backup snapshot back (after snapshotting the present)."""
        src = Path(backup_dir) / "memory"
        if not src.is_dir():
            raise FileNotFoundError(f"{src} not found")
        files = {s.filename: s for s in self.stores.values()}
        found = [p for p in src.glob("*.md") if p.name in files]
        if not found:
            raise FileNotFoundError(f"no memory files in {src}")
        for p in found:  # validate everything before touching anything
            files[p.name].parse(p.read_text(encoding="utf-8"))
        before = self.snapshot("pre-restore")
        for p in found:
            store = files[p.name]
            store.replace_all(store.parse(p.read_text(encoding="utf-8")).entries)
        n = self.sync_db_from_markdown()
        log.info("memory restored", extra={"detail": f"from {backup_dir}; pre-restore backup {before}"})
        return {"restored_from": str(backup_dir), "files": sorted(p.name for p in found),
                "pre_restore_backup": str(before), "db_entries": n}

    def memory_tokens(self) -> int:
        return sum(estimate_tokens(s.path.read_text(encoding="utf-8"))
                   for s in self.stores.values() if s.path.exists())
