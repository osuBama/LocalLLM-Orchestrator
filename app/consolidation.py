"""Idle-time memory consolidation (Phase 4, conservative).

What it does, per category:
  * MERGE near-duplicate entries into one (the others are deactivated, so they
    stay in the file's inactive section and in history);
  * REWRITE overgrown entries more briefly;
  * list active entries that were never injected for N days, for a human to
    review. Nothing is ever deleted or deactivated on usage alone.

How it stays safe:
  * the memory model only sees groups picked by a deterministic similarity
    pre-filter, never the whole store;
  * proposals are checked in code: ids must exist, be active and belong to the
    group; the shared safety rules apply; merged/rewritten text must keep
    `identifier_coverage` of the names, numbers, paths and versions of the
    originals; rewrites must actually be shorter;
  * a backup snapshot is taken before anything is applied, and every applied or
    rejected proposal is recorded in memory_changes;
  * at most `max_changes_per_run` entries change per run, and the run stops as
    soon as real work (summaries, digests, extraction) is waiting.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .markdown_store import normalize_content, normalize_title
from .memory_retriever import tokenize
from .memory_validator import ValidatedChange, safety_issue, similar
from .schemas import Category, MemoryEntry, Operation
from .util import estimate_tokens, now_iso

log = logging.getLogger("memory")

CONSOLIDATION_SOURCE = "consolidation"

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "merges": {"type": "array", "items": {"type": "object", "properties": {
            "keep": {"type": "string"}, "remove": {"type": "array", "items": {"type": "string"}},
            "title": {"type": "string"}, "content": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["keep", "remove", "title", "content", "reason"]}},
        "rewrites": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "title": {"type": "string"}, "content": {"type": "string"},
            "reason": {"type": "string"}}, "required": ["id", "title", "content", "reason"]}},
    },
    "required": ["merges", "rewrites"],
}

# Things that must survive a merge or rewrite: anything with a digit, a path or
# URL fragment, a dotted/colon/underscored identifier, or an ALL-CAPS acronym.
_IDENT_RE = re.compile(r"[A-Za-z0-9_.:/\\-]*\d[A-Za-z0-9_.:/\\-]*"
                       r"|[A-Za-z0-9_-]*[./\\:_][A-Za-z0-9_./\\:-]+"
                       r"|\b[A-Z]{2,}[A-Z0-9]*\b")


def identifiers(text: str) -> set[str]:
    out = set()
    for m in _IDENT_RE.findall(text or ""):
        t = m.strip(".,:;-_/\\()[]'\"").lower()
        if len(t) >= 2:
            out.add(t)
    return out


def coverage(originals: list[str], new_text: str) -> float:
    need = set().union(*(identifiers(t) for t in originals)) if originals else set()
    if not need:
        return 1.0
    have = (new_text or "").lower()
    return sum(1 for t in need if t in have) / len(need)


def entry_similarity(a: MemoryEntry, b: MemoryEntry) -> float:
    ta, tb = set(tokenize(a.title + " " + a.content)), set(tokenize(b.title + " " + b.content))
    jacc = len(ta & tb) / len(ta | tb) if ta and tb else 0.0
    return max(jacc, similar(a.content, b.content), similar(a.title, b.title) * 0.9)


def clusters(entries: list[MemoryEntry], threshold: float, max_size: int) -> list[list[MemoryEntry]]:
    """Groups of 2+ mutually-related entries (single-link, capped in size)."""
    parent = {e.entry_id: e.entry_id for e in entries}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(entries):
        for b in entries[i + 1:]:
            if entry_similarity(a, b) >= threshold:
                parent[find(a.entry_id)] = find(b.entry_id)
    groups: dict[str, list[MemoryEntry]] = {}
    for e in entries:
        groups.setdefault(find(e.entry_id), []).append(e)
    out = []
    for g in groups.values():
        g.sort(key=lambda e: int(e.entry_id.split("-")[1]))
        for i in range(0, len(g), max_size):
            chunk = g[i:i + max_size]
            if len(chunk) >= 2:
                out.append(chunk)
    return out


@dataclass
class RunReport:
    started_at: str
    trigger: str
    dry_run: bool = False
    backup: str | None = None
    jobs: int = 0
    proposals: int = 0
    applied: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    interrupted: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class Consolidator:
    def __init__(self, orch):
        self.orch = orch
        self.cfg = orch.config.consolidation
        self.max_chars = orch.config.memory.max_entry_chars
        self.prompt = orch.config.prompt("consolidation.txt")
        self.last_report: RunReport | None = None

    # ---------------------------------------------------------------- gating
    def due(self, now: float | None = None) -> tuple[bool, str]:
        if not self.cfg.enabled:
            return False, "disabled"
        now = now or time.time()
        idle_for = now - self.orch.last_request_at
        if idle_for < self.cfg.idle_minutes * 60:
            return False, f"not idle ({int(idle_for)}s)"
        if self.orch.db.pending_task_count():
            return False, "queue busy"
        last = float(self.orch.db.kv_get("consolidation.last_run_ts", "0") or 0)
        if now - last < self.cfg.min_interval_hours * 3600:
            return False, "ran recently"
        since = int(self.orch.db.kv_get("consolidation.last_change_id", "0") or 0)
        changed = self.orch.db.approved_changes_since(since, exclude_conversation=CONSOLIDATION_SOURCE)
        if changed < self.cfg.min_changes_since_last:
            return False, f"only {changed} memory changes since last run"
        return True, f"idle, {changed} changes since last run"

    def _interrupted(self, forced: bool) -> bool:
        if forced:
            return False
        if self.orch.db.pending_task_count():
            return True
        return time.time() - self.orch.last_request_at < self.cfg.idle_minutes * 60

    # ------------------------------------------------------------------ jobs
    def plan_jobs(self) -> list[tuple[str, Category, list[MemoryEntry]]]:
        jobs: list[tuple[str, Category, list[MemoryEntry]]] = []
        for cat, store in self.orch.manager.stores.items():
            active = store.entries(active_only=True)
            grouped: set[str] = set()
            for g in clusters(active, self.cfg.similarity_threshold, self.cfg.max_cluster_size):
                jobs.append(("merge", cat, g))
                grouped |= {e.entry_id for e in g}
            for e in active:
                if e.entry_id not in grouped and estimate_tokens(e.content) >= self.cfg.rewrite_min_tokens:
                    jobs.append(("rewrite", cat, [e]))
        return jobs

    def _messages(self, kind: str, cat: Category, group: list[MemoryEntry]) -> list[dict]:
        listing = "\n\n".join(f"[{e.entry_id}] {e.title}\n{e.content}" for e in group)
        ask = ("Do any of these entries duplicate or contain each other? Propose merges only if so."
               if kind == "merge" else
               f"This entry is long (~{estimate_tokens(group[0].content)} tokens). "
               "Propose a rewrite only if it can be said more briefly without losing anything.")
        return [{"role": "system", "content": self.prompt},
                {"role": "user", "content": f"CATEGORY: {cat.value}\n\nENTRIES:\n{listing}\n\n{ask}"}]

    # ------------------------------------------------------------ validation
    def _check_text(self, title: str, content: str, reason: str) -> str | None:
        if not title or not content:
            return "empty title or content"
        if len(content) > self.max_chars:
            return f"content too large ({len(content)} > {self.max_chars} chars)"
        return safety_issue(title, content, reason)

    def _validate(self, data: Any, cat: Category, group: list[MemoryEntry], kind: str,
                  used: set[str]) -> tuple[list[tuple[str, list[ValidatedChange], dict]], list[dict]]:
        """Returns (accepted proposals as change lists, rejections)."""
        by_id = {e.entry_id: e for e in group}
        ok: list[tuple[str, list[ValidatedChange], dict]] = []
        bad: list[dict] = []
        if not isinstance(data, dict):
            return [], [{"proposal": data, "reason": "not a JSON object"}]

        for m in (data.get("merges") or []) if kind == "merge" else []:
            reason = self._merge_issue(m, by_id, used)
            if reason:
                bad.append({"proposal": m, "reason": reason})
                continue
            keep, removes = m["keep"], list(dict.fromkeys(m["remove"]))
            title, content = normalize_title(m["title"]), normalize_content(m["content"])
            why = str(m.get("reason", ""))[:300]
            changes = [ValidatedChange(cat, Operation.update, title, content, 1.0,
                                       f"consolidation: merged {', '.join(removes)} into {keep}. {why}", keep)]
            for r in removes:
                changes.append(ValidatedChange(cat, Operation.deactivate, by_id[r].title, "", 1.0,
                                               f"consolidation: merged into {keep}", r))
            used |= {keep, *removes}
            ok.append(("merge", changes, m))

        for r in data.get("rewrites") or []:
            reason = self._rewrite_issue(r, by_id, used)
            if reason:
                bad.append({"proposal": r, "reason": reason})
                continue
            title, content = normalize_title(r["title"]), normalize_content(r["content"])
            ok.append(("rewrite", [ValidatedChange(cat, Operation.update, title, content, 1.0,
                                                   f"consolidation: rewrite. {str(r.get('reason', ''))[:300]}",
                                                   r["id"])], r))
            used.add(r["id"])
        return ok, bad

    def _merge_issue(self, m: Any, by_id: dict[str, MemoryEntry], used: set[str]) -> str | None:
        if not isinstance(m, dict) or not isinstance(m.get("remove"), list):
            return "malformed merge"
        keep, removes = m.get("keep"), [x for x in m["remove"] if isinstance(x, str)]
        if keep not in by_id:
            return f"keep id {keep!r} is not in this group"
        if not removes or len(removes) != len(m["remove"]):
            return "nothing (valid) to remove"
        for r in removes:
            if r not in by_id:
                return f"remove id {r!r} is not in this group"
            if r == keep:
                return "keep is also listed in remove"
        if used & {keep, *removes}:
            return "entry already changed in this run"
        title, content = normalize_title(str(m.get("title", ""))), normalize_content(str(m.get("content", "")))
        issue = self._check_text(title, content, str(m.get("reason", "")))
        if issue:
            return issue
        originals = [by_id[i].content for i in [keep, *removes]]
        cov = coverage(originals, content)
        if cov < self.cfg.identifier_coverage:
            return f"merged text keeps only {cov:.0%} of names/numbers/paths"
        return None

    def _rewrite_issue(self, r: Any, by_id: dict[str, MemoryEntry], used: set[str]) -> str | None:
        if not isinstance(r, dict):
            return "malformed rewrite"
        rid = r.get("id")
        if rid not in by_id:
            return f"id {rid!r} is not in this group"
        if rid in used:
            return "entry already changed in this run"
        title, content = normalize_title(str(r.get("title", ""))), normalize_content(str(r.get("content", "")))
        issue = self._check_text(title, content, str(r.get("reason", "")))
        if issue:
            return issue
        old = by_id[rid].content
        if estimate_tokens(content) > estimate_tokens(old) * self.cfg.rewrite_max_ratio:
            return "rewrite is not meaningfully shorter"
        cov = coverage([old], content)
        if cov < self.cfg.identifier_coverage:
            return f"rewrite keeps only {cov:.0%} of names/numbers/paths"
        return None

    # ------------------------------------------------------------------- run
    async def run(self, *, trigger: str = "idle", dry_run: bool = False, force: bool = False) -> RunReport:
        orch = self.orch
        report = RunReport(started_at=now_iso(), trigger=trigger, dry_run=dry_run)
        start_change_id = orch.db.max_change_id()
        jobs = self.plan_jobs()
        report.jobs = len(jobs)
        used: set[str] = set()
        budget = self.cfg.max_changes_per_run
        try:
            for kind, cat, group in jobs:
                if budget <= 0:
                    break
                if self._interrupted(force):
                    report.interrupted = True
                    break
                if used & {e.entry_id for e in group}:
                    continue
                # Re-read: entries may have changed since planning (we're between real tasks).
                current = {e.entry_id: e for e in orch.manager.stores[cat].entries(active_only=True)}
                group = [current[e.entry_id] for e in group if e.entry_id in current]
                if (kind == "merge" and len(group) < 2) or not group:
                    continue
                think = orch.config.memory.think_consolidation
                msgs = self._messages(kind, cat, group)
                data = None
                for attempt_think in ([True, False] if think else [False]):
                    resp = await orch.memory_client.chat(msgs, format=SCHEMA, options={"temperature": 0.1},
                                                         think=attempt_think)
                    try:
                        data = json.loads(resp["message"]["content"])
                        break
                    except (json.JSONDecodeError, TypeError):
                        data = None
                if data is None:
                    report.rejected.append({"category": cat.value, "reason": "invalid JSON"})
                    continue
                accepted, rejected = self._validate(data, cat, group, kind, used)
                report.proposals += len(accepted) + len(rejected)
                for rj in rejected:
                    rj["category"] = cat.value
                    report.rejected.append(rj)
                    if not dry_run:
                        orch.db.record_change(
                            conversation_id=CONSOLIDATION_SOURCE, task_id=None, category=cat.value,
                            operation="consolidate", entry_key=None, title=None,
                            content=json.dumps(rj.get("proposal"), ensure_ascii=False)[:2000],
                            confidence=None, reason=None, status="rejected", detail=rj["reason"])
                for what, changes, raw in accepted:
                    n = len(changes)
                    if n > budget:
                        report.rejected.append({"category": cat.value, "proposal": raw,
                                                "reason": "over max_changes_per_run"})
                        continue
                    if not dry_run and report.backup is None and orch.config.memory.create_backups:
                        report.backup = str(orch.manager.snapshot("consolidation"))
                    results = []
                    for vc in changes:
                        if dry_run:
                            results.append({"op": vc.operation.value, "id": vc.target_id})
                            continue
                        r = orch.manager._apply_one(vc, CONSOLIDATION_SOURCE, None)
                        results.append({"op": r.operation, "id": r.entry_id, "status": r.status})
                    budget -= n
                    report.applied.append({"category": cat.value, "kind": what, "changes": results,
                                           "title": changes[0].title})
        except Exception as e:  # the memory GPU being down must not crash the worker
            report.error = f"{type(e).__name__}: {e}"
            log.warning("consolidation stopped", extra={"detail": report.error})
        if report.applied and not dry_run and getattr(orch, "indexer", None) is not None:
            orch.queue_embed()
        if not dry_run and not report.interrupted and not report.error:
            orch.db.kv_set("consolidation.last_run_ts", str(time.time()))
            orch.db.kv_set("consolidation.last_change_id", str(max(start_change_id, orch.db.max_change_id())))
        orch.db.kv_set("consolidation.last_report", json.dumps(report.to_dict(), ensure_ascii=False,
                                                                default=str)[:20000])
        self.last_report = report
        log.info("consolidation run", extra={"detail": f"jobs={report.jobs} applied={len(report.applied)} "
                                                       f"rejected={len(report.rejected)} "
                                                       f"interrupted={report.interrupted}"})
        return report

    # ---------------------------------------------------------------- review
    def review(self) -> list[dict]:
        """Active entries not injected for stale_after_days (or never), older than that too."""
        cutoff = datetime.now().astimezone() - timedelta(days=self.cfg.stale_after_days)
        usage = self.orch.db.usage(self.orch.project_id)
        out = []

        def parse(ts: str):
            try:
                return datetime.fromisoformat(ts)
            except (TypeError, ValueError):
                return None

        for cat, store in self.orch.manager.stores.items():
            for e in store.entries(active_only=True):
                created = parse(e.created_at)
                if created is None or created > cutoff:
                    continue
                u = usage.get(e.entry_id)
                last = parse(u["last_used_at"]) if u else None
                if last is None or last < cutoff:
                    out.append({"entry_id": e.entry_id, "category": cat.value, "title": e.title,
                                "created_at": e.created_at, "last_used_at": u["last_used_at"] if u else None,
                                "use_count": u["use_count"] if u else 0})
        return out
