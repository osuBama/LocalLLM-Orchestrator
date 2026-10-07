"""Web console at /ui: status, chat, memory, settings and eval comparison.

Served by the orchestrator itself (no extra service, works offline). The page is
a single static file (app/ui/index.html); everything it needs comes from the
JSON endpoints below plus the existing API.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import APP_DIR, load_config
from .schemas import Category

UI_FILE = APP_DIR / "ui" / "index.html"

# Settings the console exposes. live=True: applied to the running orchestrator on save
# (read at request time); live=False: saved, takes effect after a restart.
SETTINGS: list[dict[str, Any]] = [
    # group, key, label, help, type, extra
    {"group": "Models", "key": "ollama.primary.model", "label": "Primary model", "type": "str", "live": False,
     "help": "Model on the primary GPU. Pull it first; re-run setup.ps1 to re-fit the context."},
    {"group": "Models", "key": "ollama.primary.num_ctx", "label": "Primary context (tokens)", "type": "int",
     "min": 2048, "max": 262144, "live": False,
     "help": "Must fit 100% on the GPU (setup.ps1 measures it). Keep the client's context window equal to this."},
    {"group": "Models", "key": "ollama.memory.model", "label": "Memory model", "type": "str", "live": False,
     "help": "Model on the memory GPU that maintains memory, summaries and digests."},
    {"group": "Models", "key": "ollama.memory.num_ctx", "label": "Memory model context", "type": "int",
     "min": 2048, "max": 131072, "live": False, "help": "Context for the memory model."},
    {"group": "Models", "key": "embeddings.model", "label": "Embedding model", "type": "str", "live": False,
     "help": "Changing it requires a reindex (Memory tab)."},

    {"group": "Memory", "key": "memory.max_context_tokens", "label": "Memory budget per prompt", "type": "int",
     "min": 300, "max": 20000, "live": False,
     "help": "Total memory injected per prompt (the cached base is part of it). About 15% of the primary context."},
    {"group": "Memory", "key": "stable_memory.enabled", "label": "Cached memory base", "type": "bool", "live": True,
     "help": "Constraints, objectives and environment go into the cached system prompt."},
    {"group": "Memory", "key": "stable_memory.max_tokens", "label": "Memory base size", "type": "int",
     "min": 100, "max": 10000, "live": True, "help": "Part of the memory budget; must be smaller than it."},
    {"group": "Memory", "key": "memory.trigger_mode", "label": "When to extract memory", "type": "choice",
     "options": ["heuristic", "always"], "live": True,
     "help": "heuristic: only turns that look durable (corrections, errors, decisions...). always: every turn."},
    {"group": "Memory", "key": "memory.think_extraction", "label": "Memory model thinks on extraction",
     "type": "bool", "live": True, "help": "Slower but better judgement; runs in the background."},
    {"group": "Memory", "key": "session.summary_max_tokens", "label": "Session summary size", "type": "int",
     "min": 50, "max": 2000, "live": True, "help": "Raise if long sessions lose details after trimming."},

    {"group": "Context window", "key": "proxy.trim_mode", "label": "History trimming", "type": "choice",
     "options": ["size", "turns"], "live": True,
     "help": "size: only when the prompt would not fit. turns: fixed schedule by turn count."},
    {"group": "Context window", "key": "proxy.trim_target_ratio", "label": "Trim back to (share of window)",
     "type": "float", "min": 0.2, "max": 0.9, "step": 0.05, "live": True,
     "help": "Lower = deeper, rarer cuts: cheaper on long sessions, leans more on summaries."},
    {"group": "Context window", "key": "proxy.trim_keep_user_turns", "label": "Always keep recent turns",
     "type": "int", "min": 1, "max": 50, "live": True, "help": "Minimum number of recent turns kept verbatim."},
    {"group": "Context window", "key": "proxy.reply_reserve_tokens", "label": "Room reserved for the reply",
     "type": "int", "min": 0, "max": 16000, "live": True, "help": "Tokens kept free for the model's answer."},
    {"group": "Context window", "key": "compression.enabled", "label": "Compress old tool output", "type": "bool",
     "live": True, "help": "Old large tool results are replaced by digests when the window fills."},
    {"group": "Context window", "key": "compression.min_result_tokens", "label": "Compress results larger than",
     "type": "int", "min": 50, "max": 20000, "live": True, "help": "Smaller tool results are never digested."},

    {"group": "Thinking", "key": "thinking.mode", "label": "Primary thinking", "type": "choice",
     "options": ["auto", "client", "on", "off"], "live": True,
     "help": "auto: off only for clearly simple turns. client: never touched. on/off: forced."},
    {"group": "Thinking", "key": "thinking.simple_max_words", "label": "Simple turn: at most (words)",
     "type": "int", "min": 1, "max": 200, "live": True, "help": "Longer messages always keep thinking."},

    {"group": "Retrieval", "key": "embeddings.min_similarity", "label": "Semantic match threshold",
     "type": "float", "min": 0, "max": 1, "step": 0.05, "live": True,
     "help": "Entries found only by meaning must reach this. Raise if irrelevant memory appears."},
    {"group": "Retrieval", "key": "history_recall.enabled", "label": "Recall past conversations", "type": "bool",
     "live": True, "help": "Add short excerpts of older exchanges when relevant."},
    {"group": "Retrieval", "key": "history_recall.min_similarity", "label": "Recall threshold", "type": "float",
     "min": 0, "max": 1, "step": 0.05, "live": True, "help": "Raise if unrelated excerpts appear."},
    {"group": "Retrieval", "key": "history_recall.cue_min_similarity", "label": "Recall threshold after \"last time\"",
     "type": "float", "min": 0, "max": 1, "step": 0.05, "live": True,
     "help": "Lower bar when the prompt refers to the past."},
    {"group": "Retrieval", "key": "history_recall.max_tokens", "label": "Recall budget", "type": "int",
     "min": 0, "max": 5000, "live": True, "help": "Tokens of past excerpts per prompt."},

    {"group": "Maintenance", "key": "consolidation.enabled", "label": "Idle-time consolidation", "type": "bool",
     "live": True, "help": "Merge duplicates and tighten long entries when idle (with backups)."},
    {"group": "Maintenance", "key": "consolidation.idle_minutes", "label": "Idle after (minutes)", "type": "float",
     "min": 1, "max": 1440, "live": True, "help": "No requests for this long counts as idle."},
    {"group": "Maintenance", "key": "consolidation.min_interval_hours", "label": "At most every (hours)",
     "type": "float", "min": 0, "max": 720, "live": True, "help": "Minimum time between automatic runs."},
    {"group": "Maintenance", "key": "conversation.capture_corrections", "label": "Capture corrections for evals",
     "type": "bool", "live": True, "help": "\"No, it's X\" turns become golden-question candidates."},
    {"group": "Maintenance", "key": "flags.enabled", "label": "Primary model flags memory", "type": "bool",
     "live": False, "help": "The primary marks durable facts for the memory model (changes its system prompt)."},
]
_BY_KEY = {s["key"]: s for s in SETTINGS}


def _get(obj, dotted: str):
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _set(obj, dotted: str, value) -> None:
    parts = dotted.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


class SettingsUpdate(BaseModel):
    values: dict[str, Any]


class EntryEdit(BaseModel):
    category: Category
    entry_id: str | None = None          # None = add a new entry
    title: str | None = Field(None, max_length=120)
    content: str | None = Field(None, max_length=5000)
    active: bool | None = None


class EvalRun(BaseModel):
    variants: str = "baseline,full"
    last: int = Field(3, ge=0, le=50)
    repeats: int = Field(1, ge=1, le=20)
    think: str = "off"
    max_turns: int | None = Field(None, ge=1)
    golden_only: bool = False


class CandidateAccept(BaseModel):
    expect: list[str] = Field(default_factory=list)
    forbid: list[str] = Field(default_factory=list)
    name: str | None = None


def build_ui_router(orch) -> APIRouter:
    router = APIRouter()
    cfg = orch.config
    eval_job: dict[str, Any] = {}

    def evals_dir() -> Path:
        return cfg.root_dir / "evals"

    @router.get("/ui", include_in_schema=False)
    async def ui_page():
        return FileResponse(UI_FILE, media_type="text/html")

    # ------------------------------------------------------------ settings
    @router.get("/ui/api/settings")
    async def get_settings():
        items = []
        for s in SETTINGS:
            try:
                value = _get(cfg, s["key"])
            except AttributeError:
                continue
            items.append({**s, "value": value})
        return {"editable": bool(cfg.source_path), "file": cfg.source_path, "settings": items}

    @router.post("/ui/api/settings")
    async def save_settings(upd: SettingsUpdate):
        if not cfg.source_path:
            raise HTTPException(409, "This orchestrator was not started from a config file.")
        unknown = [k for k in upd.values if k not in _BY_KEY]
        if unknown:
            raise HTTPException(400, f"not editable here: {', '.join(unknown)}")
        changed = {k: v for k, v in upd.values.items() if _get(cfg, k) != v}
        if not changed:
            return {"saved": [], "applied_now": [], "restart_required": []}
        from .setup_tools import set_config
        res = set_config(Path(cfg.source_path), changed)
        if not res["ok"]:
            raise HTTPException(400, res["error"])
        fresh = load_config(cfg.source_path)
        applied, restart = [], []
        for k in changed:
            if _BY_KEY[k]["live"]:
                _set(cfg, k, _get(fresh, k))
                applied.append(k)
            else:
                restart.append(k)
        if "embeddings.min_similarity" in applied and hasattr(orch.retriever, "min_similarity"):
            orch.retriever.min_similarity = cfg.embeddings.min_similarity
        return {"saved": sorted(changed), "applied_now": applied, "restart_required": restart,
                "backup": res.get("backup")}

    # -------------------------------------------------------------- memory
    @router.get("/ui/api/memory")
    async def all_memory():
        usage = orch.db.usage(orch.project_id)
        out = {}
        for cat, store in orch.manager.stores.items():
            try:
                entries = store.entries()
            except Exception as e:
                out[cat.value] = {"error": str(e), "entries": []}
                continue
            out[cat.value] = {"file": store.filename, "entries": [{
                "entry_id": e.entry_id, "title": e.title, "content": e.content, "active": e.active,
                "created_at": e.created_at, "updated_at": e.updated_at,
                "use_count": (usage.get(e.entry_id) or {}).get("use_count", 0),
                "last_used_at": (usage.get(e.entry_id) or {}).get("last_used_at")} for e in entries]}
        return out

    @router.post("/ui/api/memory/entry")
    async def edit_entry(edit: EntryEdit):
        store = orch.manager.stores[edit.category]
        try:
            if edit.entry_id is None:
                if not (edit.title and edit.content):
                    raise HTTPException(400, "title and content are required for a new entry")
                entry = store.add(edit.title, edit.content, "ui")
                op = "add"
            elif edit.active is False:
                entry = store.deactivate(edit.entry_id, source="ui")
                op = "deactivate"
            else:
                entry = store.update(edit.entry_id, title=edit.title, content=edit.content, source="ui")
                op = "update"
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        orch.db.upsert_memory(entry, orch.project_id)
        orch.db.record_change(conversation_id="ui", task_id=None, category=edit.category.value, operation=op,
                              entry_key=entry.entry_id, title=entry.title, content=entry.content,
                              confidence=None, reason="edited in the console", status="approved")
        if orch.indexer is not None:
            orch.queue_embed()
        return {"entry_id": entry.entry_id, "operation": op}

    @router.post("/ui/api/memory/reindex")
    async def reindex():
        if orch.indexer is None:
            raise HTTPException(409, "Embeddings are disabled.")
        async with orch.worker.lock:
            m = await orch.indexer.sync_memory()
            created = orch.indexer.backfill_history_from_logs()
            h = 0
            while True:
                n = await orch.indexer.sync_history()
                if not n:
                    break
                h += n
        return {"memory_embedded": m, "history_created": created, "history_embedded": h}

    # --------------------------------------------------------------- evals
    @router.get("/ui/api/evals")
    async def list_reports():
        d = evals_dir() / "reports"
        items = []
        for p in sorted(d.glob("*.json"), reverse=True) if d.exists() else []:
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            items.append({"name": p.stem, "started": r.get("started"), "model": r.get("model"),
                          "variants": list(r.get("variants", {})),
                          "sessions": len((r.get("options") or {}).get("sessions") or []),
                          "golden_cases": len((r.get("options") or {}).get("golden_cases") or []),
                          "repeats": (r.get("options") or {}).get("repeats", 1),
                          "flags": (r.get("regressions") or {}).get("flags", [])})
        return {"reports": items}

    @router.get("/ui/api/evals/{name}")
    async def get_report(name: str):
        if not all(c.isalnum() or c in "-_" for c in name):
            raise HTTPException(400, "bad report name")
        p = evals_dir() / "reports" / f"{name}.json"
        if not p.exists():
            raise HTTPException(404, "no such report")
        return json.loads(p.read_text(encoding="utf-8"))

    @router.post("/ui/api/evals/run")
    async def run_eval(req: EvalRun):
        if eval_job.get("proc") is not None and eval_job["proc"].poll() is None:
            raise HTTPException(409, "An evaluation is already running.")
        if not cfg.source_path:
            raise HTTPException(409, "This orchestrator was not started from a config file.")
        if not all(c.isalnum() or c in "-_,." for c in req.variants):
            raise HTTPException(400, "bad variant list")
        args = [sys.executable, "-m", "app.cli", "--config", cfg.source_path, "eval", "run",
                "--variant", req.variants, "--repeats", str(req.repeats),
                "--think", req.think if req.think in ("off", "on", "default") else "off"]
        args += ["--last", "0"] if req.golden_only else ["--last", str(req.last)]
        if req.max_turns:
            args += ["--max-turns", str(req.max_turns)]
        cfg.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = cfg.logs_dir / f"eval-ui-{time.strftime('%Y%m%d-%H%M%S')}.log"
        fh = open(log_path, "w", encoding="utf-8")
        env = {**__import__("os").environ, "PYTHONPATH": str(APP_DIR.parent), "PYTHONUNBUFFERED": "1"}
        eval_job.update(proc=subprocess.Popen(args, cwd=str(APP_DIR.parent), stdout=fh, stderr=subprocess.STDOUT,
                                              env=env),
                        log=log_path, started=time.time(), args=args[4:], fh=fh)
        return {"started": True, "log": str(log_path)}

    @router.get("/ui/api/evals/run/status")
    async def run_status():
        proc = eval_job.get("proc")
        if proc is None:
            return {"running": False, "log": ""}
        code = proc.poll()
        if code is not None and eval_job.get("fh"):
            eval_job["fh"].close()
            eval_job["fh"] = None
        try:
            tail = Path(eval_job["log"]).read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        except OSError:
            tail = []
        return {"running": code is None, "exit_code": code, "args": eval_job.get("args"),
                "elapsed": round(time.time() - eval_job["started"]), "log": "\n".join(tail)}

    @router.post("/ui/api/candidates/{cid}/accept")
    async def accept(cid: int, body: CandidateAccept):
        from .evaluation import accept_candidate
        try:
            return accept_candidate(orch.db, cid, evals_dir() / "golden.yaml", expect=body.expect or None,
                                    forbid=body.forbid or None, name=body.name or None)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @router.post("/ui/api/candidates/{cid}/dismiss")
    async def dismiss(cid: int):
        if not orch.db.set_candidate_status(cid, "dismissed"):
            raise HTTPException(404, "no such candidate")
        return {"dismissed": cid}

    return router


def csrf_guard_paths(path: str) -> bool:
    """True if a state-changing request to `path` must carry the X-AI-Client header.

    /api/* and /v1/* (Ollama-compatible, used by OpenClaw) and /chat (JSON-only,
    so browsers cannot forge it cross-site) are exempt. Everything else that
    changes state (memory admin, settings, evals) needs a custom header, which a
    foreign web page cannot add without a CORS preflight this server never grants.
    """
    return not (path.startswith("/api/") or path.startswith("/v1/") or path == "/chat" or path == "/api")
