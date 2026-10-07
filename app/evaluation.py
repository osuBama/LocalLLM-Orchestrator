"""Evaluation harness: replay sessions under config variants and compare.

Two measurements, both through the real orchestrator code path (an in-process
instance of the app, talking to your real Ollama instances):

1. Context replay (cost). Every recorded turn is re-sent with the *recorded*
   history and only 1 generated token, so all variants see identical history
   and differ only in context strategy. Per turn we record the prompt tokens
   Ollama actually processed (cache hits excluded), prefill time, memory
   tokens, history trimmed and tool tokens saved. Session summaries and tool
   digests are produced live by the memory model from the recorded answers,
   exactly as they would have been.

2. Golden questions (accuracy). After replaying a session, a question is
   asked with real generation (temperature 0, fixed seed) and graded
   deterministically: required strings, alternative strings, forbidden
   strings; /regex/ is supported.

Isolation: each variant runs in its own workspace under evals/runs/ with a
copy of your memory files and a fresh database. Your real memory, database
and logs are never written. A per-run marker at the very start of the system
prompt keeps one variant's cached prefix from helping another.
"""
from __future__ import annotations

import copy
import json
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import Config
from .conversation_logger import ConversationLogger
from .schemas import InteractionTask
from .util import now_iso, stamp

DEFAULT_CLIENT_SYSTEM = "You are a helpful assistant working on the user's project."

BUILTIN_VARIANTS: dict[str, dict[str, Any]] = {
    # Plain Ollama behaviour: full history, nothing injected, nothing compressed.
    "baseline": {"proxy.inject_memory": False, "proxy.append_system_prompt": False,
                 "proxy.trim_mode": "turns", "proxy.trim_trigger_user_turns": 0,
                 "compression.enabled": False,
                 "stable_memory.enabled": False, "flags.enabled": False,
                 "session.summaries_enabled": False},
    # Your config.yaml exactly as it is.
    "full": {},
}
BUILTIN_VARIANTS["baseline"].update({"embeddings.enabled": False, "history_recall.enabled": False})

# Settings the harness always forces, whatever the variant says.
_FORCED = {"memory.worker_enabled": False, "proxy.queue_memory_updates": False,
           "consolidation.enabled": False, "memory.create_backups": False,
           "application.log_level": "WARNING"}


# ------------------------------------------------------------------ inputs
class ToolEventIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: str
    arguments: Any = None
    result: str = ""


class TurnIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user: str
    assistant: str = ""
    tools: list[ToolEventIn] = Field(default_factory=list)


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    session: str | None = None          # a recorded conversation id (see `ai eval sessions`)
    turns: list[TurnIn] = Field(default_factory=list)   # or a scripted session
    upto_turn: int | None = Field(None, ge=0)            # replay only the first N turns
    as_of: str | None = None     # or: a fresh conversation at this moment (memory and history as of then)
    question: str
    expect_all: list[str] = Field(default_factory=list)
    expect_any: list[str] = Field(default_factory=list)
    forbid: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self):
        if sum(bool(x) for x in (self.session, self.turns, self.as_of)) > 1:
            raise ValueError(f"case {self.name!r}: use only one of session, turns or as_of")
        if not (self.expect_all or self.expect_any or self.forbid):
            raise ValueError(f"case {self.name!r}: needs expect_all, expect_any or forbid")
        return self


class GoldenFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cases: list[GoldenCase]


@dataclass
class EvalTurn:
    user: str
    assistant: str
    tool_events: list[dict] = field(default_factory=list)
    timestamp: str = ""          # when the user message was sent (recorded sessions only)


@dataclass
class EvalSession:
    id: str
    turns: list[EvalTurn]
    source: str = "recorded"


def load_recorded_sessions(conversations_dir: Path) -> dict[str, EvalSession]:
    sessions: dict[str, EvalSession] = {}
    for t in ConversationLogger(conversations_dir).iter_interactions():
        s = sessions.setdefault(t["conversation_id"], EvalSession(t["conversation_id"], []))
        s.turns.append(EvalTurn(t.get("user_message", ""), t.get("assistant_response", ""),
                                list(t.get("tool_events") or []), t.get("timestamp") or ""))
    return sessions


def load_golden(path: Path) -> list[GoldenCase]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return GoldenFile.model_validate(data).cases


def scripted_session(case: GoldenCase) -> EvalSession:
    turns = []
    for t in case.turns:
        events: list[dict] = []
        for ev in t.tools:
            events.append({"type": "tool_call", "tool": ev.tool, "arguments": ev.arguments})
            events.append({"type": "tool_result", "tool": ev.tool, "result": ev.result})
        turns.append(EvalTurn(t.user, t.assistant, events))
    return EvalSession(f"golden:{case.name}", turns, "scripted")


def load_variants(path: Path | None) -> dict[str, dict[str, Any]]:
    variants = dict(BUILTIN_VARIANTS)
    if path and Path(path).exists():
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for name, overrides in (data.get("variants") or {}).items():
            variants[str(name)] = dict(overrides or {})
    return variants


# ----------------------------------------------------------------- grading
def _match(pattern: str, text: str) -> bool:
    if len(pattern) > 2 and pattern.startswith("/") and pattern.endswith("/"):
        return re.search(pattern[1:-1], text, re.I | re.S) is not None
    return pattern.lower() in text.lower()


def grade(case: GoldenCase, answer: str) -> tuple[bool, list[str]]:
    problems = [f"missing {p!r}" for p in case.expect_all if not _match(p, answer)]
    if case.expect_any and not any(_match(p, answer) for p in case.expect_any):
        problems.append(f"none of {case.expect_any!r}")
    problems += [f"contains forbidden {p!r}" for p in case.forbid if _match(p, answer)]
    return not problems, problems


# ---------------------------------------------------------------- variants
def variant_config(base: Config, overrides: dict[str, Any], workspace: Path) -> Config:
    data = copy.deepcopy(base.model_dump())
    for key, value in {**overrides, **_FORCED}.items():
        node = data
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise ValueError(f"unknown setting {key!r}")
            node = node[p]
        if parts[-1] not in node:
            raise ValueError(f"unknown setting {key!r}")
        node[parts[-1]] = value
    data["paths"] = {**data["paths"], "root": str(workspace), "memory": "memory",
                     "conversations": "conversations", "database": "database/memory.db",
                     "logs": "logs", "backups": "backups", "prompts": str(base.prompts_dir)}
    data["source_path"] = None
    return Config(**data)


def history_messages(turns: list[EvalTurn], *, final_tools: bool) -> list[dict]:
    """Recorded turns as an Ollama/OpenClaw-style message list.

    The last turn's tool chain is included when `final_tools` (the final step of
    that turn); its assistant answer is never included (that is what we ask for).
    """
    msgs: list[dict] = []
    for i, t in enumerate(turns):
        last = i == len(turns) - 1
        msgs.append({"role": "user", "content": t.user})
        if not last or final_tools:
            for ev in t.tool_events:
                if ev.get("type") == "tool_call":
                    msgs.append({"role": "assistant", "content": "", "tool_calls": [
                        {"function": {"name": ev.get("tool", "?"), "arguments": ev.get("arguments") or {}}}]})
                elif ev.get("type") == "tool_result":
                    r = ev.get("result")
                    msgs.append({"role": "tool", "tool_name": ev.get("tool", "?"),
                                 "content": r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)})
        if not last:
            msgs.append({"role": "assistant", "content": t.assistant})
    return msgs


# --------------------------------------------------------------------- run
@dataclass
class TurnResult:
    variant: str
    session: str
    turn: int
    processed_tokens: int | None
    prefill_s: float | None
    total_s: float
    est_context_tokens: int | None
    memory_tokens: int = 0
    memory_base_tokens: int = 0
    user_turns_dropped: int = 0
    tool_tokens_saved: int = 0
    est_cache_hit: float | None = None
    est_full_tokens: int | None = None
    over_ctx: bool = False          # prompt larger than num_ctx: Ollama silently truncates it
    error: str | None = None


@dataclass
class GoldenResult:
    variant: str
    case: str
    passed: bool
    problems: list[str]
    answer: str
    processed_tokens: int | None
    total_s: float
    error: str | None = None
    repeat: int = 0
    memory_as_of: str | None = None
    approximate_memory: bool = False
    diagnosis: str | None = None          # see eval_diagnose.CATEGORIES
    evidence_in: list = field(default_factory=list)   # where the expected answer was in the prompt
    evidence_in_store: bool = False       # did memory / allowed past conversations contain it?


class VariantRunner:
    """One variant: its own workspace and in-process orchestrator."""

    def __init__(self, name: str, cfg: Config, *, client_system: str, think: bool | None,
                 extract: bool, primary_transport=None, memory_transport=None):
        from .api import create_app  # local import keeps `ai eval sessions` light
        self.name = name
        self.cfg = cfg
        self.app = create_app(cfg, primary_transport=primary_transport,
                              memory_transport=memory_transport, console_logs=False)
        self.orch = self.app.state.orch
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),
                                        base_url="http://eval", timeout=cfg.ollama.primary.timeout_seconds + 30)
        self.marker = f"[eval-run {uuid.uuid4().hex[:10]}]"
        self.orch.capture = {}            # record outgoing prompts for diagnosis
        self.client_system = f"{self.marker}\n{client_system}"
        self.think = think
        self.extract = extract

    async def aclose(self) -> None:
        await self.client.aclose()
        await self.orch.aclose()

    async def set_point_in_time(self, texts: dict[str, str], when, exclude: set[str]) -> None:
        """Put this variant's memory (and history recall) in the state of a past moment."""
        for store in self.orch.manager.stores.values():
            store.path.write_text(texts.get(store.filename, ""), encoding="utf-8")
        self.orch.manager.sync_db_from_markdown()
        self.orch._bases.clear()
        if self.orch.indexer is not None:
            self.orch.indexer.cutoff = when
            self.orch.indexer.exclude_conversations = set(exclude)
            await self.orch.indexer.sync_memory()

    def _body(self, model: str, msgs: list[dict], options: dict) -> dict:
        body = {"model": model, "stream": False, "options": options,
                "messages": [{"role": "system", "content": self.client_system}, *msgs]}
        if self.think is not None:
            body["think"] = self.think
        return body

    async def _drain(self) -> None:
        while await self.orch.worker.process_next():
            pass

    async def _send(self, cid: str, model: str, msgs: list[dict], options: dict) -> tuple[dict, dict, float]:
        t0 = time.perf_counter()
        r = await self.client.post("/api/chat", json=self._body(model, msgs, options),
                                   headers={"X-Conversation-Id": cid})
        dt = time.perf_counter() - t0
        rec = self.orch.metrics.requests[-1] if self.orch.metrics.requests else {}
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        return r.json(), rec, dt

    async def replay(self, session: EvalSession, model: str, max_turns: int | None,
                     measure: bool = True) -> list[TurnResult]:
        cid = f"eval-{self.name}-{session.id}"[:64]
        turns = session.turns[:max_turns] if max_turns else session.turns
        out: list[TurnResult] = []
        for i in range(len(turns)):
            upto = turns[: i + 1]
            res = TurnResult(self.name, session.id, i + 1, None, None, 0.0, None)
            if measure:
                try:
                    _, rec, dt = await self._send(cid, model, history_messages(upto, final_tools=True),
                                                  {"num_predict": 1, "temperature": 0})
                    res.total_s = round(dt, 3)
                    res.processed_tokens = rec.get("prompt_tokens")
                    res.prefill_s = rec.get("prefill_time")
                    res.est_context_tokens = rec.get("total_context_tokens")
                    res.memory_tokens = rec.get("memory_tokens", 0)
                    res.memory_base_tokens = rec.get("memory_base_tokens", 0)
                    res.user_turns_dropped = rec.get("user_turns_dropped", 0)
                    res.tool_tokens_saved = rec.get("tool_tokens_saved", 0)
                except Exception as e:
                    res.error = f"{type(e).__name__}: {e}"
                out.append(res)
            # Feed the memory model what really happened (the recorded answer), as production would.
            t = turns[i]
            task = InteractionTask(cid, now_iso(), t.user, t.assistant, list(t.tool_events),
                                   project_id=self.orch.project_id, source="eval", turn_number=i + 1)
            self.orch.worker.enqueue_summary(task)
            self.orch.worker.enqueue_digests(task)
            if self.orch.indexer is not None:
                self.orch.indexer.add_history(cid, i + 1, t.user, t.assistant, list(t.tool_events))
                self.orch.queue_embed()
            if self.extract:
                self.orch.worker.enqueue(task)
            await self._drain()
        _estimate_cache_hits(out, self.cfg.ollama.primary.num_ctx)
        return out

    def _evidence_sources(self, cid: str, turns: list[EvalTurn]) -> dict[str, str]:
        """What the system had available for this question (memory as of now, allowed history, the session)."""
        memory = "\n".join(f"{e.title}: {e.content}" for st in self.orch.manager.stores.values()
                           for e in st.entries(active_only=True))
        history = ""
        ix = self.orch.indexer
        if ix is not None and self.cfg.history_recall.enabled:
            from .eval_asof import parse_ts
            for c in self.orch.db.history_chunks().values():
                if c["conversation_id"] in ix.exclude_conversations or c["conversation_id"] == cid:
                    continue
                ts = parse_ts(c["created_at"])
                if ix.cutoff is not None and (ts is None or ts >= ix.cutoff):
                    continue
                history += c["text"] + "\n"
        session = "\n".join(f"{t.user}\n{t.assistant}\n" + "\n".join(
            str(ev.get("result") or "") for ev in t.tool_events) for t in turns)
        return {"memory": memory, "history": history, "session": session}

    def _diagnose(self, g: "GoldenResult", case: GoldenCase, cid: str, sources: dict[str, str]) -> None:
        from .eval_diagnose import diagnose, evidence_in, split_prompt
        sent = (self.orch.capture or {}).get(cid)
        if not sent:
            return
        parts = split_prompt(sent, case.question)
        g.diagnosis, g.evidence_in = diagnose(case, g.passed, parts, memory_text=sources["memory"],
                                              history_text=sources["history"], session_text=sources["session"])
        g.evidence_in_store = evidence_in(case, sources["memory"]) or evidence_in(case, sources["history"])

    async def golden(self, case: GoldenCase, session: EvalSession, model: str,
                     max_answer_tokens: int, seed: int, repeats: int = 1) -> list[GoldenResult]:
        """Replay once, then ask `repeats` times with seeds seed, seed+1, ...

        With repeats > 1 the temperature is 0.7 so different seeds actually sample
        different answers; with a single run it is 0 (deterministic).
        """
        turns = session.turns[: case.upto_turn] if case.upto_turn is not None else session.turns
        await self.replay(EvalSession(session.id, turns, session.source), model, None, measure=False)
        cid = f"eval-{self.name}-{session.id}"[:64]
        msgs = history_messages([*turns, EvalTurn(case.question, "")], final_tools=False)
        temperature = 0 if repeats <= 1 else 0.7
        sources = self._evidence_sources(cid, turns)
        out = []
        for r in range(max(1, repeats)):
            try:
                data, rec, dt = await self._send(cid, model, msgs, {"num_predict": max_answer_tokens,
                                                                   "temperature": temperature, "seed": seed + r})
                answer = (data.get("message") or {}).get("content", "") or ""
                ok, problems = grade(case, answer)
                g = GoldenResult(self.name, case.name, ok, problems, answer[:2000],
                                 rec.get("prompt_tokens"), round(dt, 3), repeat=r)
                self._diagnose(g, case, cid, sources)
                out.append(g)
            except Exception as e:
                out.append(GoldenResult(self.name, case.name, False, ["request failed"], "", None, 0.0,
                                        f"{type(e).__name__}: {e}", repeat=r))
        return out


def _estimate_cache_hits(results: list[TurnResult], num_ctx: int | None = None) -> None:
    """Calibrate on the first (cold, thanks to the run marker) request, then estimate.

    Ollama reports only the tokens it had to process. The first request of a
    run cannot hit the cache, so processed/estimated gives a per-run scale
    factor (it also absorbs chat-template and tool-schema overhead); later turns
    are compared against their scaled estimate.
    """
    first = next((r for r in results if r.processed_tokens and r.est_context_tokens), None)
    if not first:
        return
    k = first.processed_tokens / max(1, first.est_context_tokens)
    first.est_cache_hit = 0.0
    for r in results:
        if not r.est_context_tokens:
            continue
        full = r.est_context_tokens * k
        r.est_full_tokens = int(full)
        r.over_ctx = bool(num_ctx) and full > num_ctx
        if r is first or not r.processed_tokens:
            continue
        r.est_cache_hit = round(min(1.0, max(0.0, 1 - r.processed_tokens / full)), 3)


# ------------------------------------------------------------------ driver
async def run_eval(base: Config, *, variants: dict[str, dict], variant_names: list[str],
                   sessions: list[EvalSession], golden: list[GoldenCase], max_turns: int | None = None,
                   memory: str = "asof", extract: bool = False, think: bool | None = False,
                   client_system: str = DEFAULT_CLIENT_SYSTEM, max_answer_tokens: int = 512,
                   seed: int = 42, repeats: int = 1, out_dir: Path | None = None, primary_transport=None,
                   memory_transport=None, progress=print) -> dict:
    """Run every variant on every session and golden case.

    Fairness:
      * memory="asof" (default): each recorded session/case starts from memory as it was
        at that moment (no knowledge from the future); scripted cases start empty.
      * variants are interleaved per session/case in a seeded random order, so no
        variant is systematically first (cold) or last (warm).
    """
    import random
    from .eval_asof import empty_memory, memory_as_of, parse_ts

    if memory not in ("asof", "current", "empty"):
        raise ValueError("memory must be asof, current or empty")
    for name in variant_names:
        if name not in variants:
            raise ValueError(f"unknown variant {name!r}; known: {', '.join(variants)}")
    out_dir = Path(out_dir or (base.root_dir / "evals"))
    run_dir = out_dir / "runs" / stamp()
    model = base.ollama.primary.model
    rng = random.Random(seed)
    turn_results: list[TurnResult] = []
    golden_results: list[GoldenResult] = []
    order_log: list[dict] = []
    current_texts = {p.name: p.read_text(encoding="utf-8") for p in base.memory_dir.glob("*.md")} \
        if base.memory_dir.exists() else {}
    empty_texts = empty_memory(base.memory_dir)

    def memory_for(when_iso: str | None, scripted: bool) -> tuple[dict[str, str], object, bool]:
        """(texts, cutoff datetime or None, approximate)."""
        if memory == "empty" or (memory == "asof" and scripted):
            from datetime import datetime, timezone
            return empty_texts, datetime(1970, 1, 1, tzinfo=timezone.utc), False
        if memory == "current":
            return current_texts, None, False
        when = parse_ts(when_iso)
        if when is None:
            return empty_texts, None, True
        texts, approx = memory_as_of(base.memory_dir, when, history_dir=base.history_dir,
                                     versions_to_keep=base.memory.history_versions_per_file)
        return texts, when, approx

    runners: dict[str, VariantRunner] = {}
    try:
        for name in variant_names:
            ws = run_dir / name
            (ws / "memory").mkdir(parents=True, exist_ok=True)
            cfg = variant_config(base, variants[name], ws)
            runners[name] = VariantRunner(name, cfg, client_system=client_system, think=think, extract=extract,
                                          primary_transport=primary_transport, memory_transport=memory_transport)
        for name, runner in runners.items():
            progress(f"[{name}] preparing (warm-up{', history index' if runner.orch.indexer else ''})…")
            await runner._drain()
            if runner.orch.indexer is not None and runner.cfg.history_recall.enabled:
                # Real past conversations, with their original timestamps; recall is then
                # limited per case to what existed before that case.
                runner.orch.indexer.backfill_history_from_logs(base.conversations_dir)
                runner.orch.queue_embed()
                await runner._drain()
            try:
                await runner.orch.primary.chat([{"role": "user", "content": "hi"}], options={"num_predict": 1})
                await runner.orch.memory_client.chat([{"role": "user", "content": "hi"}], options={"num_predict": 1})
            except Exception as e:
                progress(f"[{name}] warm-up failed: {e}")

        def shuffled() -> list[str]:
            names = list(variant_names)
            rng.shuffle(names)
            return names

        for s in sessions:
            when = s.turns[0].timestamp if s.turns else None
            texts, cutoff, approx = memory_for(when, s.source == "scripted")
            order = shuffled()
            order_log.append({"session": s.id, "order": order})
            for name in order:
                progress(f"[{name}] replaying {s.id} ({len(s.turns)} turns)")
                await runners[name].set_point_in_time(texts, cutoff, {s.id})
                turn_results += await runners[name].replay(s, model, max_turns)

        recorded = None
        for case in golden:
            if case.session:
                if recorded is None:
                    recorded = load_recorded_sessions(base.conversations_dir)
                if case.session not in recorded:
                    for name in variant_names:
                        golden_results.append(GoldenResult(name, case.name, False, ["session not found"], "", None,
                                                           0.0, f"no recorded session {case.session}"))
                    continue
                src = recorded[case.session]
                upto = case.upto_turn if case.upto_turn is not None else len(src.turns)
                # The moment the question was (or would have been) asked.
                ask_turn = src.turns[upto] if upto < len(src.turns) else (src.turns[-1] if src.turns else None)
                when = ask_turn.timestamp if ask_turn else None
                texts, cutoff, approx = memory_for(when, False)
            elif case.as_of:
                # A fresh conversation at a past moment: only memory and history recall can answer.
                src = EvalSession(f"asof:{case.name}", [], "asof")
                when = case.as_of
                texts, cutoff, approx = memory_for(when, False)
            else:
                src = scripted_session(case)
                when = None
                texts, cutoff, approx = memory_for(None, True)
            sess = EvalSession(f"{src.id}#{case.name}", src.turns, src.source)
            order = shuffled()
            order_log.append({"case": case.name, "order": order})
            for name in order:
                progress(f"[{name}] golden: {case.name}")
                await runners[name].set_point_in_time(texts, cutoff, {case.session} if case.session else set())
                for g in await runners[name].golden(case, sess, model, max_answer_tokens, seed, repeats):
                    g.memory_as_of = when if memory == "asof" else memory
                    g.approximate_memory = approx
                    golden_results.append(g)
    finally:
        for runner in runners.values():
            await runner.aclose()

    report = summarize(variant_names, turn_results, golden_results)
    report["paired"] = paired(variant_names, turn_results, golden_results)
    from .eval_diagnose import summarize_diagnoses
    report["diagnosis"] = summarize_diagnoses([asdict(g) for g in golden_results], variant_names)
    report["approximate_memory_cases"] = sorted({g.case for g in golden_results if g.approximate_memory})
    report.update(started=run_dir.name, model=model, memory_model=base.ollama.memory.model,
                  options={"max_turns": max_turns, "memory": memory, "extract": extract, "think": think,
                           "repeats": repeats, "seed": seed,
                           "sessions": [s.id for s in sessions], "golden_cases": [c.name for c in golden]},
                  order=order_log,
                  turns=[asdict(r) for r in turn_results], golden=[asdict(g) for g in golden_results])
    out_dir.joinpath("reports").mkdir(parents=True, exist_ok=True)
    report["regressions"] = compare_with_previous(report, out_dir / "reports")
    jpath = out_dir / "reports" / f"{run_dir.name}.json"
    mpath = out_dir / "reports" / f"{run_dir.name}.md"
    jpath.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    mpath.write_text(render_markdown(report), encoding="utf-8")
    report["files"] = {"json": str(jpath), "markdown": str(mpath), "workspaces": str(run_dir)}
    return report


def summarize(variant_names: list[str], turns: list[TurnResult], golden: list[GoldenResult]) -> dict:
    rows = {}
    for v in variant_names:
        ts = [t for t in turns if t.variant == v and not t.error]
        gs = [g for g in golden if g.variant == v]
        hits = [t.est_cache_hit for t in ts if t.est_cache_hit is not None and t.turn > 1]
        rows[v] = {
            "turns": len(ts),
            "errors": sum(1 for t in turns if t.variant == v and t.error),
            "processed_tokens": sum(t.processed_tokens or 0 for t in ts),
            "prefill_s": round(sum(t.prefill_s or 0 for t in ts), 3),
            "request_s": round(sum(t.total_s for t in ts), 3),
            "avg_memory_tokens": round(sum(t.memory_tokens for t in ts) / len(ts), 1) if ts else None,
            "avg_memory_base_tokens": round(sum(t.memory_base_tokens for t in ts) / len(ts), 1) if ts else None,
            "tool_tokens_saved": sum(t.tool_tokens_saved for t in ts),
            "turns_trimmed": sum(1 for t in ts if t.user_turns_dropped),
            "turns_over_ctx": sum(1 for t in ts if t.over_ctx),
            "max_context_tokens": max((t.est_full_tokens or 0 for t in ts), default=0),
            "est_cache_hit": round(sum(hits) / len(hits), 3) if hits else None,
            "golden_passed": sum(1 for g in gs if g.passed),
            "golden_total": len(gs),
            "golden_pass_rate_ci95": wilson(sum(1 for g in gs if g.passed), len(gs)),
        }
    base = variant_names[0] if variant_names else None
    for v, r in rows.items():
        b = rows.get(base)
        if b and v != base and b["processed_tokens"]:
            r["processed_tokens_vs_" + base] = round(r["processed_tokens"] / b["processed_tokens"] - 1, 3)
        if b and v != base and b["prefill_s"]:
            r["prefill_vs_" + base] = round(r["prefill_s"] / b["prefill_s"] - 1, 3)
    return {"baseline": base, "variants": rows}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: b = only A passed, c = only B passed."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(variant_names: list[str], turns: list[TurnResult], golden: list[GoldenResult]) -> dict:
    """Paired comparisons of every variant against the first, on identical items."""
    from statistics import median
    if not variant_names:
        return {}
    base = variant_names[0]
    out = {}
    g_by = {(g.variant, g.case, g.repeat): g for g in golden if not g.error}
    t_by = {(t.variant, t.session, t.turn): t for t in turns if not t.error and t.processed_tokens}
    for v in variant_names[1:]:
        keys = [(k[1], k[2]) for k in g_by if k[0] == base and (v, k[1], k[2]) in g_by]
        only_base = sum(1 for c, r in keys if g_by[(base, c, r)].passed and not g_by[(v, c, r)].passed)
        only_v = sum(1 for c, r in keys if g_by[(v, c, r)].passed and not g_by[(base, c, r)].passed)
        both = sum(1 for c, r in keys if g_by[(v, c, r)].passed and g_by[(base, c, r)].passed)
        tk = [(s, n) for (vv, s, n) in t_by if vv == base and (v, s, n) in t_by]
        ratios = [t_by[(v, s, n)].processed_tokens / t_by[(base, s, n)].processed_tokens for s, n in tk]
        out[v] = {
            "vs": base,
            "golden_pairs": len(keys), "both_pass": both,
            "only_this_passes": only_v, "only_baseline_passes": only_base,
            "mcnemar_p": round(mcnemar_exact(only_base, only_v), 4) if keys else None,
            "turn_pairs": len(tk),
            "median_token_ratio": round(median(ratios), 3) if ratios else None,
            "turns_cheaper": sum(1 for r in ratios if r < 1),
        }
    return out


def wilson(passed: int, total: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson score interval for a pass rate: honest about small samples."""
    if total == 0:
        return None
    p = passed / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denom
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def compare_with_previous(report: dict, reports_dir: Path) -> dict:
    """Compare with the most recent earlier report that ran the same variants."""
    names = list(report["variants"])
    prev = None
    for p in sorted(Path(reports_dir).glob("*.json"), reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("started") != report.get("started") and list(data.get("variants", {})) == names:
            prev = (p.name, data)
            break
    if prev is None:
        return {"previous": None, "flags": []}
    flags = []
    for v, r in report["variants"].items():
        old = prev[1]["variants"].get(v) or {}
        if r.get("golden_total") and old.get("golden_total"):
            now_rate = r["golden_passed"] / r["golden_total"]
            old_rate = old["golden_passed"] / old["golden_total"]
            old_ci = old.get("golden_pass_rate_ci95") or [0, 1]
            if now_rate < old_rate and now_rate < old_ci[0]:
                flags.append(f"{v}: golden pass rate fell {old_rate:.0%} -> {now_rate:.0%} "
                             f"(below the previous run's 95% interval)")
        if r.get("processed_tokens") and old.get("processed_tokens") and r["turns"] == old.get("turns"):
            change = r["processed_tokens"] / old["processed_tokens"] - 1
            if change > 0.1:
                flags.append(f"{v}: processed prompt tokens up {change:+.0%} on the same sessions")
        if r.get("turns_over_ctx", 0) > old.get("turns_over_ctx", 0) and r["turns"] == old.get("turns"):
            flags.append(f"{v}: turns over num_ctx {old.get('turns_over_ctx', 0)} -> {r['turns_over_ctx']}")
    return {"previous": prev[0], "flags": flags}


def _pct(x):
    return "" if x is None else f"{x * 100:+.0f}%"


def render_markdown(report: dict) -> str:
    base = report["baseline"]
    lines = [f"# Evaluation {report.get('started', '')}", "",
             f"Primary `{report.get('model')}`, memory `{report.get('memory_model')}`. "
             f"Options: `{json.dumps(report.get('options', {}), ensure_ascii=False)}`", "",
             "## Cost (context replay)", "",
             f"| variant | turns | processed prompt tokens | vs {base} | prefill s | vs {base} | "
             "est. cache hit | peak context | turns over num_ctx | memory tok/turn (+base) | "
             "tool tokens saved | turns trimmed |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, r in report["variants"].items():
        hit = "" if r["est_cache_hit"] is None else f"{r['est_cache_hit'] * 100:.0f}%"
        lines.append(
            f"| {v} | {r['turns']}{' (' + str(r['errors']) + ' errors)' if r['errors'] else ''} | "
            f"{r['processed_tokens']:,} | {_pct(r.get('processed_tokens_vs_' + base))} | {r['prefill_s']} | "
            f"{_pct(r.get('prefill_vs_' + base))} | {hit} | ~{r['max_context_tokens']:,} | "
            f"{r['turns_over_ctx'] or ''}{' ⚠' if r['turns_over_ctx'] else ''} | "
            f"{r['avg_memory_tokens']} (+{r['avg_memory_base_tokens']}) | "
            f"{r['tool_tokens_saved']:,} | {r['turns_trimmed']} |")
    golden = report.get("golden") or []
    if golden:
        names = list(report["variants"])
        lines += ["", "## Accuracy (golden questions)", "",
                  "| case | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for case in dict.fromkeys(g["case"] for g in golden):
            cells = []
            for v in names:
                gs = [x for x in golden if x["case"] == case and x["variant"] == v]
                if not gs:
                    cells.append("")
                elif len(gs) == 1:
                    g = gs[0]
                    cells.append("pass" if g["passed"] else "FAIL: " + "; ".join(g["problems"]))
                else:
                    n_ok = sum(1 for g in gs if g["passed"])
                    worst = next((g for g in gs if not g["passed"]), None)
                    cells.append(f"{n_ok}/{len(gs)}" + ("" if worst is None else f" (e.g. {'; '.join(worst['problems'])})"))
            lines.append(f"| {case} | " + " | ".join(cells) + " |")
        lines += ["", "| variant | passed | 95% interval |", "|---|---|---|"]
        for v, r in report["variants"].items():
            ci = r.get("golden_pass_rate_ci95")
            ci_txt = "" if not ci else f"{ci[0]:.0%}–{ci[1]:.0%}"
            lines.append(f"| {v} | {r['golden_passed']}/{r['golden_total']} | {ci_txt} |")
        lines += ["", "Overlapping intervals mean the difference may be noise: add cases or use --repeats."]
    dg = report.get("diagnosis") or {}
    if any(sum(x["counts"].values()) for x in dg.values()):
        from .eval_diagnose import CATEGORIES, LABELS
        used = [c for c in CATEGORIES if any(x["counts"].get(c) for x in dg.values())]
        lines += ["", "## Why answers passed or failed", "",
                  "Where the expected answer was when the question was asked.", "",
                  "| variant | " + " | ".join(LABELS[c] for c in used) + " | memory reached the prompt |",
                  "|---|" + "---|" * (len(used) + 1)]
        for v, x in dg.items():
            mr = x["memory_retrieval"]
            rate = "–" if mr["rate"] is None else f"{mr['reached_prompt']}/{mr['available']}"
            lines.append(f"| {v} | " + " | ".join(str(x["counts"].get(c, 0)) for c in used) + f" | {rate} |")
        lines += ["", "*Model missed it*: the answer was in the prompt. *Not retrieved*: it was in memory or past "
                  "conversations but didn't reach the prompt. *Lost from the session*: said earlier in the "
                  "session, then trimmed or compressed away. *Never available*: nothing the system had contained "
                  "it. *Answered without evidence*: passed without the answer in the prompt (known or guessed)."]
    pr = report.get("paired") or {}
    if pr:
        lines += ["", f"## Paired comparison against {report['baseline']}", "",
                  "Same questions and the same turns for both variants, so only the differences count.", "",
                  "| variant | questions both pass | only this passes | only baseline passes | p (McNemar) | "
                  "median processed tokens vs baseline | turns cheaper |", "|---|---|---|---|---|---|---|"]
        for v, x in pr.items():
            p = x["mcnemar_p"]
            p_txt = "–" if p is None else (f"{p:.3f}" + (" significant" if p < 0.05 else ""))
            ratio = "–" if x["median_token_ratio"] is None else f"{(x['median_token_ratio'] - 1) * 100:+.0f}%"
            lines.append(f"| {v} | {x['both_pass']}/{x['golden_pairs']} | {x['only_this_passes']} | "
                         f"{x['only_baseline_passes']} | {p_txt} | {ratio} | {x['turns_cheaper']}/{x['turn_pairs']} |")
        lines += ["", "p < 0.05 means the accuracy difference is unlikely to be chance. Repeats of one question "
                  "are not fully independent, so treat p-values from --repeats as optimistic."]
    opts = report.get("options") or {}
    mem_note = {"asof": "each session and question started from memory as it was at that moment (no future "
                        "knowledge); scripted questions started empty",
                "current": "every session and question used today's memory, which may contain answers "
                           "learned later (flatters memory variants)",
                "empty": "every session and question started from empty memory"}.get(opts.get("memory"))
    if mem_note:
        lines += ["", f"Memory: {mem_note}. Variant order was shuffled per session and question (seed "
                  f"{opts.get('seed')})."]
    if report.get("approximate_memory_cases"):
        lines += [f"Approximate point-in-time memory (older snapshots already pruned): "
                  f"{', '.join(report['approximate_memory_cases'])}."]
    reg = report.get("regressions") or {}
    if reg.get("previous"):
        lines += ["", f"## Compared with {reg['previous']}", ""]
        lines += [f"- ⚠ {f}" for f in reg["flags"]] or ["- No regressions flagged."]
    lines += ["", "**Turns over num_ctx** means the prompt did not fit the model's context window: Ollama "
              "silently drops the oldest content, so cost numbers for those turns look fine while the model "
              "has lost information. Compare accuracy with golden questions before trusting a cheaper variant.",
              "", "Processed prompt tokens are what Ollama actually had to evaluate (cache hits excluded). "
              "Cache hit is an estimate calibrated on each run's first, cold request. History in replay is the "
              "recorded history, so every variant sees the same conversation; the client's real system prompt "
              "and tool schemas are not in the logs unless passed with --client-system."]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------- golden candidates
def accept_candidate(db, candidate_id: int, golden_path: Path, *, expect: list[str] | None = None,
                     forbid: list[str] | None = None, name: str | None = None) -> dict:
    """Turn a captured correction into a golden case appended to golden.yaml.

    The case replays the session up to just before the question that was answered
    wrongly, asks it again, and checks for what the correction said. Appends (keeps
    comments in the file), validates, and rolls back if the result does not load.
    """
    cand = next((c for c in db.candidates(status=None, limit=100000) if c["id"] == candidate_id), None)
    if cand is None:
        raise ValueError(f"no candidate #{candidate_id}")
    if cand["status"] != "pending":
        raise ValueError(f"candidate #{candidate_id} is already {cand['status']}")
    expect = list(expect or []) or list(cand["suggested_expect"])
    forbid = list(forbid or []) or list(cand.get("suggested_forbid") or [])
    if not expect and not forbid:
        raise ValueError("nothing to check: no expectation was suggested; pass --expect (and/or --forbid)")
    generated = cand.get("kind") == "generated"
    name = name or (f"generated-{candidate_id}" if generated else f"correction-{candidate_id}")
    upto = cand["upto_turn"] if generated and cand.get("upto_turn") is not None else max(0, cand["turn"] - 2)
    if generated and cand.get("as_of"):
        case = {"name": name, "as_of": cand["as_of"], "question": cand["question"]}
    else:
        case = {"name": name, "session": cand["conversation_id"], "upto_turn": upto, "question": cand["question"]}
    if expect:
        case["expect_all"] = expect
    if forbid:
        case["forbid"] = forbid
    GoldenCase.model_validate(case)

    golden_path = Path(golden_path)
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    original = golden_path.read_text(encoding="utf-8") if golden_path.exists() else None
    if original is not None:
        existing = yaml.safe_load(original) or {}
        if any(c.get("name") == name for c in existing.get("cases") or []):
            raise ValueError(f"a case named {name!r} already exists in {golden_path}")
    block = yaml.safe_dump([case], allow_unicode=True, sort_keys=False, width=100)
    block = "".join("  " + line if line.strip() else line for line in block.splitlines(True))
    origin = (f"generated #{candidate_id} from: {cand.get('context', '')[:80]!r}" if generated
              else f"from correction #{candidate_id}: {cand['correction'][:80]!r}")
    comment = f"  # {origin}\n".replace("\n", " ").rstrip() + "\n"
    if original is None or not (yaml.safe_load(original) or {}).get("cases"):
        text = "cases:\n" + comment + block
    else:
        text = original.rstrip("\n") + "\n\n" + comment + block
    golden_path.write_text(text, encoding="utf-8")
    try:
        load_golden(golden_path)
    except Exception as e:
        if original is None:
            golden_path.unlink()
        else:
            golden_path.write_text(original, encoding="utf-8")
        raise ValueError(f"golden file would not validate, left unchanged: {e}") from e
    db.set_candidate_status(candidate_id, "accepted", name)
    return {"accepted": candidate_id, "case": case, "file": str(golden_path)}
