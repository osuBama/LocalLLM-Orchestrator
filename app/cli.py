"""Administration CLI (spec §42).

    ai serve
    ai ui                             # open the web console
    ai chat
    ai status | ai metrics
    ai memory show [category]
    ai memory search "MCP 404"
    ai memory context "prompt"        # what would be injected
    ai memory changes | ai memory tasks | ai memory sessions
    ai memory validate
    ai memory backup
    ai memory rebuild [--replay] [--reset]
    ai memory consolidate

`chat`, `status` and `metrics` talk to the running server. The memory
commands work on the files directly, so they also work with the server down.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx

from .config import load_config
from .schemas import Category


def _server(cfg) -> str:
    host = cfg.application.host if cfg.application.host not in ("0.0.0.0", "::") else "127.0.0.1"
    return f"http://{host}:{cfg.application.port}"


def _orch(cfg):
    from .orchestrator import Orchestrator
    return Orchestrator(cfg)


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def cmd_chat(cfg, args) -> int:
    url = _server(cfg) + "/chat"
    cid = args.conversation_id
    print("Chat with the primary model (Ctrl+C or /quit to exit).")
    with httpx.Client(timeout=cfg.ollama.primary.timeout_seconds + 30) as c:
        while True:
            try:
                msg = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not msg:
                continue
            if msg in ("/quit", "/exit"):
                return 0
            try:
                r = c.post(url, json={"message": msg, "conversation_id": cid})
                r.raise_for_status()
            except httpx.HTTPError as e:
                print(f"error: {e}")
                continue
            data = r.json()
            cid = data["conversation_id"]
            print(f"\nai> {data['response']}")
            if args.verbose:
                print(f"   [memory used: {', '.join(data['memory_entries_used']) or 'none'}; "
                      f"update queued: {data['memory_update_queued']}]")


def cmd_http_get(cfg, path: str) -> int:
    try:
        r = httpx.get(_server(cfg) + path, timeout=30)
        r.raise_for_status()
    except httpx.HTTPError as e:
        print(f"Server not reachable at {_server(cfg)} ({e}). Is `ai serve` running?")
        return 1
    _print(r.json())
    return 0


def cmd_memory(cfg, args) -> int:
    orch = _orch(cfg)
    try:
        sub = args.memory_cmd
        if sub == "show":
            cats = [Category(args.category)] if args.category else list(Category)
            for c in cats:
                store = orch.manager.stores[c]
                print(store.path.read_text(encoding="utf-8") if store.path.exists() else f"# {c.value}: (missing)")
        elif sub == "search":
            for h in orch.retriever.search(args.query, args.limit, include_inactive=args.all):
                flag = "" if h.entry.active else " (inactive)"
                print(f"{h.score:6.2f}  [{h.entry.entry_id}] {h.entry.title}{flag}\n        {h.entry.content}")
        elif sub == "context":
            base = orch.memory_base("__cli__", ("cli",))
            budget = cfg.memory.max_context_tokens - (base.tokens if base else 0)
            ctx = orch.context_builder.build(args.query, max_tokens=budget, base=base)
            if base and base.text:
                print("== memory base (system prompt, cached) ==")
                print(base.text)
                print(f"-- ~{base.tokens} tokens; dropped (did not fit) {base.dropped}\n")
            print("== per-turn block (with the user's message) ==")
            print(ctx.text or "(nothing relevant)")
            print(f"\n-- ~{ctx.token_estimate} tokens; total budget {cfg.memory.max_context_tokens}; "
                  f"included {ctx.included}; dropped {ctx.dropped}")
        elif sub == "validate":
            report = orch.manager.validate_files()
            _print(report)
            return 0 if all(r["ok"] for r in report) else 2
        elif sub == "backup":
            print(orch.manager.snapshot("manual"))
        elif sub == "rebuild":
            from .rebuild import rebuild_memory
            if args.reset and not args.replay:
                print("--reset only makes sense with --replay")
                return 2
            _print(rebuild_memory(orch, replay=args.replay, reset=args.reset))
            if args.replay:
                print("Replay tasks are queued; the running server's worker will process them.")
        elif sub == "changes":
            for ch in orch.db.list_changes(args.limit):
                print(f"#{ch['id']:<5} {ch['status']:<9} {ch['operation'] or '?':<10} "
                      f"{ch['category'] or '?':<11} {ch['entry_key'] or '-':<7} {ch['title'] or ''}"
                      + (f"  -- {ch['status_detail']}" if ch['status_detail'] else ""))
        elif sub == "sessions":
            for row in orch.db.list_summaries(args.limit):
                print(f"== {row['conversation_id']}  (turns 1-{row['covered_turns']}, {row['updated_at']})\n"
                      f"{row['summary']}\n")
        elif sub == "tasks":
            _print({"counts": orch.db.task_counts(), "recent": orch.db.list_tasks(args.limit)})
        elif sub == "consolidate":
            report = asyncio.run(orch.consolidator.run(trigger="cli", dry_run=args.dry_run, force=True))
            for a in report.applied:
                print(f"{'would apply' if report.dry_run else 'applied'}: {a['kind']:<7} {a['category']:<11} "
                      f"{a['title']}  {a['changes']}")
            for r in report.rejected:
                print(f"rejected: {r.get('category', '?'):<11} {r['reason']}")
            print(f"-- jobs {report.jobs}, proposals {report.proposals}, applied {len(report.applied)}, "
                  f"rejected {len(report.rejected)}" + (f", backup {report.backup}" if report.backup else "")
                  + (f", error {report.error}" if report.error else ""))
        elif sub == "review":
            rows = orch.consolidator.review()
            if not rows:
                print(f"Nothing unused for {cfg.consolidation.stale_after_days}+ days.")
            for r in rows:
                print(f"[{r['entry_id']}] {r['title']}  (created {r['created_at'][:10]}, "
                      f"last used {(r['last_used_at'] or 'never')[:10]}, used {r['use_count']}x)")
        elif sub == "reindex":
            if orch.indexer is None:
                print("Embeddings are disabled (embeddings.enabled: false).")
                return 2

            async def go():
                m = await orch.indexer.sync_memory()
                created = orch.indexer.backfill_history_from_logs() if not args.memory_only else 0
                h = 0
                while not args.memory_only:
                    n = await orch.indexer.sync_history()
                    if not n:
                        break
                    h += n
                    print(f"  embedded {h} history chunks...", flush=True)
                return m, created, h
            try:
                m, created, h = asyncio.run(go())
            except Exception as e:
                print(f"Reindex failed: {e}. Is the memory Ollama instance running with {cfg.embeddings.model} pulled?")
                return 1
            print(f"memory entries embedded: {m}; history chunks created: {created}, embedded: {h}")
        elif sub == "restore":
            _print(orch.manager.restore(args.backup_dir))
        return 0
    finally:
        asyncio.run(orch.aclose())


def cmd_eval(cfg, args) -> int:
    from pathlib import Path

    from . import evaluation as ev
    if args.eval_cmd in ("candidates", "accept", "dismiss"):
        from .database import Database
        db = Database(cfg.database_path)
        if args.eval_cmd == "candidates":
            rows = db.candidates(status=None if args.all else "pending")
            if not rows:
                print("No pending candidates. They appear when you correct the model (e.g. 'no, it's X').")
            for r in rows:
                print(f"#{r['id']} [{r['status']}] {r['conversation_id']} turn {r['turn']}")
                print(f"   asked:      {r['question'][:100]!r}")
                print(f"   answered:   {r['wrong_answer'][:100]!r}")
                print(f"   correction: {r['correction'][:100]!r}")
                print(f"   suggested expect_all: {r['suggested_expect'] or '(none)'}")
                print(f"   suggested forbid:     {r.get('suggested_forbid') or '(none)'}")
            return 0
        if args.eval_cmd == "dismiss":
            ok = db.set_candidate_status(args.id, "dismissed")
            print("dismissed" if ok else f"no candidate #{args.id}")
            return 0 if ok else 2
        golden = Path(args.golden) if args.golden else cfg.root_dir / "evals" / "golden.yaml"
        try:
            out = ev.accept_candidate(db, args.id, golden, expect=args.expect, forbid=args.forbid, name=args.name)
        except ValueError as e:
            print(f"Not accepted: {e}")
            return 2
        print(f"Added case {out['case']['name']!r} to {out['file']}")
        return 0

    recorded = ev.load_recorded_sessions(cfg.conversations_dir)
    if args.eval_cmd == "sessions":
        if not recorded:
            print(f"No recorded sessions in {cfg.conversations_dir}")
        for sid, s in recorded.items():
            tools = sum(len(t.tool_events) for t in s.turns) // 2
            print(f"{sid:<40} {len(s.turns):>4} turns  {tools:>4} tool calls  "
                  f"first: {s.turns[0].user[:50]!r}")
        return 0

    evals_dir = cfg.root_dir / "evals"
    variants = ev.load_variants(Path(args.variants) if args.variants else evals_dir / "variants.yaml")
    names = [v.strip() for v in args.variant.split(",") if v.strip()]
    golden_path = Path(args.golden) if args.golden else evals_dir / "golden.yaml"
    golden = ev.load_golden(golden_path) if golden_path.exists() and not args.no_golden else []

    if args.sessions:
        wanted = [x.strip() for x in args.sessions.split(",") if x.strip()]
        missing = [w for w in wanted if w not in recorded]
        if missing:
            print(f"Unknown session(s): {', '.join(missing)} (see `ai eval sessions`)")
            return 2
        sessions = [recorded[w] for w in wanted]
    else:
        pool = [s for s in recorded.values() if len(s.turns) >= args.min_turns]
        sessions = pool[-args.last:] if args.last else []
    if not sessions and not golden:
        print("Nothing to evaluate: no recorded sessions selected and no golden cases "
              f"({golden_path}). See docs/DOCUMENTATION.md §12.")
        return 2

    try:
        httpx.get(_server(cfg) + "/health", timeout=1)
        print("WARNING: the orchestrator server is running. Live traffic to the same Ollama instances "
              "distorts cache and timing measurements; stop it for clean numbers.")
    except httpx.HTTPError:
        pass

    client_system = Path(args.client_system).read_text(encoding="utf-8") if args.client_system \
        else ev.DEFAULT_CLIENT_SYSTEM
    print(f"Variants: {', '.join(names)} | sessions: {len(sessions)} | golden cases: {len(golden)}")
    report = asyncio.run(ev.run_eval(
        cfg, variants=variants, variant_names=names, sessions=sessions, golden=golden,
        max_turns=args.max_turns, memory=args.memory, extract=args.extract,
        think=None if args.think == "default" else args.think == "on", client_system=client_system,
        max_answer_tokens=args.max_answer_tokens or (512 if args.think == "off" else 4096), seed=args.seed,
        repeats=args.repeats))
    print()
    print(Path(report["files"]["markdown"]).read_text(encoding="utf-8"))
    print(f"Report: {report['files']['markdown']}\nJSON:   {report['files']['json']}")
    if report.get("regressions", {}).get("flags"):
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ai", description="Local AI orchestrator administration")
    p.add_argument("--config", help="path to config.yaml (default: config/config.yaml or $AI_CONFIG)")
    sp = p.add_subparsers(dest="cmd", required=True)

    sp.add_parser("serve", help="run the orchestrator server")
    sp.add_parser("ui", help="open the web console in your browser")
    c = sp.add_parser("chat", help="interactive chat through the running server")
    c.add_argument("--conversation-id")
    c.add_argument("-v", "--verbose", action="store_true")
    sp.add_parser("status", help="health of both Ollama instances, GPU residency, queue")
    sp.add_parser("metrics", help="request/memory metrics from the running server")

    m = sp.add_parser("memory", help="memory administration")
    msp = m.add_subparsers(dest="memory_cmd", required=True)
    s = msp.add_parser("show")
    s.add_argument("category", nargs="?", choices=[x.value for x in Category])
    s = msp.add_parser("search")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--all", action="store_true", help="include inactive entries")
    s = msp.add_parser("context", help="show the memory block a prompt would receive")
    s.add_argument("query")
    msp.add_parser("validate")
    msp.add_parser("backup")
    s = msp.add_parser("rebuild")
    s.add_argument("--replay", action="store_true", help="re-run raw history through the memory model")
    s.add_argument("--reset", action="store_true", help="with --replay: start from empty memory")
    s = msp.add_parser("changes")
    s.add_argument("--limit", type=int, default=30)
    s = msp.add_parser("sessions", help="rolling session summaries")
    s.add_argument("--limit", type=int, default=5)
    s = msp.add_parser("tasks")
    s.add_argument("--limit", type=int, default=20)
    s = msp.add_parser("consolidate", help="merge duplicates / tighten long entries now")
    s.add_argument("--dry-run", action="store_true", help="show proposals without applying")
    msp.add_parser("review", help="active entries unused for consolidation.stale_after_days")
    s = msp.add_parser("reindex", help="(re)build embeddings for memory and all recorded history")
    s.add_argument("--memory-only", action="store_true")
    s = msp.add_parser("restore", help="restore memory files from a backups\\... snapshot")
    s.add_argument("backup_dir")

    e = sp.add_parser("eval", help="evaluation harness (docs/DOCUMENTATION.md §12)")
    esp = e.add_subparsers(dest="eval_cmd", required=True)
    esp.add_parser("sessions", help="list recorded sessions available for replay")
    s = esp.add_parser("candidates", help="corrections captured as golden-question candidates")
    s.add_argument("--all", action="store_true", help="include accepted/dismissed")
    s = esp.add_parser("accept", help="add a candidate to evals/golden.yaml")
    s.add_argument("id", type=int)
    s.add_argument("--expect", action="append", help="must appear in the answer (repeatable; /regex/ ok)")
    s.add_argument("--forbid", action="append", help="must NOT appear (repeatable)")
    s.add_argument("--name")
    s.add_argument("--golden", help="golden file (default: evals/golden.yaml)")
    s = esp.add_parser("dismiss", help="discard a candidate")
    s.add_argument("id", type=int)
    s = esp.add_parser("run", help="replay sessions / golden questions under config variants")
    s.add_argument("--variant", default="baseline,full",
                   help="comma-separated; first is the comparison baseline (default: baseline,full)")
    s.add_argument("--variants", help="variants file (default: evals/variants.yaml)")
    s.add_argument("--golden", help="golden questions file (default: evals/golden.yaml)")
    s.add_argument("--no-golden", action="store_true")
    s.add_argument("--sessions", help="comma-separated recorded session ids")
    s.add_argument("--last", type=int, default=3, help="otherwise: the last N recorded sessions (default 3)")
    s.add_argument("--min-turns", type=int, default=4, help="skip shorter sessions (default 4)")
    s.add_argument("--max-turns", type=int, help="replay at most N turns per session")
    s.add_argument("--memory", choices=["current", "empty"], default="current",
                   help="start each variant from a copy of current memory, or from empty memory")
    s.add_argument("--extract", action="store_true", help="also run memory extraction during replay")
    s.add_argument("--think", choices=["off", "on", "default"], default="off")
    s.add_argument("--client-system", help="file with your client's real system prompt (e.g. OpenClaw's)")
    s.add_argument("--max-answer-tokens", type=int,
                   help="default 512 with --think off, 4096 otherwise (thinking counts toward the limit)")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--repeats", type=int, default=1,
                   help="ask each golden question N times (seeds seed..seed+N-1, temperature 0.7)")

    args = p.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(cfg), host=cfg.application.host, port=cfg.application.port,
                    log_level=cfg.application.log_level.lower())
        return 0
    if args.cmd == "ui":
        import webbrowser
        url = _server(cfg) + "/ui"
        try:
            httpx.get(_server(cfg) + "/health", timeout=3)
        except httpx.HTTPError:
            print(f"The orchestrator isn't running at {_server(cfg)}. Start it first (scripts\\start-orchestrator.ps1).")
            return 1
        print(f"Opening {url}")
        webbrowser.open(url)
        return 0
    if args.cmd == "chat":
        return cmd_chat(cfg, args)
    if args.cmd == "status":
        return cmd_http_get(cfg, "/health")
    if args.cmd == "metrics":
        return cmd_http_get(cfg, "/metrics")
    if args.cmd == "memory":
        return cmd_memory(cfg, args)
    if args.cmd == "eval":
        return cmd_eval(cfg, args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
