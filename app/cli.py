"""Administration CLI (spec §42).

    ai serve
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
            ctx = orch.context_builder.build(args.query)
            print(ctx.text or "(nothing relevant)")
            print(f"\n-- ~{ctx.token_estimate} tokens / budget {cfg.memory.max_context_tokens}; "
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
            print("Memory consolidation is Phase 4 and not implemented yet.")
            return 3
        return 0
    finally:
        asyncio.run(orch.aclose())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ai", description="Local AI orchestrator administration")
    p.add_argument("--config", help="path to config.yaml (default: config/config.yaml or $AI_CONFIG)")
    sp = p.add_subparsers(dest="cmd", required=True)

    sp.add_parser("serve", help="run the orchestrator server")
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
    msp.add_parser("consolidate")

    args = p.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(cfg), host=cfg.application.host, port=cfg.application.port,
                    log_level=cfg.application.log_level.lower())
        return 0
    if args.cmd == "chat":
        return cmd_chat(cfg, args)
    if args.cmd == "status":
        return cmd_http_get(cfg, "/health")
    if args.cmd == "metrics":
        return cmd_http_get(cfg, "/metrics")
    if args.cmd == "memory":
        return cmd_memory(cfg, args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
