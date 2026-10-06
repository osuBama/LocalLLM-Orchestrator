import asyncio
import json

import pytest
import yaml

from app import evaluation as ev
from app.conversation_logger import ConversationLogger
from app.schemas import Category
from app.markdown_store import all_stores


def record_session(cfg, cid, n, tool_turn=None, tools_every_turn=False):
    lg = ConversationLogger(cfg.conversations_dir)
    for i in range(1, n + 1):
        lg.message(cid, "user", f"question {i} about the MCP server " + "context " * 40)
        if i == tool_turn or tools_every_turn:
            lg.tool_call(cid, "read_log", {"path": "x.log"})
            lg.tool_result(cid, "read_log", "\n".join(f"log line {k} pool idle=3" for k in range(300)))
        lg.message(cid, "assistant", f"answer {i} " + "detail " * 40)


def run(cfg, fakes, **kw):
    primary, memory = fakes
    primary.simulate_cache = True
    kw.setdefault("variants", ev.load_variants(None))
    kw.setdefault("variant_names", ["baseline", "full"])
    kw.setdefault("golden", [])
    return asyncio.run(ev.run_eval(cfg, primary_transport=primary.transport,
                                   memory_transport=memory.transport, progress=lambda *_: None, **kw))


def test_load_recorded_sessions(cfg):
    record_session(cfg, "s1", 3, tool_turn=2)
    s = ev.load_recorded_sessions(cfg.conversations_dir)["s1"]
    assert len(s.turns) == 3 and len(s.turns[1].tool_events) == 2


def test_variant_config_overrides_and_forced_isolation(cfg, tmp_path):
    v = ev.variant_config(cfg, {"compression.enabled": False}, tmp_path / "ws")
    assert v.compression.enabled is False and v.memory.worker_enabled is False
    assert v.memory_dir == tmp_path / "ws" / "memory" and v.memory.create_backups is False
    with pytest.raises(ValueError):
        ev.variant_config(cfg, {"compression.nonsense": 1}, tmp_path / "ws")
    with pytest.raises(Exception):  # still validated like config.yaml
        ev.variant_config(cfg, {"proxy.trim_trigger_user_turns": 2}, tmp_path / "ws")


def test_grading():
    c = ev.GoldenCase(name="x", question="q", expect_all=["11435"], expect_any=["memory", "helper"],
                      forbid=["/port\\s+11434/"])
    assert ev.grade(c, "The memory instance uses 11435.") == (True, [])
    ok, problems = ev.grade(c, "It is on port 11434")
    assert not ok and len(problems) == 3


def test_golden_file_validation(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump({"cases": [{"name": "a", "question": "q"}]}))
    with pytest.raises(Exception):
        ev.load_golden(p)                       # no expectations
    p.write_text(open("examples/golden.example.yaml", encoding="utf-8").read())
    assert [c.name for c in ev.load_golden(p)] == ["port-from-early-turn", "detail-in-old-tool-output"]


def test_replay_compares_variants_and_never_touches_real_memory(cfg, fakes):
    stores = all_stores(cfg.memory_dir)
    stores[Category.constraint].add("No cloud", "No cloud inference; run everything locally.")
    real_before = stores[Category.constraint].path.read_text(encoding="utf-8")
    record_session(cfg, "s1", 24, tools_every_turn=True)            # ~45k tokens of history
    sessions = list(ev.load_recorded_sessions(cfg.conversations_dir).values())
    rep = run(cfg, fakes, sessions=sessions)
    b, f = rep["variants"]["baseline"], rep["variants"]["full"]
    assert b["turns"] == f["turns"] == 24 and not b["errors"] and not f["errors"]
    assert f["turns_trimmed"] > 0 and f["tool_tokens_saved"] > 0       # features actually exercised
    assert b["turns_trimmed"] == 0 and b["avg_memory_tokens"] == 0
    # The real point of trimming + compression: baseline outgrows the 16k window, full doesn't.
    assert b["turns_over_ctx"] > 5 and f["turns_over_ctx"] == 0
    assert f["max_context_tokens"] < 16384 < b["max_context_tokens"]
    assert f["est_cache_hit"] is not None and b["est_cache_hit"] is not None
    # Isolation: real memory unchanged, real DB untouched by the eval.
    assert stores[Category.constraint].path.read_text(encoding="utf-8") == real_before
    assert not cfg.database_path.exists()
    # Reports written.
    md = open(rep["files"]["markdown"], encoding="utf-8").read()
    assert "| full |" in md and "processed prompt tokens" in md
    assert json.load(open(rep["files"]["json"], encoding="utf-8"))["variants"]["full"]["turns"] == 24


def test_summaries_are_built_from_recorded_answers(cfg, fakes):
    record_session(cfg, "s1", 3)
    primary, memory = fakes
    memory.reply = "summary"
    run(cfg, fakes, sessions=list(ev.load_recorded_sessions(cfg.conversations_dir).values()),
        variant_names=["full"])
    summary_prompts = [r["messages"][1]["content"] for r in memory.requests
                       if r.get("format") is None and len(r["messages"]) > 1
                       and "LATEST TURN" in r["messages"][1]["content"]]
    assert summary_prompts and "answer 1 detail" in summary_prompts[0]


def test_golden_cases_run_and_grade(cfg, fakes):
    primary, memory = fakes
    primary.reply = "The memory instance uses port 11435."
    cases = ev.load_golden("examples/golden.example.yaml")
    rep = run(cfg, fakes, sessions=[], golden=cases)
    results = {(g["variant"], g["case"]): g for g in rep["golden"]}
    assert results[("full", "port-from-early-turn")]["passed"]
    assert not results[("full", "detail-in-old-tool-output")]["passed"]
    assert rep["variants"]["full"]["golden_total"] == 2
    q = primary.requests[-1]
    assert q["options"]["temperature"] == 0 and q["options"]["seed"] == 42 and q["think"] is False


def test_run_marker_isolates_cache_between_variants(cfg, fakes):
    record_session(cfg, "s1", 2)
    primary, _ = fakes
    run(cfg, fakes, sessions=list(ev.load_recorded_sessions(cfg.conversations_dir).values()))
    systems = {r["messages"][0]["content"].split("\n")[0] for r in primary.requests
               if r["messages"][0]["role"] == "system"}
    assert len([s for s in systems if s.startswith("[eval-run")]) == 2


def test_unknown_variant_rejected(cfg, fakes):
    with pytest.raises(ValueError):
        run(cfg, fakes, sessions=[], variant_names=["nope"])


def test_cli_sessions_and_nothing_to_do(cfg, tmp_path, capsys, monkeypatch):
    import yaml as _y
    from app import cli
    p = tmp_path / "c.yaml"
    p.write_text(_y.safe_dump(json.loads(cfg.model_dump_json(exclude={"source_path"}))))
    record_session(cfg, "s1", 2)
    assert cli.main(["--config", str(p), "eval", "sessions"]) == 0
    assert "s1" in capsys.readouterr().out
    assert cli.main(["--config", str(p), "eval", "run", "--min-turns", "10"]) == 2
