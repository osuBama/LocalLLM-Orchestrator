import asyncio
import json

import pytest

from app import evaluation as ev
from app.config import OllamaEndpoint
from app.database import Database
from app.eval_generate import cloze_question, generate, leaks, mine, phrase, session_end_plus, value_kind
from app.ollama_client import OllamaClient
from app.schemas import Category


def session(n=12, sid="s1"):
    turns = [ev.EvalTurn("The memory instance listens on port 11435 now.", "Noted.", [], "2026-02-01T10:00:00+00:00")]
    turns.append(ev.EvalTurn("Check the log", "Done.", [
        {"type": "tool_call", "tool": "read_log", "arguments": {}},
        {"type": "tool_result", "tool": "read_log", "result": "INFO start\nERROR db refused host=db-02 code E4471\nINFO retry"}],
        "2026-02-01T10:01:00+00:00"))
    turns.append(ev.EvalTurn("We use qwen3:14b and qwen3:14b only, version 2.3.1 of the gateway.", "OK.", [],
                             "2026-02-01T10:02:00+00:00"))
    turns.append(ev.EvalTurn("Again: qwen3:14b.", "Yes.", [], "2026-02-01T10:03:00+00:00"))
    for i in range(4, n):
        turns.append(ev.EvalTurn(f"small talk {i}", f"reply {i}", [], f"2026-02-01T10:{i:02d}:00+00:00"))
    return ev.EvalSession(sid, turns)


def test_mining_rules():
    needles = {n.value.lower(): n for n in mine(session(), gap=6)}
    assert "11435" in needles and needles["11435"].upto_turn == 6 and needles["11435"].where == "user"
    tool = [n for n in needles.values() if n.where == "tool output"]
    assert len(tool) == 1 and tool[0].value.lower() in ("db-02", "e4471")    # one per source turn
    assert "qwen3:14b" not in needles               # mentioned 3 times: not a needle
    assert not any(v in needles for v in ("4", "10", "12"))                     # trivial numbers
    assert mine(session(n=5), gap=6) == []          # too short for the gap


def test_value_mentioned_again_before_the_question_is_skipped():
    s = session()
    s.turns[3] = ev.EvalTurn("Reminder: port 11435.", "ok", [], "2026-02-01T10:03:00+00:00")
    assert "11435" not in {n.value for n in mine(s, gap=6)}


def test_cloze_and_leaks():
    n = mine(session(), gap=6)[0]
    q = cloze_question(n)
    assert "____" in q and not leaks(q, n.value) and q.endswith("?")
    assert leaks("Is it port 11435?", "11435") and leaks("version 2-3-1?", "2.3.1")
    assert value_kind("11435") == "port" and value_kind("E4471") == "code" and value_kind("qwen3:14b") == "model or tag"
    assert value_kind("/etc/hosts") == "path" and value_kind("10.0.0.5") == "IP address"
    assert value_kind("db-02") == "name"


def test_phrasing_validated_with_fallback(fakes):
    _, memory = fakes
    client = OllamaClient(OllamaEndpoint(base_url="http://memory.test", model="m"), transport=memory.transport)
    n = mine(session(), gap=6)[0]
    memory.memory_json = {"usable": True, "question": "Which port did we move the memory instance to?"}
    assert asyncio.run(phrase(client, n)) == ("Which port did we move the memory instance to?", "model")
    memory.memory_json = {"usable": True, "question": f"Is it {n.value}?"}            # leaks the answer
    assert asyncio.run(phrase(client, n))[1] == "fill-in-the-blank"
    memory.memory_json = {"usable": False, "question": "?"}
    assert asyncio.run(phrase(client, n))[1] == "fill-in-the-blank"
    memory.memory_json = "not json"
    assert asyncio.run(phrase(client, n))[1] == "fill-in-the-blank"
    assert asyncio.run(phrase(None, n, "new-session"))[0].startswith("In an earlier conversation")


def test_generate_both_modes_and_dedupe(tmp_path):
    db = Database(tmp_path / "m.db")
    out = asyncio.run(generate(db, [session()], gap=6, mode="both"))
    kinds = {c["kind"] for c in out["created"]}
    assert kinds == {"later", "new-session"}
    ns = next(c for c in out["created"] if c["kind"] == "new-session")
    assert ns["as_of"] == "2026-02-01T11:11:00+00:00"                 # last turn + 1 hour
    again = asyncio.run(generate(db, [session()], gap=6, mode="both"))
    assert again["created"] == [] and again["skipped_duplicates"] == len(out["created"])
    cands = db.candidates()
    assert all(c["kind"] == "generated" for c in cands)


def test_accept_builds_both_case_shapes(tmp_path):
    db = Database(tmp_path / "m.db")
    out = asyncio.run(generate(db, [session()], gap=6, mode="both"))
    later = next(c for c in out["created"] if c["kind"] == "later" and c["expect"] == "11435")
    fresh = next(c for c in out["created"] if c["kind"] == "new-session" and c["expect"] == "11435")
    g = tmp_path / "golden.yaml"
    a = ev.accept_candidate(db, later["id"], g)["case"]
    b = ev.accept_candidate(db, fresh["id"], g)["case"]
    assert a["session"] == "s1" and a["upto_turn"] == 6 and a["expect_all"] == ["11435"]
    assert b["as_of"] == fresh["as_of"] and "session" not in b and "upto_turn" not in b
    names = [c.name for c in ev.load_golden(g)]
    assert names == [f"generated-{later['id']}", f"generated-{fresh['id']}"]
    assert "# generated #" in g.read_text(encoding="utf-8")


def test_new_session_case_is_answerable_only_from_memory(cfg, fakes):
    """End to end: the question arrives in a fresh conversation; the fact must come from memory."""
    primary, memory = fakes
    from app.markdown_store import MarkdownDocument, MarkdownStore
    from app.schemas import MemoryEntry
    st = MarkdownStore(cfg.memory_dir, Category.environment, history_dir=cfg.history_dir)
    st.path.parent.mkdir(parents=True, exist_ok=True)
    learned = "2026-02-01T10:30:00+00:00"                            # extracted after the session, before as_of
    st.path.write_text(st.render(MarkdownDocument("ENVIRONMENT", [], [
        MemoryEntry("E-001", Category.environment, "Memory instance", "Listens on port 11435.", True, learned, learned)])),
        encoding="utf-8")
    case = ev.GoldenCase(name="fresh", as_of="2026-02-01T11:11:00+00:00",
                         question="Which port is the memory instance on?", expect_all=["11435"])
    primary.reply = "It is 11435."
    rep = asyncio.run(ev.run_eval(cfg, variants=ev.load_variants(None), variant_names=["baseline", "full"],
                                  sessions=[], golden=[case], primary_transport=primary.transport,
                                  memory_transport=memory.transport, progress=lambda *_: None))
    d = {g["variant"]: g for g in rep["golden"]}
    assert d["full"]["diagnosis"] == "answered_from_prompt" and "memory_base" in d["full"]["evidence_in"]
    assert d["baseline"]["diagnosis"] == "answered_without_evidence"   # baseline has no memory: a guess
    # And before the fact was learned, memory must not know it.
    early = ev.GoldenCase(name="early", as_of="2026-02-01T10:00:00+00:00",
                          question="Which port is the memory instance on?", expect_all=["11435"])
    rep = asyncio.run(ev.run_eval(cfg, variants=ev.load_variants(None), variant_names=["full"], sessions=[],
                                  golden=[early], primary_transport=primary.transport,
                                  memory_transport=memory.transport, progress=lambda *_: None))
    assert rep["golden"][0]["diagnosis"] == "answered_without_evidence"


def test_golden_case_validation():
    with pytest.raises(Exception):
        ev.GoldenCase(name="x", session="s", as_of="2026-01-01T00:00:00+00:00", question="q", expect_all=["a"])


def test_cli_generate(cfg, tmp_path, capsys):
    import yaml
    from app import cli
    cfg.conversations_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.conversations_dir / "2026-02-01.jsonl", "w") as f:
        for t in session().turns:
            f.write(json.dumps({"conversation_id": "s1", "type": "message", "role": "user", "content": t.user,
                                "timestamp": t.timestamp}) + "\n")
            for e in t.tool_events:
                f.write(json.dumps({"conversation_id": "s1", **({"type": "tool_result", "tool": e["tool"], "result": e["result"]}
                                    if e["type"] == "tool_result" else {"type": "tool_call", "tool": e["tool"], "arguments": {}})}) + "\n")
            f.write(json.dumps({"conversation_id": "s1", "type": "message", "role": "assistant",
                                "content": t.assistant, "timestamp": t.timestamp}) + "\n")
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(json.loads(cfg.model_dump_json(exclude={"source_path"}))))
    assert cli.main(["--config", str(p), "eval", "generate", "--no-model", "--accept-all",
                     "--golden", str(tmp_path / "g.yaml")]) == 0
    out = capsys.readouterr().out
    assert "fill-in-the-blank" in out and "Accepted all" in out
    assert len(ev.load_golden(tmp_path / "g.yaml")) >= 4
