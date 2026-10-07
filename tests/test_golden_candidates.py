import pytest
from fastapi.testclient import TestClient

from app import evaluation as ev
from app.api import create_app
from app.cli import main as cli_main
from app.conversation_logger import ConversationLogger


@pytest.fixture
def env(cfg, fakes):
    primary, memory = fakes
    app = create_app(cfg, primary_transport=primary.transport, memory_transport=memory.transport,
                     console_logs=False)
    with TestClient(app) as client:
        yield client, app.state.orch, primary


def convo(*turns):
    msgs = []
    for u, a in turns:
        msgs.append({"role": "user", "content": u})
        if a is not None:
            msgs.append({"role": "assistant", "content": a})
    return {"model": "qwen3:14b", "stream": False, "messages": msgs}


def test_correction_captured_with_suggestion(env):
    client, orch, primary = env
    body = convo(("which port is the memory instance on?", "It runs on port 11434."),
                 ("no, that's wrong: it's 11435", None))
    client.post("/api/chat", json=body, headers={"X-Conversation-Id": "c1"})
    [c] = orch.db.candidates()
    assert c["question"] == "which port is the memory instance on?" and c["turn"] == 2
    assert "11434" in c["wrong_answer"] and c["suggested_expect"] == ["11435"]


def test_no_candidate_without_correction_or_history(env):
    client, orch, primary = env
    client.post("/api/chat", json=convo(("thanks, that worked", None)))
    client.post("/api/chat", json=convo(("q", "a"), ("great, next step?", None)))
    assert orch.db.candidates() == []


def test_chat_endpoint_captures_too(env):
    client, orch, primary = env
    primary.reply = "The default model is qwen3:8b."
    client.post("/chat", json={"conversation_id": "k", "message": "what's the primary model?"})
    client.post("/chat", json={"conversation_id": "k", "message": "actually, it's qwen3:14b"})
    [c] = orch.db.candidates()
    assert c["suggested_expect"] == ["qwen3:14b"]


def test_accept_appends_loadable_case_and_keeps_comments(env, tmp_path):
    client, orch, primary = env
    client.post("/api/chat", json=convo(("which port?", "11434"), ("no, it's 11435", None)),
                headers={"X-Conversation-Id": "c9"})
    golden = tmp_path / "golden.yaml"
    golden.write_text(open("examples/golden.example.yaml", encoding="utf-8").read(), encoding="utf-8")
    cid = orch.db.candidates()[0]["id"]
    out = ev.accept_candidate(orch.db, cid, golden, forbid=["/\\b11434\\b/"])
    cases = ev.load_golden(golden)
    new = cases[-1]
    assert new.name == f"correction-{cid}" and new.session == "c9" and new.upto_turn == 0
    assert new.expect_all == ["11435"] and new.forbid == ["/\\b11434\\b/"]
    assert "# Copy to <install>" in golden.read_text(encoding="utf-8")   # comments kept
    assert orch.db.candidates() == [] and orch.db.candidates(status="accepted")[0]["case_name"] == new.name
    with pytest.raises(ValueError):
        ev.accept_candidate(orch.db, cid, golden)                          # already accepted


def test_accept_requires_an_expectation(env, tmp_path):
    client, orch, primary = env
    client.post("/api/chat", json=convo(("summarize it", "It is about cats."), ("no, that's wrong, it is about dogs", None)))
    cid = orch.db.candidates()[0]["id"]
    assert orch.db.candidates()[0]["suggested_expect"] == []
    with pytest.raises(ValueError):
        ev.accept_candidate(orch.db, cid, tmp_path / "g.yaml")
    out = ev.accept_candidate(orch.db, cid, tmp_path / "g.yaml", expect=["dogs"])
    assert ev.load_golden(tmp_path / "g.yaml")[0].expect_all == ["dogs"]


def test_accepted_case_replays_through_the_harness(env, cfg, tmp_path):
    client, orch, primary = env
    lg = ConversationLogger(cfg.conversations_dir)
    lg.message("s1", "user", "which port?")
    lg.message("s1", "assistant", "11434")
    lg.message("s1", "user", "no, it's 11435")
    lg.message("s1", "assistant", "Right, 11435.")
    cid = orch.db.add_candidate("s1", 2, "which port?", "11434", "no, it's 11435", ["11435"])
    ev.accept_candidate(orch.db, cid, tmp_path / "g.yaml")
    case = ev.load_golden(tmp_path / "g.yaml")[0]
    sess = ev.load_recorded_sessions(cfg.conversations_dir)[case.session]
    assert len(sess.turns[: case.upto_turn]) == 0 and case.question == sess.turns[0].user


def test_cli_candidates_accept_dismiss(env, cfg, tmp_path, capsys):
    import json as _j
    import yaml as _y
    client, orch, primary = env
    orch.db.add_candidate("s", 3, "q?", "wrong", "no, it's 42", ["42"])
    orch.db.add_candidate("s", 5, "q2?", "wrong", "no, it's 7", ["7"])
    p = tmp_path / "c.yaml"
    p.write_text(_y.safe_dump(_j.loads(cfg.model_dump_json(exclude={"source_path"}))))
    assert cli_main(["--config", str(p), "eval", "candidates"]) == 0
    assert "no, it's 42" in capsys.readouterr().out
    assert cli_main(["--config", str(p), "eval", "accept", "1", "--golden", str(tmp_path / "g.yaml")]) == 0
    assert cli_main(["--config", str(p), "eval", "dismiss", "2"]) == 0
    assert [c["status"] for c in orch.db.candidates(status=None)] == ["dismissed", "accepted"]
    assert client.get("/eval/candidates", params={"all": True}).json()["candidates"][0]["status"] == "dismissed"


def test_negated_values_become_forbid_not_expect(env):
    client, orch, primary = env
    body = convo(("which port is the memory instance on?", "It listens on 11435 most likely."),
                 ("no, it's 11436, not 11435", None))
    client.post("/api/chat", json=body, headers={"X-Conversation-Id": "neg"})
    [c] = orch.db.candidates()
    assert c["suggested_expect"] == ["11436"]
    assert c["suggested_forbid"] == ["/\\b11435\\b/"]
    case = ev.accept_candidate(orch.db, c["id"], orch.config.root_dir / "g.yaml")["case"]
    ok, _ = ev.grade(ev.GoldenCase(**case), "It is on port 11436.")
    bad, problems = ev.grade(ev.GoldenCase(**case), "Port 11435.")
    assert ok and not bad and any("forbidden" in p for p in problems)
