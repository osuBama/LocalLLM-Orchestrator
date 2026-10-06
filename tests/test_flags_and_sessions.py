import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.flags import FlagStripper, strip_flags
from app.proxy import drop_user_turns, stepped_drop
from app.schemas import Category, InteractionTask

FLAG = '<memory_flag category="lesson">MCP route is /mcp, not /sse</memory_flag>'


# ------------------------------------------------------------------ flags
def test_flag_split_at_every_character():
    text = "Fixed: the route was wrong.\n\n" + FLAG
    s = FlagStripper()
    out = "".join(s.feed(ch) for ch in text) + s.finish()
    assert "memory_flag" not in out and out.startswith("Fixed: the route was wrong.")
    assert [f.to_dict() for f in s.flags] == [{"category": "lesson", "text": "MCP route is /mcp, not /sse"}]


def test_lookalikes_and_partial_prefix_released():
    out, flags = strip_flags("compare <memo and <memory_flagged> and a < b")
    assert out == "compare <memo and <memory_flagged> and a < b" and flags == []


def test_unclosed_flag_at_end_is_captured():
    out, flags = strip_flags('Done.\n<memory_flag category="state">port 11435 is the memory GPU')
    assert out == "Done." and flags[0].category == "state"


def test_flag_limits_and_bad_category():
    text = "ok" + "".join(f'<memory_flag category="{c}">fact {i}</memory_flag>'
                          for i, c in enumerate(["lesson", "nonsense", "state", "decision"]))
    out, flags = strip_flags(text, max_flags=3)
    assert out == "ok" and len(flags) == 3 and flags[1].category is None


# --------------------------------------------------------------- trimming
@pytest.mark.parametrize("turns,expected", [(1, 0), (10, 0), (11, 6), (15, 6), (16, 12), (21, 12), (22, 18)])
def test_stepped_drop(turns, expected):
    assert stepped_drop(turns, 10, 4) == expected
    assert stepped_drop(turns, 0, 4) == 0


def test_drop_user_turns_keeps_system_and_tool_chains():
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "u1"}, {"role": "assistant", "content": "", "tool_calls": [{}]},
            {"role": "tool", "content": "r1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"}]
    out = drop_user_turns(msgs, 1)
    assert [m["content"] for m in out] == ["s", "u2", "a2", "u3"]


# ------------------------------------------------------------- end to end
@pytest.fixture
def env(cfg, fakes):
    primary, memory = fakes
    app = create_app(cfg, primary_transport=primary.transport, memory_transport=memory.transport,
                     console_logs=False)
    with TestClient(app) as client:
        yield client, app.state.orch, primary, memory


def history(n_turns, last="What next?"):
    msgs = [{"role": "system", "content": "You are OpenClaw."}]
    for i in range(1, n_turns):
        msgs += [{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": f"answer {i}"}]
    msgs.append({"role": "user", "content": last})
    return {"model": "qwen3:14b", "messages": msgs}


def test_stream_flags_hidden_from_client_and_force_extraction(env):
    client, orch, primary, memory = env
    primary.reply = "Route fixed. " + FLAG   # split into words by the fake -> tag spans chunks
    body = history(1, last="ok")             # trivial message: heuristics alone would skip it
    with client.stream("POST", "/api/chat", json=body) as r:
        lines = [json.loads(l) for l in r.iter_lines() if l]
    seen = "".join(l["message"]["content"] for l in lines)
    assert "memory_flag" not in seen and "Route fixed." in seen and lines[-1]["done"]
    tasks = {t["kind"]: t for t in orch.db.list_tasks()}
    assert tasks["extract"]["status"] == "pending"         # forced by the flag
    rec = [r for r in orch.conv_log.iter_records() if r.get("role") == "assistant"][0]
    assert rec["memory_flags"][0]["text"] == "MCP route is /mcp, not /sse"
    assert "memory_flag" not in rec["content"]
    # The extractor sees the flag as a hint.
    memory.memory_json = {"changes": []}
    while client.portal.call(orch.worker.process_next):
        pass
    extract_req = [r for r in memory.requests if r.get("format")][0]
    assert "MCP route is /mcp, not /sse" in extract_req["messages"][1]["content"]


def test_non_stream_flags_stripped(env):
    client, _, primary, _ = env
    primary.reply = "Done.\n" + FLAG
    data = client.post("/api/chat", json={**history(1), "stream": False}).json()
    assert data["message"]["content"] == "Done."


def test_primary_prompt_includes_flag_instruction(env):
    client, _, primary, _ = env
    client.post("/api/chat", json={**history(1), "stream": False})
    assert "<memory_flag" in primary.requests[-1]["messages"][0]["content"]


def test_summary_task_priority_and_staleness(env):
    client, orch, primary, memory = env
    memory.reply = "User is debugging the MCP route."
    t = lambda n: InteractionTask("c9", "now", f"the MCP failed {n}", "fixed it", turn_number=n)
    orch.worker.enqueue(t(1))            # extract first...
    orch.worker.enqueue_summary(t(1))    # ...but summary has priority
    client.portal.call(orch.worker.process_next)
    assert orch.db.get_summary("c9")["covered_turns"] == 1
    assert [x["kind"] for x in orch.db.list_tasks(status="pending")] == ["extract"]
    orch.db.set_summary("c9", "newer", 5)
    assert client.portal.call(orch.worker.process_summary, t(3)) is False
    assert orch.db.get_summary("c9")["summary"] == "newer"


def test_trimming_with_summary_and_stable_prefix(env):
    client, orch, primary, _ = env
    body11 = history(11)
    cid = None
    # No summary yet -> nothing may be cut (trim_requires_summary).
    client.post("/api/chat", json={**body11, "stream": False})
    sent = primary.requests[-1]["messages"]
    assert sum(m["role"] == "user" for m in sent) == 11
    cid = orch.db.list_tasks()[0]["conversation_id"]
    orch.db.set_summary(cid, "User asked ten questions about the MCP setup.", 11)

    client.post("/api/chat", json={**history(12), "stream": False})
    s12 = primary.requests[-1]["messages"]
    assert sum(m["role"] == "user" for m in s12) == 6          # turns 7-12 kept
    assert "SESSION SO FAR" in s12[-1]["content"]
    assert "ten questions about the MCP" in s12[-1]["content"]
    assert s12[1]["content"] == "question 7"

    client.post("/api/chat", json={**history(13), "stream": False})
    s13 = primary.requests[-1]["messages"]
    # Same cut point -> identical prefix up to the previous turn (prompt cache hits).
    assert s13[:len(s12) - 1] == s12[:-1]


def test_chat_endpoint_uses_summary(env):
    client, orch, primary, _ = env
    r = client.post("/chat", json={"conversation_id": "abc", "message": "hello, the MCP failed"}).json()
    assert r["turn"] == 1
    orch.db.set_summary("abc", "Session about the MCP 404.", 1)
    client.post("/chat", json={"conversation_id": "abc", "message": "and now?"})
    assert "Session about the MCP 404." in primary.requests[-1]["messages"][-1]["content"]
