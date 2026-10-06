import json

import pytest
from fastapi.testclient import TestClient

from app import compression
from app.api import create_app
from app.compression import compress_boundary, result_hash
from app.proxy import stepped_drop
from app.schemas import InteractionTask

BIG = "\n".join(f"line {i}: DEBUG connection pool stats ok idle=3 busy=0" for i in range(300))
BIG2 = BIG.replace("DEBUG", "TRACE")


@pytest.mark.parametrize("turn,expected", [(2, 0), (7, 0), (8, 6), (10, 6), (11, 8), (15, 8),
                                           (16, 14), (21, 14), (22, 20)])
def test_boundary_moves_with_trim_point(turn, expected):
    drop = stepped_drop(turn, 10, 4)
    assert compress_boundary(turn, drop, trim_keep=4, keep_recent=2, step=6, trimming=True) == expected


def test_boundary_without_trimming():
    b = lambda u: compress_boundary(u, 0, trim_keep=4, keep_recent=2, step=6, trimming=False)
    assert [b(u) for u in (3, 7, 8, 13, 14, 20)] == [0, 0, 6, 6, 12, 18]


def msgs_with_tools():
    return [{"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_log"}}]},
            {"role": "tool", "tool_name": "read_log", "content": BIG},
            {"role": "tool", "tool_name": "read_log", "content": "small"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "tool", "tool_name": "read_file", "content": BIG2},
            {"role": "user", "content": "u3"}]


def test_apply_only_old_large_frozen_results():
    m = msgs_with_tools()
    frozen = {result_hash(BIG): "- 300 pool-stat lines, all idle=3 busy=0", result_hash(BIG2): "x"}
    out, n, saved = compression.apply(m, 1, 1, frozen, 50, set())
    assert n == 1 and saved > 1000
    assert out[3]["content"].startswith("[compressed tool output") and "idle=3" in out[3]["content"]
    assert out[4]["content"] == "small" and out[7]["content"] == BIG2   # small / too recent
    assert m[3]["content"] == BIG                                         # input not mutated
    out, n, _ = compression.apply(m, 1, 2, frozen, 50, {"read_file"})
    assert n == 1 and out[7]["content"] == BIG2                          # excluded tool


def test_missing_digest_stays_verbatim():
    out, n, _ = compression.apply(msgs_with_tools(), 1, 2, {"other": "d"}, 50, set())
    assert n == 0


# ---------------------------------------------------------------- end to end
@pytest.fixture
def env(cfg, fakes):
    primary, memory = fakes
    app = create_app(cfg, primary_transport=primary.transport, memory_transport=memory.transport,
                     console_logs=False)
    with TestClient(app) as client:
        yield client, app.state.orch, primary, memory


def session(n_turns, tool_in_turn1=BIG, tool_in_last=None):
    msgs = [{"role": "system", "content": "You are OpenClaw."},
            {"role": "user", "content": "check the pool log"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_log", "arguments": {}}}]},
            {"role": "tool", "tool_name": "read_log", "content": tool_in_turn1},
            {"role": "assistant", "content": "pool looks idle"}]
    for i in range(2, n_turns):
        msgs += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    msgs.append({"role": "user", "content": "and now?"})
    if tool_in_last:
        msgs += [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_log"}}]},
                 {"role": "tool", "tool_name": "read_log", "content": tool_in_last}]
    return {"model": "qwen3:14b", "messages": msgs, "stream": False}


def tool_turn():
    """A turn whose final request carries the tool call + result: the turn being completed."""
    return {"model": "qwen3:14b", "stream": False, "messages": [
        {"role": "system", "content": "You are OpenClaw."},
        {"role": "user", "content": "check the pool log"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_log", "arguments": {}}}]},
        {"role": "tool", "tool_name": "read_log", "content": BIG}]}


def tool_contents(primary):
    return [m["content"] for m in primary.requests[-1]["messages"] if m["role"] == "tool"]


def test_digest_generated_after_turn(env):
    client, orch, primary, memory = env
    memory.reply = "- 300 pool-stat lines, all idle=3 busy=0"
    client.post("/api/chat", json=tool_turn())
    kinds = sorted(t["kind"] for t in orch.db.list_tasks())
    assert "digest" in kinds
    while client.portal.call(orch.worker.process_next):
        pass
    d = orch.db.get_digests([result_hash(BIG)])[result_hash(BIG)]
    assert d["usable"] and d["original_tokens"] > d["digest_tokens"]
    # The same output seen again later is not digested twice.
    assert orch.worker.enqueue_digests(InteractionTask(
        "c", "t", "u", "a", tool_events=[{"type": "tool_result", "tool": "read_log", "result": BIG}])) == 0


def test_long_useless_digest_is_not_used(env):
    client, orch, primary, memory = env
    memory.reply = BIG[:3000]
    client.post("/api/chat", json=tool_turn())
    while client.portal.call(orch.worker.process_next):
        pass
    assert orch.db.get_digests([result_hash(BIG)])[result_hash(BIG)]["usable"] == 0


def test_proxy_compresses_old_results_and_keeps_prefix_stable(env):
    client, orch, primary, _ = env
    orch.config.proxy.trim_mode = "turns"   # this test covers the turn-count schedule
    orch.db.set_digest(result_hash(BIG), "read_log", "- 300 pool lines, idle=3", 2000, 10, True)

    client.post("/api/chat", json=session(7))                       # boundary 0
    assert tool_contents(primary) == [BIG]

    client.post("/api/chat", json=session(8, tool_in_last=BIG2))     # boundary 6
    s8 = primary.requests[-1]["messages"]
    tools = tool_contents(primary)
    assert tools[0].startswith("[compressed tool output") and tools[1] == BIG2  # current turn verbatim
    m = client.get("/metrics").json()["recent_requests"][-1]
    assert m["tool_results_compressed"] == 1 and m["tool_tokens_saved"] > 1000

    # A new digest lands mid-window: the frozen set must not change until the boundary moves.
    orch.db.set_digest(result_hash(BIG2), "read_log", "- trace lines", 2000, 5, True)
    client.post("/api/chat", json=session(9))
    s9 = primary.requests[-1]["messages"]
    assert s9[:5] == s8[:5]                                         # identical prefix

    # Disabled -> untouched.
    orch.config.compression.enabled = False
    client.post("/api/chat", json=session(30))
    assert BIG in tool_contents(primary) or not tool_contents(primary)
