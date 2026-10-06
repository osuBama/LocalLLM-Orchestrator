import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.compression import result_hash

BIG = "\n".join(f"line {i}: pool stats idle=3 busy=0 ok" for i in range(120))   # ~1.2k tokens


@pytest.fixture
def env(cfg, fakes):
    primary, memory = fakes
    cfg.ollama.primary.num_ctx = 8000
    cfg.memory.max_context_tokens = 1000
    cfg.stable_memory.max_tokens = 400
    cfg.proxy.reply_reserve_tokens = 500
    app = create_app(cfg, primary_transport=primary.transport, memory_transport=memory.transport,
                     console_logs=False)
    with TestClient(app) as client:
        yield client, app.state.orch, primary


def session(n, tool_every=False, tools=None, big=BIG):
    msgs = [{"role": "system", "content": "You are OpenClaw."}]
    for i in range(1, n):
        msgs.append({"role": "user", "content": f"q{i} " + "words " * 30})
        if tool_every:
            msgs += [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_log"}}]},
                     {"role": "tool", "tool_name": "read_log", "content": big.replace("line", f"t{i}")}]
        msgs.append({"role": "assistant", "content": f"a{i} " + "words " * 30})
    msgs.append({"role": "user", "content": "and now?"})
    body = {"model": "qwen3:14b", "messages": msgs, "stream": False}
    if tools:
        body["tools"] = tools
    return body


def last(client):
    return client.get("/metrics").json()["recent_requests"][-1]


def test_long_session_that_fits_is_never_touched(env):
    client, orch, primary = env
    client.post("/api/chat", json=session(40))          # 40 short turns still fit 8k
    m = last(client)
    assert m["user_turns_dropped"] == 0 and m["tool_results_compressed"] == 0
    assert not m["context_over_budget"]
    assert sum(1 for x in primary.requests[-1]["messages"] if x["role"] == "user") == 40


def test_compresses_before_cutting(env):
    client, orch, primary = env
    for i in range(1, 8):
        orch.db.set_digest(result_hash(BIG.replace("line", f"t{i}")), "read_log", "- pool stats", 1200, 5, True)
    for cid in ("s-a", "s-a2"):
        orch.db.set_summary(cid, "Summary of everything.", 50)
    orch.config.compression.enabled = False
    client.post("/api/chat", json=session(8, tool_every=True), headers={"X-Conversation-Id": "s-a2"})
    cut_without = last(client)["user_turns_dropped"]
    orch.config.compression.enabled = True
    client.post("/api/chat", json=session(8, tool_every=True), headers={"X-Conversation-Id": "s-a"})
    m = last(client)
    assert m["tool_results_compressed"] > 0
    assert m["user_turns_dropped"] < cut_without          # compression kept more turns verbatim


def test_cuts_to_target_then_holds_still(env):
    client, orch, primary = env
    medium = "\n".join(f"line {i}: pool stats idle=3 busy=0 ok" for i in range(50))   # ~530 tokens/turn
    orch.db.set_summary("s-b", "Long session about pool stats.", 200)
    sizes, drops = [], []
    for n in range(8, 26):
        client.post("/api/chat", json=session(n, tool_every=True, big=medium), headers={"X-Conversation-Id": "s-b"})
        m = last(client)
        sizes.append(m["est_prompt_tokens"]); drops.append(m["user_turns_dropped"])
        assert not m["context_over_budget"]
    assert max(sizes) <= 8000 and drops[-1] > 0
    # The cut point only moves occasionally (hysteresis), not every turn.
    moves = sum(1 for a, b in zip(drops, drops[1:]) if a != b)
    assert moves <= len(drops) // 3


def test_never_cuts_what_summary_does_not_cover(env):
    client, orch, primary = env
    client.post("/api/chat", json=session(30, tool_every=True), headers={"X-Conversation-Id": "s-c"})
    m = last(client)
    assert m["user_turns_dropped"] == 0 and m["context_over_budget"]   # flagged, not silently cut
    orch.db.set_summary("s-c", "Summary.", 10)
    client.post("/api/chat", json=session(31, tool_every=True), headers={"X-Conversation-Id": "s-c"})
    assert 0 < last(client)["user_turns_dropped"] <= 10


def test_tool_schemas_count_toward_size(env):
    client, orch, primary = env
    huge_tools = [{"type": "function", "function": {"name": f"tool_{i}", "description": "x " * 300}}
                  for i in range(20)]
    client.post("/api/chat", json=session(5), headers={"X-Conversation-Id": "s-d"})
    small = last(client)["est_prompt_tokens"]
    client.post("/api/chat", json=session(5, tools=huge_tools), headers={"X-Conversation-Id": "s-e"})
    assert last(client)["est_prompt_tokens"] > small + 3000
