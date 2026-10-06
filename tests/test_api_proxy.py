import json

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.schemas import Category


@pytest.fixture
def env(cfg, fakes):
    primary, memory = fakes
    app = create_app(cfg, primary_transport=primary.transport, memory_transport=memory.transport,
                     console_logs=False)
    orch = app.state.orch
    orch.manager.stores[Category.constraint].add("No cloud", "No cloud inference; run everything locally.")
    orch.manager.stores[Category.lesson].add("MCP 404", "When MCP returns 404, verify the route path first.")
    with TestClient(app) as client:
        yield client, orch, primary, memory


def process_all(client, orch):
    # Drive the worker inside the app's event loop.
    while client.portal.call(orch.worker.process_next):
        pass


def test_chat_injects_memory_logs_and_queues(env):
    client, orch, primary, _ = env
    r = client.post("/chat", json={"message": "MCP gives 404 again, why?"})
    assert r.status_code == 200
    data = r.json()
    assert data["response"] == "fake answer" and data["memory_update_queued"]
    sent = primary.requests[-1]["messages"]
    assert sent[0]["role"] == "system" and "primary problem-solving model" in sent[0]["content"]
    user = sent[-1]["content"]
    assert "<PROJECT_MEMORY>" in user and "[C-001]" in user and "[L-001]" in user
    assert "<USER_REQUEST>\nMCP gives 404 again, why?\n</USER_REQUEST>" in user
    recs = list(orch.conv_log.iter_records())
    assert [r["role"] for r in recs] == ["user", "assistant"]


def openclaw_body(stream=True, extra=None):
    msgs = [{"role": "system", "content": "You are OpenClaw."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "The MCP call returns 404, what now?"}]
    return {"model": "qwen3:14b", "messages": msgs + (extra or []), "stream": stream,
            "tools": [{"type": "function", "function": {"name": "http_get"}}]}


def test_proxy_streaming_relay_and_injection(env):
    client, orch, primary, _ = env
    primary.reply = "Check the route path"
    with client.stream("POST", "/api/chat", json=openclaw_body()) as r:
        lines = [json.loads(l) for l in r.iter_lines() if l]
    assert r.status_code == 200
    assert "".join(l["message"]["content"] for l in lines).strip() == "Check the route path"
    assert lines[-1]["done"] is True
    sent = primary.requests[-1]
    assert sent["tools"][0]["function"]["name"] == "http_get"      # tools untouched
    assert sent["options"]["num_ctx"] == 16384
    assert sent["messages"][0]["content"].startswith("You are OpenClaw.")
    assert "primary problem-solving model" in sent["messages"][0]["content"]
    assert sent["messages"][2]["content"] == "hi"                     # history untouched
    assert "<PROJECT_MEMORY>" in sent["messages"][-1]["content"]
    assert "<PROJECT_MEMORY>" not in sent["messages"][1]["content"]  # only the latest user msg
    tasks = orch.db.list_tasks()
    assert sorted(t["kind"] for t in tasks) == ["extract", "summary"]
    assert all(t["status"] == "pending" for t in tasks)


def test_proxy_tool_loop_queues_only_final_turn(env):
    client, orch, primary, memory = env
    primary.tool_calls = [{"function": {"name": "http_get", "arguments": {"url": "http://kali/mcp"}}}]
    r = client.post("/api/chat", json=openclaw_body(stream=False))
    assert r.json()["message"]["tool_calls"][0]["function"]["name"] == "http_get"
    assert orch.db.list_tasks() == []                                  # mid-turn: nothing queued

    first_ctx = primary.requests[-1]["messages"][-1]["content"]
    primary.tool_calls = None
    primary.reply = "The endpoint is /mcp"
    extra = [{"role": "assistant", "content": "", "tool_calls": [
                 {"function": {"name": "http_get", "arguments": {"url": "http://kali/mcp"}}}]},
             {"role": "tool", "tool_name": "http_get", "content": "404 Not Found"}]
    r = client.post("/api/chat", json=openclaw_body(stream=False, extra=extra))
    assert r.json()["message"]["content"] == "The endpoint is /mcp"
    # Same memory block on every step of the turn (keeps the KV-cache prefix stable).
    assert primary.requests[-1]["messages"][-3]["content"] == first_ctx
    tasks = orch.db.list_tasks()
    assert sorted(t["kind"] for t in tasks) == ["extract", "summary"]
    types = [r["type"] for r in orch.conv_log.iter_records()]
    assert types == ["message", "tool_call", "tool_result", "message"]


def test_proxy_passthrough_and_errors(env):
    client, _, primary, _ = env
    assert client.get("/api/tags").json()["models"][0]["name"] == "qwen3:14b"
    assert "Ollama is running" in client.get("/").text
    primary.fail_status = 500
    r = client.post("/api/chat", json=openclaw_body(stream=False))
    assert r.status_code == 500


def test_health_and_metrics(env):
    client, *_ = env
    client.post("/chat", json={"message": "hello there, the MCP fails"})
    h = client.get("/health").json()
    assert h["primary"]["reachable"] and h["memory"]["reachable"] and h["memory_files_ok"]
    m = client.get("/metrics").json()
    assert m["totals"]["requests"] == 1 and m["averages"]["prompt_tokens"] == 100
    assert client.post("/memory/consolidate").status_code == 501


def test_end_to_end_memory_is_used_later(env):
    """Definition of done 1-13: a fact learned in one turn is injected into a later one."""
    client, orch, primary, memory = env
    memory.memory_json = {"changes": [{
        "category": "environment", "operation": "add", "title": "Memory GPU",
        "content": "The memory model runs on the RTX 2070 SUPER via the Ollama instance on port 11435.",
        "confidence": 0.95, "reason": "Stated by the user."}]}
    client.post("/chat", json={"message": "I configured the memory model on the 2070 SUPER, port 11435"})
    process_all(client, orch)
    assert orch.db.list_changes()[0]["status"] == "approved"

    client.post("/chat", json={"message": "Which GPU does the memory model use?"})
    assert "[E-001] Memory GPU" in primary.requests[-1]["messages"][-1]["content"]
    assert client.get("/memory/search", params={"q": "2070 SUPER"}).json()["results"][0]["entry_id"] == "E-001"


def test_rebuild_replay_reset(env):
    client, orch, primary, memory = env
    client.post("/chat", json={"message": "We decided to use qwen3:8b for memory"})
    process_all(client, orch)
    r = client.post("/memory/rebuild", json={"replay": True, "reset": True}).json()
    assert r["queued"] == 1 and orch.manager.stores[Category.lesson].entries() == []
    assert (orch.config.backups_dir).exists() and any(orch.config.backups_dir.iterdir())
