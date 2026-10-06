import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.schemas import Category
from app.util import estimate_tokens


@pytest.fixture
def env(cfg, fakes):
    primary, memory = fakes
    app = create_app(cfg, primary_transport=primary.transport, memory_transport=memory.transport,
                     console_logs=False)
    orch = app.state.orch
    st = orch.manager.stores
    st[Category.constraint].add("No cloud", "No cloud inference; run everything locally.")
    st[Category.environment].add("Primary GPU", "Primary model runs on the larger GPU on port 11434.")
    st[Category.objective].add("Memory system", "Build the dual-GPU memory system.")
    st[Category.lesson].add("MCP 404", "When MCP returns 404, verify the route path first.")
    with TestClient(app) as client:
        yield client, orch, primary


def body(turns, last="MCP returns 404 again"):
    msgs = [{"role": "system", "content": "You are OpenClaw."}]
    for i in range(1, turns):
        msgs += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    return {"model": "qwen3:14b", "stream": False, "messages": msgs + [{"role": "user", "content": last}]}


def sent(primary):
    m = primary.requests[-1]["messages"]
    return m[0]["content"], m[-1]["content"]


def test_base_in_system_relevant_in_user(env):
    client, orch, primary = env
    client.post("/api/chat", json=body(1))
    system, user = sent(primary)
    assert system.startswith("You are OpenClaw.") and "<PROJECT_MEMORY_BASE>" in system
    for eid in ("[C-001]", "[E-001]", "[O-001]"):
        assert eid in system and eid not in user                  # no duplication
    assert "[L-001]" in user and "[L-001]" not in system          # relevant entries stay per-turn


def test_base_frozen_within_epoch_changes_arrive_as_updates(env):
    client, orch, primary = env
    client.post("/api/chat", json=body(2))
    sys2, _ = sent(primary)
    st = orch.manager.stores
    st[Category.constraint].add("No telemetry", "Never send telemetry anywhere.")
    st[Category.environment].update("E-001", content="Primary model runs on GPU A, port 11434.")
    st[Category.objective].deactivate("O-001")

    client.post("/api/chat", json=body(3))
    sys3, user3 = sent(primary)
    assert sys3 == sys2                                           # system prompt byte-identical
    assert "UPDATED SINCE THE MEMORY BASE" in user3
    assert "[C-002] No telemetry" in user3
    assert "[E-001] Primary GPU: Primary model runs on GPU A" in user3
    assert "[O-001] Memory system: no longer applies" in user3


def test_base_refreshes_at_epoch_move_and_updates_disappear(env):
    client, orch, primary = env
    orch.config.proxy.trim_mode = "turns"   # this test covers the turn-count schedule
    client.post("/api/chat", json=body(7))                        # compression boundary 0
    sys7, _ = sent(primary)
    orch.manager.stores[Category.constraint].add("No telemetry", "Never send telemetry anywhere.")
    client.post("/api/chat", json=body(8))                        # boundary moves to 6 -> new epoch
    sys8, user8 = sent(primary)
    assert sys8 != sys7 and "[C-002]" in sys8
    assert "UPDATED SINCE" not in user8


def test_unchanged_memory_gives_identical_base_across_epochs(env):
    client, orch, primary = env
    client.post("/api/chat", json=body(7))
    sys7, _ = sent(primary)
    client.post("/api/chat", json=body(8))
    sys8, _ = sent(primary)
    assert sys8 == sys7                                           # refresh, but nothing changed


def test_budget_shared_between_base_and_turn_block(env):
    client, orch, primary = env
    for i in range(40):
        orch.manager.stores[Category.lesson].add(f"MCP lesson {i}", "MCP 404 route detail " * 30)
    client.post("/api/chat", json=body(1))
    m = client.get("/metrics").json()["recent_requests"][-1]
    assert m["memory_base_tokens"] > 0
    assert m["memory_base_tokens"] + m["memory_tokens"] <= orch.config.memory.max_context_tokens


def test_base_overflow_entries_fall_back_to_retrieval(env):
    client, orch, primary = env
    orch.config.stable_memory.max_tokens = 260
    names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    for n in names:
        orch.manager.stores[Category.environment].add(f"Host {n}", f"Server {n} listens on its own port.")
    client.post("/api/chat", json=body(1, last="which port does server juliet use"))
    system, user = sent(primary)
    assert "[E-001]" in system                                   # base has room for some
    assert "[E-011]" not in system and "[E-011] Host juliet" in user  # overflow -> retrieved


def test_disabled_puts_everything_per_turn(env):
    client, orch, primary = env
    orch.config.stable_memory.enabled = False
    client.post("/api/chat", json=body(1))
    system, user = sent(primary)
    assert "PROJECT_MEMORY_BASE" not in system and "[C-001]" in user


def test_chat_endpoint_uses_base(env):
    client, orch, primary = env
    r = client.post("/chat", json={"message": "MCP 404 again"}).json()
    system, user = sent(primary)
    assert "[C-001]" in system and "[C-001]" not in user and "C-001" in r["memory_entries_used"]


def test_base_text_is_deterministic(env):
    _, orch, _ = env
    cats = [Category.constraint, Category.objective, Category.environment]
    a = orch.context_builder.build_stable(cats, 1000)
    b = orch.context_builder.build_stable(cats, 1000)
    assert a.text == b.text and a.tokens == estimate_tokens(a.text)
