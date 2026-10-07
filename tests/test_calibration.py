import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.calibration import DEFAULT_CPT, TokenCalibrator
from app.database import Database


def test_default_until_enough_observations(tmp_path):
    c = TokenCalibrator(Database(tmp_path / "m.db"))
    for _ in range(4):
        c.observe("m", 4400, 1000)
    assert c.chars_per_token("m") == DEFAULT_CPT
    c.observe("m", 4400, 1000)
    assert c.chars_per_token("m") == pytest.approx(4.4)


def test_cache_hits_are_ignored_and_minimum_is_used(tmp_path):
    c = TokenCalibrator(Database(tmp_path / "m.db"))
    for _ in range(50):
        c.observe("m", 40000, 900)               # cache hits (ratio 44): must not count at all
    assert c.snapshot().get("m", {}).get("observations", 0) == 0 and c.chars_per_token("m") == DEFAULT_CPT
    for ratio_tokens in (10000, 9500, 10500, 9800, 10200):   # uncached requests around 4 chars/token
        c.observe("m", 40000, ratio_tokens)
    assert c.chars_per_token("m") == pytest.approx(40000 / 10500, abs=1e-3)    # the conservative end
    for _ in range(50):
        c.observe("m", 40000, 900)
    assert c.chars_per_token("m") == pytest.approx(40000 / 10500, abs=1e-3)    # still unaffected by warm traffic


def test_bounds_tiny_prompts_and_persistence(tmp_path):
    db = Database(tmp_path / "m.db")
    c = TokenCalibrator(db)
    for _ in range(10):
        c.observe("m", 1000, 1000)               # 1.0 chars/token -> clamped to the lower bound
        c.observe("m", 100, 1)                   # below min_chars -> ignored
    assert c.chars_per_token("m") == 2.2
    assert TokenCalibrator(db).chars_per_token("m") == 2.2   # survives restart
    assert c.tokens("m", 2200) == 1050                        # includes the 5% margin


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


def long_session(n):
    msgs = []
    for i in range(1, n):
        msgs += [{"role": "user", "content": f"q{i} " + "word " * 200},
                 {"role": "assistant", "content": f"a{i} " + "word " * 200}]
    return {"model": "qwen3:14b", "stream": False, "messages": msgs + [{"role": "user", "content": "next?"}]}


def test_calibration_keeps_history_the_default_would_cut(env):
    client, orch, primary = env
    orch.db.set_summary("c1", "summary", 100)
    orch.db.set_summary("c2", "summary", 100)
    client.post("/api/chat", json=long_session(12), headers={"X-Conversation-Id": "c1"})
    default_drop = client.get("/metrics").json()["recent_requests"][-1]["user_turns_dropped"]
    for _ in range(10):
        orch.calibrator.observe("qwen3:14b", 50000, 10000)            # real ratio: 5 chars/token
    client.post("/api/chat", json=long_session(12), headers={"X-Conversation-Id": "c2"})
    m = client.get("/metrics").json()
    assert default_drop > 0 and m["recent_requests"][-1]["user_turns_dropped"] < default_drop
    assert m["token_calibration"]["qwen3:14b"]["calibrated"] is True


def test_proxy_feeds_observations(env):
    client, orch, primary = env
    primary.simulate_cache = True            # realistic counts: first request is uncached
    client.post("/api/chat", json=long_session(3))
    assert orch.calibrator.snapshot()["qwen3:14b"]["observations"] == 1
