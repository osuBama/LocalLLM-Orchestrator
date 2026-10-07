import json

import pytest
import yaml
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import load_config
from app.schemas import Category

H = {"X-AI-Client": "1"}


@pytest.fixture
def env(tmp_path, fakes):
    primary, memory = fakes
    cfg_file = tmp_path / "config.yaml"
    text = open("config/config.yaml", encoding="utf-8").read()
    cfg_file.write_text(text, encoding="utf-8")
    from app.setup_tools import set_config
    set_config(cfg_file, {"paths.root": str(tmp_path), "embeddings.enabled": False,
                          "memory.worker_enabled": False,
                          "ollama.primary.base_url": "http://primary.test",
                          "ollama.memory.base_url": "http://memory.test"})
    cfg = load_config(cfg_file)
    app = create_app(cfg, primary_transport=primary.transport, memory_transport=memory.transport,
                     console_logs=False)
    with TestClient(app) as client:
        yield client, app.state.orch, cfg_file, tmp_path


def test_page_is_served(env):
    client, *_ = env
    r = client.get("/ui")
    assert r.status_code == 200 and "Local AI console" in r.text and "X-AI-Client" in r.text


def test_settings_live_and_restart(env):
    client, orch, cfg_file, _ = env
    s = client.get("/ui/api/settings").json()
    assert s["editable"] and any(x["key"] == "thinking.mode" for x in s["settings"])
    r = client.post("/ui/api/settings", json={"values": {"thinking.mode": "off", "ollama.primary.num_ctx": 12288}},
                    headers=H).json()
    assert r["applied_now"] == ["thinking.mode"] and r["restart_required"] == ["ollama.primary.num_ctx"]
    assert orch.config.thinking.mode == "off"                      # live
    assert orch.config.ollama.primary.num_ctx == 16384             # not until restart
    on_disk = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    assert on_disk["thinking"]["mode"] == "off" and on_disk["ollama"]["primary"]["num_ctx"] == 12288
    assert "# Primary-model thinking per turn" in cfg_file.read_text(encoding="utf-8")   # comments kept


def test_settings_rejects_invalid_and_unknown(env):
    client, orch, cfg_file, _ = env
    before = cfg_file.read_text(encoding="utf-8")
    r = client.post("/ui/api/settings", json={"values": {"stable_memory.max_tokens": 99999}}, headers=H)
    assert r.status_code == 400 and cfg_file.read_text(encoding="utf-8") == before
    r = client.post("/ui/api/settings", json={"values": {"paths.root": "C:\\\\x"}}, headers=H)
    assert r.status_code == 400
    assert client.post("/ui/api/settings", json={"values": {"thinking.mode": "on"}}).status_code == 403


def test_memory_edit_add_deactivate(env):
    client, orch, *_ = env
    r = client.post("/ui/api/memory/entry", json={"category": "lesson", "title": "Route",
                                                  "content": "MCP route is /mcp."}, headers=H).json()
    assert r["entry_id"] == "L-001"
    client.post("/ui/api/memory/entry", json={"category": "lesson", "entry_id": "L-001",
                                              "content": "MCP route is /mcp, not /sse."}, headers=H)
    data = client.get("/ui/api/memory").json()
    assert data["lesson"]["entries"][0]["content"] == "MCP route is /mcp, not /sse."
    client.post("/ui/api/memory/entry", json={"category": "lesson", "entry_id": "L-001", "active": False}, headers=H)
    assert client.get("/ui/api/memory").json()["lesson"]["entries"][0]["active"] is False
    assert [c["conversation_id"] for c in orch.db.list_changes()][:3] == ["ui", "ui", "ui"]
    assert client.post("/ui/api/memory/entry", json={"category": "lesson", "entry_id": "L-099",
                                                     "content": "x"}, headers=H).status_code == 404


def test_reports_list_and_get(env):
    client, orch, cfg_file, root = env
    d = root / "evals" / "reports"
    d.mkdir(parents=True)
    rep = {"started": "2026-10-07_10-00-00", "model": "qwen3:14b", "variants": {"baseline": {}, "full": {}},
           "options": {"sessions": ["a", "b"], "golden_cases": ["x"], "repeats": 3},
           "regressions": {"flags": ["full: something"]}}
    (d / "2026-10-07_10-00-00.json").write_text(json.dumps(rep))
    lst = client.get("/ui/api/evals").json()["reports"]
    assert lst[0]["variants"] == ["baseline", "full"] and lst[0]["repeats"] == 3 and lst[0]["flags"]
    assert client.get("/ui/api/evals/2026-10-07_10-00-00").json()["model"] == "qwen3:14b"
    assert client.get("/ui/api/evals/..%2F..%2Fsecret").status_code in (400, 404)


def test_eval_run_validation(env):
    client, *_ = env
    assert client.post("/ui/api/evals/run", json={"variants": "full; rm -rf /"}, headers=H).status_code == 400
    assert client.get("/ui/api/evals/run/status").json()["running"] is False


def test_candidate_accept_and_dismiss(env):
    client, orch, cfg_file, root = env
    a = orch.db.add_candidate("s", 2, "which port?", "11434", "no, it's 11435", ["11435"])
    b = orch.db.add_candidate("s", 4, "q?", "x", "no, it's y", [])
    r = client.post(f"/ui/api/candidates/{a}/accept", json={}, headers=H).json()
    assert r["case"]["expect_all"] == ["11435"] and (root / "evals" / "golden.yaml").exists()
    assert client.post(f"/ui/api/candidates/{b}/accept", json={}, headers=H).status_code == 400   # no expectation
    assert client.post(f"/ui/api/candidates/{b}/dismiss", headers=H).status_code == 200


def test_chat_returns_thinking_info(env):
    client, *_ = env
    r = client.post("/chat", json={"message": "list the files"}).json()
    assert r["thinking"] == "off" and r["thinking_reason"] and "seconds" in r
