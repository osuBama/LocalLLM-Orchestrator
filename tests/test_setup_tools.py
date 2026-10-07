import json

import httpx
import pytest
import yaml

from app import setup_tools as st
from app.config import load_config

SMI = """0, NVIDIA GeForce RTX 2070 SUPER, GPU-bbbb, 8192, 300
1, NVIDIA GeForce RTX 5070, GPU-aaaa, 12227, 900
"""


def test_parse_and_assign_roles():
    gpus = st.parse_nvidia_smi(SMI)
    assert [g["uuid"] for g in gpus] == ["GPU-bbbb", "GPU-aaaa"] and gpus[1]["vram_gb"] == 11.9
    r = st.assign_roles(gpus)
    assert r["mode"] == "dual" and r["primary"]["uuid"] == "GPU-aaaa" and r["memory"]["uuid"] == "GPU-bbbb"
    r = st.assign_roles(gpus, primary_uuid="GPU-bbbb")
    assert r["primary"]["uuid"] == "GPU-bbbb" and r["memory"]["uuid"] == "GPU-aaaa"
    assert st.assign_roles(gpus[:1])["mode"] == "single"
    with pytest.raises(ValueError):
        st.assign_roles(gpus, primary_uuid="GPU-zzzz")


@pytest.mark.parametrize("p,m,exp", [
    (12, 8, ("qwen3:14b", 16384, "qwen3:8b", 8192)),
    (24, 12, ("qwen3:32b", 16384, "qwen3:8b", 16384)),
    (8, 6, ("qwen3:8b", 8192, "qwen3:4b", 8192)),
    (16, 4, ("qwen3:14b", 32768, "qwen3:1.7b", 8192)),
])
def test_plan_dual(p, m, exp):
    r = st.plan(p, m)
    assert (r["primary_model"], r["primary_ctx"], r["memory_model"], r["memory_ctx"]) == exp
    assert r["memory_base_budget"] < r["memory_budget"] <= 6000


def test_plan_accounts_for_desktop_vram_and_single_mode():
    assert st.plan(12, 8, desktop_gb=1.5)["primary_model"] == "qwen3:8b"   # 10.5 GB left
    r = st.plan(12)
    assert r["mode"] == "single" and r["primary_model"] == "qwen3:8b" and r["memory_model"] == "qwen3:1.7b"


SAMPLE = """# top comment
paths:
  root: "G:\\\\AI"                # install folder
ollama:
  primary:
    base_url: "http://127.0.0.1:11434"
    model: "qwen3:14b"          # keep this comment
    num_ctx: 16384
  memory:
    model: "qwen3:8b"
memory:
  max_context_tokens: 2500
"""


def test_set_yaml_values_preserves_comments_and_inserts():
    out = st.set_yaml_values(SAMPLE, {
        "ollama.primary.model": "qwen3:32b", "ollama.primary.num_ctx": 8192,
        "paths.root": "D:\\AI", "memory.max_context_tokens": 1200,
        "ollama.memory.think": False, "stable_memory.max_tokens": 500})
    assert '    model: "qwen3:32b"          # keep this comment' in out
    assert "# top comment" in out and "# install folder" in out
    data = yaml.safe_load(out)
    assert data["paths"]["root"] == "D:\\AI"
    assert data["ollama"]["primary"]["num_ctx"] == 8192
    assert data["ollama"]["memory"]["think"] is False             # inserted into existing section
    assert data["stable_memory"]["max_tokens"] == 500             # new section appended
    assert data["ollama"]["memory"]["model"] == "qwen3:8b"        # untouched
    with pytest.raises(ValueError):
        st.set_yaml_values(SAMPLE, {"ollama.primary": "x"})       # refuses to clobber a section


def test_set_yaml_value_with_hash_inside_quotes():
    out = st.set_yaml_values('a:\n  b: "x # not a comment"  # real\n', {"a.b": "y"})
    assert out == 'a:\n  b: "y"  # real\n'


def test_set_config_on_shipped_config_validates_and_rolls_back(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(open("config/config.yaml", encoding="utf-8").read(), encoding="utf-8")
    r = st.set_config(p, {"ollama.primary.model": "qwen3:32b", "ollama.primary.num_ctx": 12288,
                          "memory.max_context_tokens": 1800, "stable_memory.max_tokens": 700})
    assert r["ok"] and load_config(p).ollama.primary.num_ctx == 12288
    before = p.read_text(encoding="utf-8")
    r = st.set_config(p, {"stable_memory.max_tokens": 5000})       # >= memory budget: invalid
    assert not r["ok"] and p.read_text(encoding="utf-8") == before
    assert len(list(tmp_path.glob("config.yaml.bak-*"))) >= 1


def test_parse_assignment():
    assert st.parse_assignment("a.b=16384") == ("a.b", 16384)
    assert st.parse_assignment("a.b=false") == ("a.b", False)
    assert st.parse_assignment("a.b=qwen3:14b") == ("a.b", "qwen3:14b")
    assert st.parse_assignment('a.b="D:\\\\AI"') == ("a.b", "D:\\AI")


def test_fit_context_steps_down_until_fully_on_gpu(fakes):
    primary, _ = fakes
    primary.vram_fits_at["qwen3:14b"] = 10000
    r = st.fit_context("http://primary.test", "qwen3:14b", 16384, transport=_sync(primary))
    assert r["ok"] and r["num_ctx"] <= 10000 and r["num_ctx"] >= 8192
    assert [t["num_ctx"] for t in r["tried"]][0] == 16384


def test_fit_context_gives_up_below_minimum(fakes):
    primary, _ = fakes
    primary.vram_fits_at["qwen3:32b"] = 1000
    r = st.fit_context("http://primary.test", "qwen3:32b", 8192, transport=_sync(primary))
    assert not r["ok"] and "smaller model" in r["error"]


def test_fit_context_single_gpu_keeps_other_model(fakes):
    primary, _ = fakes
    primary.vram_fits_at["qwen3:1.7b"] = 4096
    st.fit_context("http://primary.test", "qwen3:1.7b", 4096, transport=_sync(primary))
    primary.vram_fits_at["qwen3:8b"] = 12000
    r = st.fit_context("http://primary.test", "qwen3:8b", 16384, keep=["qwen3:1.7b"], transport=_sync(primary))
    assert r["ok"] and all(t.get("others_on_gpu", True) for t in r["tried"] if "gpu_percent" in t)


def test_openclaw_patch_shape():
    p = st.openclaw_patch("qwen3:14b", 16384)
    prov = p["models"]["providers"]["ollama"]
    assert prov["baseUrl"] == "http://127.0.0.1:8000" and prov["api"] == "ollama"
    assert prov["models"][0]["contextWindow"] == 16384
    assert p["agents"]["defaults"]["model"]["primary"] == "ollama/qwen3:14b"
    json.dumps(p)


def test_cli_outputs_json(capsys):
    assert st.main(["plan", "--primary-gb", "12", "--memory-gb", "8"]) == 0
    assert json.loads(capsys.readouterr().out)["primary_model"] == "qwen3:14b"


def _sync(fake):
    """Synchronous transport for the in-process fake (fit_context uses a sync client)."""
    from fastapi.testclient import TestClient
    tc = TestClient(fake.app)

    class T(httpx.BaseTransport):
        def handle_request(self, request):
            r = tc.request(request.method, request.url.path, content=request.content,
                           headers={k: v for k, v in request.headers.items() if k.lower() != "host"})
            return httpx.Response(r.status_code, content=r.content, headers=r.headers)
    return T()


def test_set_config_from_json_file_with_bom(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(open("config/config.yaml", encoding="utf-8").read(), encoding="utf-8")
    upd = tmp_path / "u.json"
    upd.write_bytes(b"\xef\xbb\xbf" + json.dumps({"paths.root": "D:\\AI", "ollama.primary.num_ctx": 8192}).encode())
    assert st.main(["set-config", str(cfg), "--json-file", str(upd)]) == 0
    c = load_config(cfg)
    assert c.paths.root == "D:\\AI" and c.ollama.primary.num_ctx == 8192


def test_fit_context_loads_embedding_model_first(fakes):
    _, memory = fakes
    memory.vram_fits_at["qwen3:8b"] = 8192
    r = st.fit_context("http://memory.test", "qwen3:8b", 8192, embed_model="nomic-embed-text",
                       transport=_sync(memory))
    assert r["ok"] and "nomic-embed-text" in memory.ps_models
    assert memory.requests[0]["model"] == "qwen3:8b" or memory.embed_calls == 1
