import asyncio

import httpx
import pytest

from app.config import OllamaEndpoint
from app.ollama_client import (OllamaClient, OllamaConnectionError, OllamaResponseError,
                               OllamaTimeout)

EP = OllamaEndpoint(base_url="http://x.test", model="m", timeout_seconds=1, num_ctx=4096)


def client(handler):
    return OllamaClient(EP, transport=httpx.MockTransport(handler))


def run(coro):
    return asyncio.run(coro)


def test_success_and_defaults():
    seen = {}

    def h(req):
        import json
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "hi"}, "done": True})
    out = run(client(h).chat([{"role": "user", "content": "x"}]))
    assert out["message"]["content"] == "hi"
    assert seen["stream"] is False and seen["options"]["num_ctx"] == 4096 and seen["model"] == "m"


def test_timeout():
    def h(req):
        raise httpx.ReadTimeout("slow", request=req)
    with pytest.raises(OllamaTimeout):
        run(client(h).chat([]))


def test_connection_failure():
    def h(req):
        raise httpx.ConnectError("refused", request=req)
    with pytest.raises(OllamaConnectionError):
        run(client(h).chat([]))


def test_malformed_response():
    with pytest.raises(OllamaResponseError):
        run(client(lambda r: httpx.Response(200, text="<html>nope")).chat([]))
    with pytest.raises(OllamaResponseError):
        run(client(lambda r: httpx.Response(200, json={"done": True})).chat([]))
    with pytest.raises(OllamaResponseError) as e:
        run(client(lambda r: httpx.Response(500, text="model not found")).chat([]))
    assert e.value.status_code == 500


def test_health_reports_cpu_spill():
    def h(req):
        if req.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.99"})
        return httpx.Response(200, json={"models": [{"name": "qwen3:14b", "size": 100, "size_vram": 60}]})
    out = run(client(h).health())
    assert out["reachable"] and out["loaded_models"][0]["gpu_percent"] == 60.0 and out["warnings"]
