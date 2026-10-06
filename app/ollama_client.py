"""Minimal async client for one Ollama instance (native API).

Each instance is a separate endpoint so the primary and memory models can be
pinned to different GPUs. The app never assumes they share an endpoint.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from .config import OllamaEndpoint


class OllamaError(Exception):
    pass


class OllamaTimeout(OllamaError):
    pass


class OllamaConnectionError(OllamaError):
    pass


class OllamaResponseError(OllamaError):
    def __init__(self, message: str, status_code: int | None = None, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class OllamaClient:
    def __init__(self, endpoint: OllamaEndpoint, transport: httpx.AsyncBaseTransport | None = None):
        self.endpoint = endpoint
        self.base_url = endpoint.base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(endpoint.timeout_seconds, connect=10.0),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @property
    def http(self) -> httpx.AsyncClient:
        return self._client

    # ------------------------------------------------------------- helpers
    def default_options(self, options: dict | None = None) -> dict:
        opts = dict(options or {})
        if self.endpoint.num_ctx and "num_ctx" not in opts:
            opts["num_ctx"] = self.endpoint.num_ctx
        return opts

    async def _request(self, method: str, path: str, **kw) -> httpx.Response:
        try:
            resp = await self._client.request(method, path, **kw)
        except httpx.TimeoutException as e:
            raise OllamaTimeout(f"{self.base_url}{path}: timed out") from e
        except httpx.TransportError as e:
            raise OllamaConnectionError(f"{self.base_url}{path}: {e.__class__.__name__}: {e}") from e
        if resp.status_code >= 400:
            raise OllamaResponseError(f"{self.base_url}{path}: HTTP {resp.status_code}",
                                      resp.status_code, resp.text[:2000])
        return resp

    @staticmethod
    def _json(resp: httpx.Response) -> dict:
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as e:
            raise OllamaResponseError("malformed JSON from Ollama", resp.status_code, resp.text[:500]) from e
        if not isinstance(data, dict):
            raise OllamaResponseError("unexpected JSON shape from Ollama", resp.status_code, resp.text[:500])
        return data

    # ------------------------------------------------------------------ API
    async def chat(self, messages: list[dict], *, model: str | None = None,
                   options: dict | None = None, format: Any = None, think: bool | None = None,
                   tools: list | None = None) -> dict:
        body: dict[str, Any] = {
            "model": model or self.endpoint.model,
            "messages": messages,
            "stream": False,
            "options": self.default_options(options),
        }
        if format is not None:
            body["format"] = format
        think = self.endpoint.think if think is None else think
        if think is not None:
            body["think"] = think
        if tools:
            body["tools"] = tools
        if self.endpoint.keep_alive:
            body["keep_alive"] = self.endpoint.keep_alive
        data = self._json(await self._request("POST", "/api/chat", json=body))
        msg = data.get("message")
        if not isinstance(msg, dict) or not isinstance(msg.get("content", ""), str):
            raise OllamaResponseError("Ollama response has no message.content", 200, json.dumps(data)[:500])
        return data

    async def stream_raw(self, method: str, path: str, *, json_body: Any = None,
                         content: bytes | None = None, headers: dict | None = None,
                         params: Any = None) -> tuple[httpx.Response, AsyncIterator[bytes]]:
        """Open a streaming request; caller iterates the body and must close the response."""
        req = self._client.build_request(method, path, json=json_body, content=content,
                                         headers=headers, params=params)
        try:
            resp = await self._client.send(req, stream=True)
        except httpx.TimeoutException as e:
            raise OllamaTimeout(f"{self.base_url}{path}: timed out") from e
        except httpx.TransportError as e:
            raise OllamaConnectionError(f"{self.base_url}{path}: {e.__class__.__name__}: {e}") from e
        return resp, resp.aiter_raw()

    async def version(self) -> dict:
        return self._json(await self._request("GET", "/api/version"))

    async def ps(self) -> dict:
        return self._json(await self._request("GET", "/api/ps"))

    async def health(self) -> dict:
        """Reachability plus GPU residency of loaded models (size_vram vs size)."""
        out: dict[str, Any] = {"base_url": self.base_url, "model": self.endpoint.model}
        try:
            out["version"] = (await self.version()).get("version")
            loaded = (await self.ps()).get("models", []) or []
            out["reachable"] = True
            out["loaded_models"] = []
            for m in loaded:
                size, vram = m.get("size") or 0, m.get("size_vram") or 0
                pct = round(100 * vram / size, 1) if size else None
                out["loaded_models"].append({
                    "name": m.get("name"), "size_bytes": size, "vram_bytes": vram,
                    "gpu_percent": pct, "fully_on_gpu": bool(size) and vram >= size,
                    "context_length": m.get("context_length"),
                })
            out["warnings"] = [f"{m['name']} is only {m['gpu_percent']}% on GPU (CPU spill)"
                               for m in out["loaded_models"] if m["gpu_percent"] is not None
                               and not m["fully_on_gpu"]]
        except OllamaError as e:
            out["reachable"] = False
            out["error"] = str(e)
        return out
