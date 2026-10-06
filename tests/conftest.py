import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config  # noqa: E402


def make_config(root: Path, **memory_overrides) -> Config:
    mem = {"worker_enabled": False, "retry_base_seconds": 0, "max_attempts": 3,
           "max_context_tokens": 2500}
    mem.update(memory_overrides)
    return Config(
        paths={"root": str(root)},
        ollama={
            "primary": {"base_url": "http://primary.test", "model": "qwen3:14b", "num_ctx": 16384,
                        "timeout_seconds": 5},
            "memory": {"base_url": "http://memory.test", "model": "qwen3:8b", "num_ctx": 8192,
                       "timeout_seconds": 5, "think": False},
        },
        memory=mem,
        application={"log_level": "WARNING"},
    )


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path)


class FakeOllama:
    """A tiny stand-in for one Ollama instance, served in-process via ASGITransport."""

    def __init__(self, name: str):
        self.name = name
        self.requests: list[dict] = []
        self.reply = "fake answer"
        self.tool_calls: list | None = None
        self.memory_json: dict | str = {"changes": []}
        self.fail_status: int | None = None
        self.simulate_cache = False     # prompt_eval_count = chars after the shared prefix / 4
        self._last_prompt = ""
        self.app = FastAPI()
        app = self.app

        @app.post("/api/chat")
        async def chat(request: Request):
            body = await request.json()
            self.requests.append(body)
            if self.fail_status:
                return JSONResponse({"error": "boom"}, status_code=self.fail_status)
            if body.get("format") is not None:  # memory model call
                content = self.memory_json if isinstance(self.memory_json, str) else json.dumps(self.memory_json)
                return {"model": body["model"], "message": {"role": "assistant", "content": content},
                        "done": True, "total_duration": 1_000_000}
            evaluated = 100
            if self.simulate_cache:
                prompt = "".join(f"{m.get('role')}:{m.get('content', '')}\n" for m in body["messages"])
                lcp = 0
                for a, b in zip(prompt, self._last_prompt):
                    if a != b:
                        break
                    lcp += 1
                self._last_prompt = prompt
                evaluated = max(1, (len(prompt) - lcp) // 4)
            final = {"model": body["model"], "done": True, "done_reason": "stop",
                     "message": {"role": "assistant", "content": ""},
                     "prompt_eval_count": evaluated, "prompt_eval_duration": evaluated * 100_000,
                     "eval_count": 10, "eval_duration": 200_000_000, "total_duration": 300_000_000}
            if body.get("stream", True) is False:
                msg = {"role": "assistant", "content": self.reply}
                if self.tool_calls:
                    msg = {"role": "assistant", "content": "", "tool_calls": self.tool_calls}
                return {**final, "message": msg}

            def gen():
                if self.tool_calls:
                    yield json.dumps({"model": body["model"], "done": False, "message": {
                        "role": "assistant", "content": "", "tool_calls": self.tool_calls}}) + "\n"
                else:
                    for word in self.reply.split(" "):
                        yield json.dumps({"model": body["model"], "done": False, "message": {
                            "role": "assistant", "content": word + " "}}) + "\n"
                yield json.dumps(final) + "\n"
            return StreamingResponse(gen(), media_type="application/x-ndjson")

        @app.get("/api/tags")
        async def tags():
            return {"models": [{"name": "qwen3:14b"}]}

        @app.get("/api/version")
        async def version():
            return {"version": "0.99.0"}

        @app.get("/api/ps")
        async def ps():
            return {"models": [{"name": "qwen3:14b", "size": 100, "size_vram": 100, "context_length": 16384}]}

        @app.get("/")
        async def root():
            return "Ollama is running"

    @property
    def transport(self):
        return httpx.ASGITransport(app=self.app)


@pytest.fixture
def fakes():
    return FakeOllama("primary"), FakeOllama("memory")
