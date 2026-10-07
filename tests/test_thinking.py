import pytest
from fastapi.testclient import TestClient

from app import thinking
from app.api import create_app


@pytest.mark.parametrize("text,simple", [
    ("run the tests", True),
    ("what port is the memory instance on?", True),
    ("list the files in src", True),
    ("corre os testes", True),
    ("why does the MCP server return 404?", False),
    ("debug the pool exhaustion", False),
    ("implement retry with backoff in the client", False),
    ("porque é que o servidor falha?", False),
    ("think about this: rename the variable", False),
    ("rename x to y step by step", False),
    ("Traceback (most recent call last): boom", False),
    ("```py\nx=1\n```", False),
    ("word " * 40, False),
])
def test_classify(text, simple):
    assert thinking.classify(text)[0] is simple


def test_decide_modes():
    K = thinking.KEEP
    assert thinking.decide("auto", None, "run the tests")[0] is False
    assert thinking.decide("auto", True, "run the tests")[0] is False         # downgrade allowed
    assert thinking.decide("auto", None, "why is it slow?")[0] is K            # never upgraded
    assert thinking.decide("auto", False, "why is it slow?")[0] is K           # client off respected
    assert thinking.decide("client", None, "run the tests")[0] is K
    assert thinking.decide("on", False, "hi")[0] is True
    assert thinking.decide("off", True, "why?")[0] is False


@pytest.fixture
def env(cfg, fakes):
    primary, memory = fakes
    app = create_app(cfg, primary_transport=primary.transport, memory_transport=memory.transport,
                     console_logs=False)
    with TestClient(app) as client:
        yield client, app.state.orch, primary


def body(text, think=None, extra=None):
    b = {"model": "qwen3:14b", "stream": False,
         "messages": [{"role": "user", "content": text}] + (extra or [])}
    if think is not None:
        b["think"] = think
    return b


def test_proxy_turns_thinking_off_for_simple_turns_only(env):
    client, orch, primary = env
    client.post("/api/chat", json=body("run the tests", think=True))
    assert primary.requests[-1]["think"] is False
    client.post("/api/chat", json=body("why do the tests fail on CI?", think=True))
    assert primary.requests[-1]["think"] is True                          # left as the client sent
    client.post("/api/chat", json=body("why do they fail now?"))
    assert "think" not in primary.requests[-1]                            # never added
    m = client.get("/metrics").json()["recent_requests"]
    assert m[0]["thinking"] == "off" and m[1]["thinking"] == "client"


def test_decision_is_stable_across_tool_steps(env):
    client, orch, primary = env
    client.post("/api/chat", json=body("run the tests", think=True), headers={"X-Conversation-Id": "t1"})
    steps = [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "shell"}}]},
             {"role": "tool", "tool_name": "shell", "content": "FAILED test_x - AssertionError"}]
    client.post("/api/chat", json=body("run the tests", think=True, extra=steps), headers={"X-Conversation-Id": "t1"})
    assert primary.requests[-1]["think"] is False          # same turn: same decision despite the error output


def test_chat_endpoint_and_client_mode(env):
    client, orch, primary = env
    client.post("/chat", json={"message": "list the files"})
    assert primary.requests[-1]["think"] is False
    orch.config.thinking.mode = "client"
    client.post("/api/chat", json=body("list the files", think=True))
    assert primary.requests[-1]["think"] is True
