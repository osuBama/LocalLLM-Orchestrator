import asyncio
import time

import pytest

from app.orchestrator import Orchestrator
from app.schemas import Category, InteractionTask


def task(user="MCP still returns 404 after the fix", assistant="The route is /mcp not /sse; fixed."):
    return InteractionTask("c1", "2026-09-30T12:00:00+01:00", user, assistant)


@pytest.fixture
def orch(cfg, fakes):
    primary, memory = fakes
    o = Orchestrator(cfg, primary_transport=primary.transport, memory_transport=memory.transport)
    yield o, primary, memory
    asyncio.run(o.aclose())


def test_successful_update(orch):
    o, _, memory = orch
    memory.memory_json = {"changes": [{"category": "state", "operation": "add", "title": "MCP endpoint",
                                       "content": "MCP endpoint route is /mcp; 404 resolved.",
                                       "confidence": 0.95, "reason": "Confirmed."}]}
    _, queued, _ = o.worker.enqueue(task())
    assert queued
    assert asyncio.run(o.worker.process_next())
    assert o.manager.stores[Category.state].entries()[0].content.startswith("MCP endpoint route")
    assert o.db.task_counts() == {"done": 1}
    assert o.db.list_changes()[0]["status"] == "approved"
    assert o.db.list_memories()[0]["entry_key"] == "S-001"
    sent = memory.requests[0]
    assert sent["format"]["required"] == ["changes"] and sent["think"] is True   # background: may think


def test_failed_update_retries_then_fails(orch):
    o, _, memory = orch
    memory.fail_status = 500
    o.worker.enqueue(task())
    for _ in range(3):
        assert asyncio.run(o.worker.process_next())
        time.sleep(0.01)
    assert o.db.task_counts() == {"failed": 1}
    assert o.manager.stores[Category.state].entries() == []


def test_retry_is_scheduled_in_future(cfg, fakes):
    cfg.memory.retry_base_seconds = 60
    primary, memory = fakes
    memory.fail_status = 500
    o = Orchestrator(cfg, primary_transport=primary.transport, memory_transport=memory.transport)
    o.worker.enqueue(task())
    asyncio.run(o.worker.process_next())
    t = o.db.list_tasks()[0]
    assert t["status"] == "pending" and t["attempts"] == 1
    assert asyncio.run(o.worker.process_next()) is False  # not due yet
    asyncio.run(o.aclose())


def test_unparseable_output_changes_nothing(orch):
    o, _, memory = orch
    memory.memory_json = "Sure! Here are the changes: ..."
    o.worker.enqueue(task())
    asyncio.run(o.worker.process_next())
    assert all(not s.entries() for s in o.manager.stores.values())
    assert o.db.list_tasks()[0]["status"] == "pending"


def test_trivial_interaction_skipped(orch):
    o, _, _ = orch
    _, queued, reasons = o.worker.enqueue(task(user="thanks!", assistant="You're welcome."))
    assert not queued and o.db.task_counts() == {"skipped": 1}


def test_memory_failure_does_not_affect_primary(orch):
    o, primary, memory = orch
    memory.fail_status = 503
    out = asyncio.run(o.chat("The MCP endpoint failed with 404 again"))
    assert out["response"] == "fake answer" and out["memory_update_queued"]
    asyncio.run(o.worker.process_next())
    assert o.db.list_tasks()[0]["status"] == "pending"


def test_extraction_retries_without_thinking_when_json_breaks(orch):
    o, _, memory = orch
    calls = []
    good = {"changes": [{"category": "lesson", "operation": "add", "title": "Route",
                         "content": "The MCP route is /mcp.", "confidence": 0.9, "reason": "r"}]}
    orig = memory.app.router.routes

    @memory.app.middleware("http")
    async def think_breaks_json(request, call_next):
        body = await request.json() if request.url.path == "/api/chat" else None
        if body is not None:
            calls.append(body.get("think"))
            memory.memory_json = "<think>hmm</think> not json" if body.get("think") else good
        return await call_next(request)
    o.worker.enqueue(task())
    asyncio.run(o.worker.process_next())
    assert calls == [True, False]
    assert o.manager.stores[Category.lesson].entries()[0].content == "The MCP route is /mcp."


def test_summaries_and_digests_never_think(orch):
    o, _, memory = orch
    memory.reply = "summary"
    t = InteractionTask("c1", "t", "u", "a", turn_number=1)
    o.worker.enqueue_summary(t)
    asyncio.run(o.worker.process_next())
    assert memory.requests[-1]["think"] is False
