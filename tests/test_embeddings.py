import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.embeddings import Embedder, RECALL_CUES, history_chunk_text
from app.memory_retriever import HybridRetriever
from app.schemas import Category, InteractionTask


@pytest.fixture
def env(cfg, fakes):
    primary, memory = fakes
    cfg.embeddings.enabled = True
    app = create_app(cfg, primary_transport=primary.transport, memory_transport=memory.transport,
                     console_logs=False)
    orch = app.state.orch
    L = orch.manager.stores[Category.lesson]
    L.add("Orchestrator from WSL", "OpenClaw in WSL fails to connect to the orchestrator unless mirrored networking is on.")
    L.add("Sourdough", "Feed the starter twice a day.")
    orch.manager.stores[Category.environment].add("Memory instance", "The memory model listens on port 11435.")
    with TestClient(app) as client:
        drain(client, orch)                           # startup vector sync
        yield client, orch, primary, memory


def drain(client, orch):
    while client.portal.call(orch.worker.process_next):
        pass


def body(text, history=(), cid=None):
    msgs = []
    for u, a in history:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return {"model": "qwen3:14b", "stream": False, "messages": msgs + [{"role": "user", "content": text}]}


def user_block(primary):
    return primary.requests[-1]["messages"][-1]["content"]


def test_startup_sync_and_hybrid_retriever(env):
    client, orch, primary, memory = env
    assert isinstance(orch.retriever, HybridRetriever)
    assert orch.indexer.index.count("memory") == 3
    h = client.get("/health").json()["embeddings"]
    assert h["memory_vectors"] == 3 and h["query_ok"]


def test_paraphrase_found_that_keywords_miss(env):
    client, orch, primary, memory = env
    orch.retriever.min_similarity = 0.2         # fake embedder scores this paraphrase ~0.24
    q = "connection refused, can't reach it"           # shares no keyword with L-001, only meaning
    from app.memory_retriever import KeywordRetriever
    assert KeywordRetriever.search(orch.retriever, q, 5, categories=[Category.lesson]) == []
    client.post("/api/chat", json=body(q))
    assert "[L-001]" in user_block(primary)
    assert "[L-002]" not in user_block(primary)        # unrelated entry stays out (similarity floor)


def test_falls_back_to_keywords_when_embedder_is_down(env):
    client, orch, primary, memory = env
    memory.embed_fail = True
    client.post("/api/chat", json=body("which port does the memory instance use?"))
    assert "[E-001]" in primary.requests[-1]["messages"][0]["content"] or "[E-001]" in user_block(primary)
    assert client.get("/health").json()["embeddings"]["query_ok"] is False


def test_query_embedded_once_per_turn(env):
    client, orch, primary, memory = env
    before = memory.embed_calls
    client.post("/api/chat", json=body("check the orchestrator logs"), headers={"X-Conversation-Id": "t"})
    steps = [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read"}}]},
             {"role": "tool", "tool_name": "read", "content": "log text"}]
    b = body("check the orchestrator logs")
    b["messages"] += steps
    client.post("/api/chat", json=b, headers={"X-Conversation-Id": "t"})
    assert memory.embed_calls == before + 1


def test_memory_changes_get_embedded(env):
    client, orch, primary, memory = env
    memory.memory_json = {"changes": [{"category": "decision", "operation": "add", "title": "Embedding model",
                                       "content": "Use nomic-embed-text on the memory card.", "confidence": 0.9,
                                       "reason": "decided"}]}
    client.post("/api/chat", json=body("we decided to use nomic-embed-text on the memory GPU"))
    drain(client, orch)
    assert orch.indexer.index.count("memory") == 4


def test_history_recall_across_conversations(env):
    client, orch, primary, memory = env
    old = InteractionTask("old-session", "t", "The gateway kept crashing with error E1234 after the upgrade",
                          "Root cause: the plugin cache was stale; clearing ~/.openclaw/cache fixed E1234.",
                          turn_number=1)
    client.portal.call(orch.queue_memory, old)
    drain(client, orch)
    assert orch.indexer.index.count("history") == 1
    client.post("/api/chat", json=body("last time the gateway crashed with E1234, what fixed it?"),
                headers={"X-Conversation-Id": "new-session"})
    blk = user_block(primary)
    assert "RELATED PAST EXCHANGES" in blk and "clearing ~/.openclaw/cache" in blk and "old-session" in blk
    client.post("/api/chat", json=body("write a haiku about autumn leaves"), headers={"X-Conversation-Id": "x"})
    assert "RELATED PAST EXCHANGES" not in user_block(primary)          # unrelated prompt: nothing recalled


def test_no_recall_of_turns_still_in_the_prompt(env):
    client, orch, primary, memory = env
    for i in (1, 2):
        t = InteractionTask("same", "t", f"gateway error E1234 attempt {i}", f"tried fix {i} for E1234", turn_number=i)
        client.portal.call(orch.queue_memory, t)
    drain(client, orch)
    hist = [("gateway error E1234 attempt 1", "tried fix 1 for E1234"), ("gateway error E1234 attempt 2", "tried fix 2 for E1234")]
    client.post("/api/chat", json=body("gateway error E1234 again?", history=hist), headers={"X-Conversation-Id": "same"})
    assert "RELATED PAST EXCHANGES" not in user_block(primary)          # both turns are verbatim already


def test_recall_cues_and_chunk_text():
    assert RECALL_CUES.search("what did we decide last time?") and RECALL_CUES.search("lembras-te disto?")
    assert not RECALL_CUES.search("write a function")
    t = history_chunk_text("u " * 2000, 'done <memory_flag category="lesson">x</memory_flag>',
                           [{"type": "tool_result", "tool": "read", "result": "r" * 5000}], 1500)
    assert len(t) <= 1500 and "memory_flag" not in t and "[read]" in t


def test_reindex_backfills_history_from_logs(env):
    client, orch, primary, memory = env
    from app.conversation_logger import ConversationLogger
    lg = orch.conv_log
    lg.message("logged", "user", "the NAS backup job failed with rsync code 23")
    lg.message("logged", "assistant", "Code 23 means partial transfer; a file was locked.")
    n = orch.indexer.backfill_history_from_logs()
    assert n == 1
    assert client.portal.call(orch.indexer.sync_history) == 1
    assert orch.indexer.index.count("history") == 1
