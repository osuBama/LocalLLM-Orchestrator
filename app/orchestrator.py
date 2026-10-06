"""Service container and the primary request flow (spec §15).

Used by both entry points:
  * POST /chat          - the spec's own API
  * POST /api/chat      - Ollama-compatible proxy for OpenClaw (see proxy.py)
"""
from __future__ import annotations

import logging
import time
import uuid
from collections import deque

import httpx

from .config import Config
from .context_builder import ContextBuilder, wrap_user_request
from .flags import strip_flags
from .conversation_logger import ConversationLogger
from .database import Database
from .memory_manager import MemoryManager
from .memory_retriever import KeywordRetriever
from .memory_worker import MemoryWorker
from .metrics import Metrics
from .ollama_client import OllamaClient, OllamaError
from .schemas import InteractionTask
from .util import estimate_tokens, now_iso

log = logging.getLogger("orchestrator")
plog = logging.getLogger("primary")


class Orchestrator:
    def __init__(self, config: Config, *, primary_transport: httpx.AsyncBaseTransport | None = None,
                 memory_transport: httpx.AsyncBaseTransport | None = None):
        config.ensure_dirs()
        self.config = config
        self.project_id = config.memory.project_id
        self.db = Database(config.database_path)
        self.conv_log = ConversationLogger(config.conversations_dir, config.conversation.retain_raw_history)
        self.manager = MemoryManager(config, self.db)
        self.manager.ensure_files()
        self.retriever = KeywordRetriever(self.manager.stores)
        self.context_builder = ContextBuilder(
            self.manager.stores, self.retriever, config.prompt("context_builder.txt"),
            config.memory.max_context_tokens, config.memory.max_entry_tokens)
        self.primary = OllamaClient(config.ollama.primary, transport=primary_transport)
        self.memory_client = OllamaClient(config.ollama.memory, transport=memory_transport)
        self.metrics = Metrics()
        self.worker = MemoryWorker(config, self.db, self.manager, self.memory_client,
                                   self.context_builder, self.metrics)
        self.primary_system = config.prompt("primary_system.txt")
        if config.flags.enabled:
            self.primary_system += "\n\n" + config.prompt("primary_flags.txt")
        self._recent: dict[str, deque] = {}
        self._turns: dict[str, int] = {}

    async def aclose(self) -> None:
        await self.worker.stop()
        await self.primary.aclose()
        await self.memory_client.aclose()

    # ------------------------------------------------------------ memory
    async def queue_memory(self, task: InteractionTask) -> bool:
        """Queue the session-summary update and (if triggered) memory extraction."""
        self.worker.enqueue_summary(task)
        self.worker.enqueue_digests(task)
        _, queued, _ = self.worker.enqueue(task)
        if not self.config.memory.asynchronous_updates:
            # Synchronous mode (debugging): process inline, still isolated from errors.
            try:
                while await self.worker.process_next():
                    pass
            except Exception:
                log.exception("inline memory processing failed")
        return queued

    def session_summary(self, conversation_id: str) -> dict | None:
        if not self.config.session.summaries_enabled:
            return None
        return self.db.get_summary(conversation_id)

    def record_turn(self, *, conversation_id: str, user_message: str, assistant_response: str,
                    tool_events: list[dict], source: str, log_user: bool = True,
                    flags: list[dict] | None = None, turn_number: int = 0) -> InteractionTask:
        """Persist a finished turn to raw history and build the memory task."""
        self.db.touch_conversation(conversation_id, self.project_id)
        if log_user:
            self.conv_log.message(conversation_id, "user", user_message, project_id=self.project_id)
        for ev in tool_events:
            if ev.get("type") == "tool_call":
                self.conv_log.tool_call(conversation_id, ev.get("tool", "?"), ev.get("arguments"),
                                        project_id=self.project_id)
            elif ev.get("type") == "tool_result":
                self.conv_log.tool_result(conversation_id, ev.get("tool", "?"), ev.get("result"),
                                          project_id=self.project_id)
        extra = {"memory_flags": flags} if flags else {}
        self.conv_log.message(conversation_id, "assistant", assistant_response,
                              project_id=self.project_id, **extra)
        return InteractionTask(conversation_id=conversation_id, timestamp=now_iso(),
                               user_message=user_message, assistant_response=assistant_response,
                               tool_events=tool_events, project_id=self.project_id, source=source,
                               flags=list(flags or []), turn_number=turn_number)

    # -------------------------------------------------------------- /chat
    async def chat(self, message: str, conversation_id: str | None = None) -> dict:
        request_id = uuid.uuid4().hex[:12]
        conversation_id = conversation_id or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        self.db.touch_conversation(conversation_id, self.project_id)
        self.conv_log.message(conversation_id, "user", message, project_id=self.project_id,
                              request_id=request_id)

        if conversation_id not in self._turns:
            prev = self.db.get_summary(conversation_id)
            self._turns[conversation_id] = prev["covered_turns"] if prev else 0
        self._turns[conversation_id] += 1
        turn_number = self._turns[conversation_id]
        summary = self.session_summary(conversation_id)
        # /chat never resends full history, so the summary is always useful once it exists.
        ctx = self.context_builder.build(message, session_summary=summary["summary"] if summary else None)
        messages = [{"role": "system", "content": self.primary_system}]
        recent = self._recent.setdefault(conversation_id, deque(maxlen=max(1, self.config.conversation.recent_turns) * 2))
        if self.config.conversation.recent_turns:
            messages.extend(recent)
        messages.append({"role": "user", "content": wrap_user_request(ctx.text, message)})

        rec = {"request_id": request_id, "conversation_id": conversation_id, "mode": "chat",
               "primary_model": self.config.ollama.primary.model,
               "memory_model": self.config.ollama.memory.model,
               "memory_retrieval_count": len(ctx.included), "memory_tokens": ctx.token_estimate,
               "total_context_tokens": sum(estimate_tokens(m["content"]) for m in messages)}
        try:
            resp = await self.primary.chat(messages)
        except OllamaError as e:
            rec.update(error=str(e), total_request_time=round(time.perf_counter() - t0, 3))
            self.metrics.record_request(rec)
            self.conv_log.system_event(conversation_id, "primary_failed", project_id=self.project_id,
                                       request_id=request_id, error=str(e))
            plog.error("primary request failed", extra=rec)
            raise
        answer = resp["message"].get("content", "")
        flags: list[dict] = []
        if self.config.flags.enabled:
            answer, found = strip_flags(answer, self.config.flags.max_per_turn, self.config.flags.max_chars)
            flags = [f.to_dict() for f in found]
        rec["memory_flags"] = len(flags)
        rec.update(Metrics.from_ollama(resp))
        rec["total_request_time"] = round(time.perf_counter() - t0, 3)
        self.metrics.record_request(rec)
        plog.info("primary request", extra=rec)

        recent.append({"role": "user", "content": message})
        recent.append({"role": "assistant", "content": answer})
        task = self.record_turn(conversation_id=conversation_id, user_message=message,
                                assistant_response=answer, tool_events=[], source="chat",
                                log_user=False, flags=flags, turn_number=turn_number)
        queued = await self.queue_memory(task)
        return {"conversation_id": conversation_id, "response": answer, "memory_update_queued": queued,
                "request_id": request_id, "memory_entries_used": ctx.included,
                "memory_flags": flags, "turn": turn_number}
