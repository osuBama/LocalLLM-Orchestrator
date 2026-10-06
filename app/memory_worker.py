"""Background memory maintenance (spec §24-25).

Tasks are persisted in SQLite so they survive restarts, processed one at a
time, and retried with exponential backoff. Nothing here can affect a
response that has already been returned to the user.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

from .config import Config
from .context_builder import ContextBuilder, sanitize_memory_text
from .flags import remove_flag_tags
from .database import Database
from .memory_manager import AppliedChange, MemoryManager
from .memory_validator import MemoryValidator
from .ollama_client import OllamaClient, OllamaError
from .schemas import MEMORY_CHANGE_JSON_SCHEMA, InteractionTask
from .triggers import should_extract
from .compression import result_hash
from .util import estimate_tokens, head_tail, truncate_tokens

SUMMARY_PRIORITY = 10  # summaries jump the queue: they must be ready for the next prompt
DIGEST_PRIORITY = 5    # digests next: compression can only use what exists
DIGEST_INPUT_TOKENS = 3000


def format_tool_events(events: list[dict], budget_tokens: int) -> str:
    lines = []
    for ev in events[:20]:
        if ev.get("type") == "tool_call":
            args = json.dumps(ev.get("arguments"), ensure_ascii=False, default=str)
            lines.append(f"- call {ev.get('tool')}: {truncate_tokens(args, 80)}")
        else:
            res = ev.get("result")
            res = res if isinstance(res, str) else json.dumps(res, ensure_ascii=False, default=str)
            lines.append(f"- result {ev.get('tool')}: {head_tail(res or '', 150)}")
    return truncate_tokens("\n".join(lines), max(100, budget_tokens)) if lines else "(none)"

log = logging.getLogger("memory")


class MemoryParseError(Exception):
    pass


class MemoryWorker:
    def __init__(self, config: Config, db: Database, manager: MemoryManager,
                 client: OllamaClient, context_builder: ContextBuilder, metrics=None):
        self.config = config
        self.db = db
        self.manager = manager
        self.client = client
        self.context_builder = context_builder
        self.validator = MemoryValidator(config.memory.max_entry_chars, config.memory.min_confidence)
        self.system_prompt = config.prompt("memory_extractor.txt")
        self.summary_prompt = config.prompt("session_summary.txt")
        self.digest_prompt = config.prompt("tool_digest.txt")
        self.metrics = metrics
        self._task: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self._stopping = False

    # -------------------------------------------------------------- queue
    def enqueue(self, task: InteractionTask, force: bool = False) -> tuple[int, bool, list[str]]:
        """Returns (task_id, queued, reasons). Non-triggering tasks are stored as 'skipped'."""
        run, reasons = (True, ["forced"]) if force else should_extract(task, self.config.memory.trigger_mode)
        status = "pending" if run else "skipped"
        task_id = self.db.enqueue_task(task.to_dict(), status=status,
                                       detail=None if run else ",".join(reasons))
        log.info("memory task " + ("queued" if run else "skipped"),
                 extra={"conversation_id": task.conversation_id, "task_id": task_id,
                        "detail": ",".join(reasons)})
        if run:
            self._wake.set()
        return task_id, run, reasons

    def enqueue_summary(self, task: InteractionTask) -> int | None:
        if not self.config.session.summaries_enabled or not task.turn_number:
            return None
        task_id = self.db.enqueue_task(task.to_dict(), kind="summary", priority=SUMMARY_PRIORITY)
        self._wake.set()
        return task_id

    def enqueue_digests(self, task: InteractionTask) -> int:
        """Queue a digest for every large tool result in the turn that doesn't have one yet."""
        cc = self.config.compression
        if not cc.enabled:
            return 0
        pending: dict[str, dict] = {}
        last_args: dict[str, object] = {}
        for ev in task.tool_events:
            if ev.get("type") == "tool_call":
                last_args[ev.get("tool") or "?"] = ev.get("arguments")
                continue
            content = ev.get("result")
            tool = ev.get("tool") or "?"
            if not isinstance(content, str) or tool in cc.never_compress_tools:
                continue
            if estimate_tokens(content) < cc.min_result_tokens:
                continue
            pending.setdefault(result_hash(content), {
                "kind": "digest", "conversation_id": task.conversation_id, "project_id": task.project_id,
                "tool": tool, "arguments": last_args.get(tool), "result": content,
                "user_message": truncate_tokens(task.user_message, 150)})
        if not pending:
            return 0
        known = self.db.get_digests(list(pending))
        n = 0
        for h, payload in pending.items():
            if h in known:
                continue
            self.db.enqueue_task({**payload, "hash": h}, kind="digest", priority=DIGEST_PRIORITY)
            n += 1
        if n:
            self._wake.set()
        return n

    # ------------------------------------------------------------ running
    def start(self) -> None:
        if self._task is None:
            n = self.db.reset_stale_tasks()
            if n:
                log.warning("re-queued stale tasks", extra={"detail": str(n)})
            self._stopping = False
            self._task = asyncio.create_task(self._loop(), name="memory-worker")

    async def stop(self) -> None:
        self._stopping = True
        self._wake.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _loop(self) -> None:
        while not self._stopping:
            try:
                did = await self.process_next()
            except Exception:  # never let the loop die
                log.exception("memory worker loop error")
                did = False
            if not did:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=self.config.memory.worker_poll_seconds)
                except asyncio.TimeoutError:
                    pass

    async def process_next(self) -> bool:
        claimed = self.db.claim_next_task(time.time())
        if claimed is None:
            return False
        task_id = claimed["id"]
        t0 = time.perf_counter()
        if claimed.get("kind") == "digest":
            try:
                await self.process_digest(claimed["payload"])
            except Exception as e:
                if not isinstance(e, (OllamaError, MemoryParseError)):
                    log.exception("unexpected digest task error", extra={"task_id": task_id})
                self._fail(claimed, e, time.perf_counter() - t0)
                return True
            self.db.finish_task(task_id, "done", duration=time.perf_counter() - t0)
            return True
        task = InteractionTask.from_dict(claimed["payload"])
        if claimed.get("kind") == "summary":
            try:
                await self.process_summary(task)
            except Exception as e:
                if not isinstance(e, (OllamaError, MemoryParseError)):
                    log.exception("unexpected summary task error", extra={"task_id": task_id})
                self._fail(claimed, e, time.perf_counter() - t0)
                return True
            self.db.finish_task(task_id, "done", duration=time.perf_counter() - t0)
            return True
        try:
            applied = await self.process(task, task_id)
        except (OllamaError, MemoryParseError) as e:
            self._fail(claimed, e, time.perf_counter() - t0)
            return True
        except Exception as e:
            log.exception("unexpected memory task error", extra={"task_id": task_id})
            self._fail(claimed, e, time.perf_counter() - t0)
            return True
        dur = time.perf_counter() - t0
        self.db.finish_task(task_id, "done", duration=dur)
        if self.metrics:
            self.metrics.record_memory(dur, "done", len([a for a in applied if a.status == "approved"]))
        return True

    def _fail(self, claimed: dict, err: Exception, dur: float) -> None:
        attempts = claimed["attempts"]
        max_attempts = self.config.memory.max_attempts
        msg = f"{type(err).__name__}: {err}"[:1000]
        if attempts >= max_attempts:
            self.db.finish_task(claimed["id"], "failed", msg, dur)
            status = "failed"
        else:
            delay = self.config.memory.retry_base_seconds * (2 ** (attempts - 1))
            self.db.finish_task(claimed["id"], "pending", msg, dur, next_attempt_at=time.time() + delay)
            status = "retry"
        log.warning("memory task " + status, extra={"task_id": claimed["id"], "detail": msg,
                                                   "conversation_id": claimed.get("conversation_id")})
        if self.metrics:
            self.metrics.record_memory(dur, status, 0)

    # ---------------------------------------------------------- pipeline
    def build_messages(self, task: InteractionTask) -> list[dict]:
        cfg = self.config.memory
        existing = self.context_builder.build(
            f"{task.user_message}\n{task.assistant_response}", max_tokens=cfg.extractor_memory_tokens)
        budget = cfg.extractor_interaction_tokens
        user = head_tail(task.user_message, budget // 3)
        assistant = head_tail(task.assistant_response, budget // 2)
        tool_text = format_tool_events(task.tool_events, budget // 6)
        if task.flags:
            flag_text = "\n".join(f"- [{f.get('category') or 'uncategorised'}] {f.get('text', '')}"
                                  for f in task.flags)
        else:
            flag_text = "(none)"
        body = (
            "EXISTING MEMORY (ids can be used as target_id):\n"
            f"{existing.text or '(empty)'}\n\n"
            "NEW INTERACTION:\n"
            f"USER:\n{user}\n\n"
            f"ASSISTANT:\n{assistant}\n\n"
            f"TOOL EVENTS:\n{tool_text}\n\n"
            f"PRIMARY MODEL FLAGS (hints to verify, not facts):\n{flag_text}\n\n"
            "Return the JSON object now."
        )
        return [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": body}]

    async def process(self, task: InteractionTask, task_id: int | None = None) -> list[AppliedChange]:
        messages = self.build_messages(task)
        resp = await self.client.chat(messages, format=MEMORY_CHANGE_JSON_SCHEMA,
                                      options={"temperature": 0.1})
        raw = resp["message"]["content"]
        result = self.validator.validate_text(raw, self.manager.existing())
        if not result.parsed:
            # Spec §22: log, do not modify memory, keep the raw interaction (it is in JSONL).
            log.warning("memory model output rejected", extra={
                "task_id": task_id, "conversation_id": task.conversation_id,
                "detail": f"{result.error}; raw={raw[:300]!r}"})
            raise MemoryParseError(result.error or "unparseable output")
        applied = self.manager.apply(result, conversation_id=task.conversation_id, task_id=task_id)
        log.info("memory task processed", extra={
            "task_id": task_id, "conversation_id": task.conversation_id,
            "detail": f"accepted={len(result.accepted)} rejected={len(result.rejected)}",
            "memory_latency": resp.get("total_duration", 0) / 1e9})
        return applied

    # ------------------------------------------------------- session summary
    def build_summary_messages(self, task: InteractionTask, previous: str) -> list[dict]:
        max_tokens = self.config.session.summary_max_tokens
        system = self.summary_prompt.replace("{max_words}", str(int(max_tokens * 0.7)))
        body = (
            f"PREVIOUS SUMMARY (turns 1-{max(0, task.turn_number - 1)}):\n{previous or '(empty)'}\n\n"
            f"LATEST TURN (turn {task.turn_number}):\n"
            f"USER:\n{head_tail(task.user_message, 800)}\n\n"
            f"TOOLS:\n{format_tool_events(task.tool_events, 300)}\n\n"
            f"ASSISTANT:\n{head_tail(remove_flag_tags(task.assistant_response), 1000)}\n\n"
            "Write the updated summary now."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": body}]

    async def process_summary(self, task: InteractionTask) -> bool:
        prev = self.db.get_summary(task.conversation_id)
        if prev and prev["covered_turns"] >= task.turn_number:
            return False  # stale (e.g. a retry overtaken by a newer turn)
        max_tokens = self.config.session.summary_max_tokens
        messages = self.build_summary_messages(task, prev["summary"] if prev else "")
        resp = await self.client.chat(messages, options={"temperature": 0.2,
                                                         "num_predict": int(max_tokens * 1.6)})
        text = remove_flag_tags(resp["message"]["content"]).strip()
        if not text:
            raise MemoryParseError("empty session summary")
        text = sanitize_memory_text(truncate_tokens(text, max_tokens))
        stored = self.db.set_summary(task.conversation_id, text, task.turn_number, task.project_id)
        log.info("session summary updated" if stored else "session summary stale",
                 extra={"conversation_id": task.conversation_id, "detail": f"turn={task.turn_number}"})
        return stored

    # ---------------------------------------------------------- tool digests
    async def process_digest(self, payload: dict) -> bool:
        h = payload["hash"]
        if self.db.get_digests([h]):
            return False  # another task already produced it
        content = payload.get("result") or ""
        original_tokens = estimate_tokens(content)
        max_tokens = self.config.compression.digest_max_tokens
        args = json.dumps(payload.get("arguments"), ensure_ascii=False, default=str)
        body = (f"TOOL: {payload.get('tool')}\n"
                f"ARGUMENTS: {truncate_tokens(args, 150)}\n"
                f"WHAT THE USER WAS DOING: {payload.get('user_message') or '(unknown)'}\n\n"
                f"OUTPUT (~{original_tokens} tokens):\n{head_tail(content, DIGEST_INPUT_TOKENS)}\n\n"
                "Write the digest now.")
        system = self.digest_prompt.replace("{max_words}", str(int(max_tokens * 0.7)))
        resp = await self.client.chat([{"role": "system", "content": system},
                                       {"role": "user", "content": body}],
                                      options={"temperature": 0.1, "num_predict": int(max_tokens * 1.6)})
        text = remove_flag_tags(resp["message"]["content"]).strip()
        if not text:
            raise MemoryParseError("empty tool digest")
        raw_tokens = estimate_tokens(text)
        text = sanitize_memory_text(truncate_tokens(text, max_tokens))
        digest_tokens = estimate_tokens(text)
        # Usable only if (a) the model respected the length limit, so nothing was cut off
        # arbitrarily, and (b) it saves at least half. Otherwise the original stays verbatim.
        usable = raw_tokens <= max_tokens * 1.25 and digest_tokens * 2 <= original_tokens
        self.db.set_digest(h, payload.get("tool"), text, original_tokens, digest_tokens, usable)
        log.info("tool digest stored", extra={"conversation_id": payload.get("conversation_id"),
                                              "detail": f"{original_tokens}->{digest_tokens} usable={usable}"})
        return usable
