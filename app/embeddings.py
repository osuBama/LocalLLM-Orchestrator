"""Embeddings for semantic memory retrieval and history recall.

Vectors live in the existing SQLite database (no separate vector DB): at the
scale of one person's memory and history, exact brute-force cosine search over
a numpy matrix takes milliseconds.

Request path: only the *query* is embedded, once per user turn, with a short
timeout. On timeout or error retrieval falls back to keywords. Document vectors
(memory entries, history chunks) are built by the background worker.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import threading
from collections import OrderedDict

import httpx
import numpy as np

from .flags import remove_flag_tags
from .util import estimate_tokens, head_tail

log = logging.getLogger("memory")

# Recommended task prefixes for models that are trained with them.
_KNOWN_PREFIXES = {
    "nomic-embed-text": ("search_query: ", "search_document: "),
    "mxbai-embed-large": ("Represent this sentence for searching relevant passages: ", ""),
}

RECALL_CUES = re.compile(
    r"\b(last time|earlier|before|previous(ly)?|remember|we (did|had|tried|discussed|decided|fixed)|"
    r"again|that (error|bug|issue|thing)|the other day|last (week|month)|yesterday|"
    r"da última vez|antes|lembras|já (tivemos|fizemos)|outra vez|ontem|semana passada)\b", re.I)


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]


class Embedder:
    def __init__(self, base_url: str, model: str, *, query_prefix: str | None = None,
                 document_prefix: str | None = None, timeout: float = 3.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.model = model
        base = model.split(":")[0]
        q, d = _KNOWN_PREFIXES.get(base, ("", ""))
        self.query_prefix = q if query_prefix is None else query_prefix
        self.document_prefix = d if document_prefix is None else document_prefix
        self.timeout = timeout
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), transport=transport,
                                         timeout=httpx.Timeout(120.0, connect=5.0))
        self._qcache: OrderedDict[str, np.ndarray] = OrderedDict()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _embed(self, texts: list[str]) -> np.ndarray:
        r = await self._client.post("/api/embed", json={"model": self.model, "input": texts, "keep_alive": "30m"})
        r.raise_for_status()
        vecs = np.asarray(r.json()["embeddings"], dtype=np.float32)
        if vecs.ndim != 2 or len(vecs) != len(texts):
            raise ValueError("unexpected /api/embed response shape")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-12)

    async def documents(self, texts: list[str], batch: int = 32) -> np.ndarray:
        out = [await self._embed([self.document_prefix + t for t in texts[i:i + batch]])
               for i in range(0, len(texts), batch)]
        return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)

    async def query(self, text: str) -> np.ndarray | None:
        """Query vector, or None if the embedder is slow/unavailable (callers fall back to keywords)."""
        key = text_hash(text)
        if key in self._qcache:
            self._qcache.move_to_end(key)
            return self._qcache[key]
        try:
            v = (await asyncio.wait_for(self._embed([self.query_prefix + text]), self.timeout))[0]
        except Exception as e:
            log.warning("query embedding unavailable, keyword search only", extra={"detail": str(e)[:200]})
            return None
        self._qcache[key] = v
        while len(self._qcache) > 256:
            self._qcache.popitem(last=False)
        return v


class VectorIndex:
    """Normalised vectors per kind, cached in memory as one matrix, reloaded when written."""

    def __init__(self, db, model: str):
        self.db = db
        self.model = model
        self._cache: dict[str, tuple[list[str], np.ndarray]] = {}
        self._lock = threading.Lock()

    def invalidate(self, kind: str | None = None) -> None:
        with self._lock:
            if kind:
                self._cache.pop(kind, None)
            else:
                self._cache.clear()

    def _matrix(self, kind: str) -> tuple[list[str], np.ndarray]:
        with self._lock:
            if kind not in self._cache:
                rows = self.db.load_vectors(kind, self.model)
                if rows:
                    dim = rows[0][2]
                    rows = [r for r in rows if r[2] == dim]
                    keys = [r[0] for r in rows]
                    mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), dim)
                else:
                    keys, mat = [], np.zeros((0, 0), dtype=np.float32)
                self._cache[kind] = (keys, mat)
            return self._cache[kind]

    def search(self, kind: str, qvec: np.ndarray, top: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        keys, mat = self._matrix(kind)
        if not keys or qvec is None or mat.shape[1] != qvec.shape[0]:
            return []
        sims = mat @ qvec
        order = np.argsort(-sims)
        out = []
        for i in order:
            k = keys[i]
            if allowed is not None and k not in allowed:
                continue
            out.append((k, float(sims[i])))
            if len(out) >= top:
                break
        return out

    def count(self, kind: str) -> int:
        return len(self._matrix(kind)[0])


def history_chunk_text(user: str, assistant: str, tool_events: list[dict], max_chars: int) -> str:
    tools = []
    for ev in tool_events[:6]:
        if ev.get("type") == "tool_result":
            res = ev.get("result")
            res = res if isinstance(res, str) else str(res)
            tools.append(f"[{ev.get('tool')}] {head_tail(res, 60)}")
    budget_tok = max(50, max_chars // 4)
    parts = [f"USER: {head_tail(user, budget_tok // 2)}"]
    if tools:
        parts.append("TOOLS: " + " | ".join(tools))
    parts.append(f"ASSISTANT: {head_tail(remove_flag_tags(assistant), budget_tok // 2)}")
    return "\n".join(parts)[:max_chars]


class Indexer:
    """Keeps memory-entry and history-chunk vectors in sync (runs in the background worker)."""

    def __init__(self, orch, embedder: Embedder):
        self.orch = orch
        self.embedder = embedder
        self.index = VectorIndex(orch.db, embedder.model)

    def _memory_docs(self) -> dict[str, str]:
        docs = {}
        for store in self.orch.manager.stores.values():
            for e in store.entries(active_only=True):
                docs[e.entry_id] = f"{e.title}: {e.content}"
        return docs

    async def sync_memory(self) -> int:
        docs = self._memory_docs()
        have = self.orch.db.vector_hashes("memory", self.embedder.model)
        stale = [k for k, t in docs.items() if have.get(k) != text_hash(t)]
        gone = [k for k in have if k not in docs]
        if gone:
            self.orch.db.delete_vectors("memory", self.embedder.model, gone)
        if stale:
            vecs = await self.embedder.documents([docs[k] for k in stale])
            self.orch.db.upsert_vectors("memory", self.embedder.model, [
                (k, vecs[i].astype(np.float32).tobytes(), vecs.shape[1], text_hash(docs[k]))
                for i, k in enumerate(stale)])
        if stale or gone:
            self.index.invalidate("memory")
        return len(stale)

    def add_history(self, conversation_id: str, turn: int, user: str, assistant: str,
                    tool_events: list[dict]) -> str | None:
        cfg = self.orch.config.history_recall
        if not (cfg.enabled and user and assistant and turn):
            return None
        key = f"{conversation_id}:{turn}"
        text = history_chunk_text(user, assistant, tool_events, cfg.chunk_chars)
        self.orch.db.add_history_chunk(key, conversation_id, turn, text)
        return key

    async def sync_history(self, limit: int = 256) -> int:
        chunks = self.orch.db.history_chunks()
        have = self.orch.db.vector_hashes("history", self.embedder.model)
        todo = [k for k, c in chunks.items() if have.get(k) != text_hash(c["text"])][:limit]
        if not todo:
            return 0
        vecs = await self.embedder.documents([chunks[k]["text"] for k in todo])
        self.orch.db.upsert_vectors("history", self.embedder.model, [
            (k, vecs[i].astype(np.float32).tobytes(), vecs.shape[1], text_hash(chunks[k]["text"]))
            for i, k in enumerate(todo)])
        self.index.invalidate("history")
        return len(todo)

    def backfill_history_from_logs(self) -> int:
        """Create chunks for every recorded turn in raw JSONL (embedding happens in sync_history)."""
        counters: dict[str, int] = {}
        n = 0
        for t in self.orch.conv_log.iter_interactions():
            cid = t["conversation_id"]
            counters[cid] = counters.get(cid, 0) + 1
            if self.add_history(cid, counters[cid], t.get("user_message", ""), t.get("assistant_response", ""),
                                t.get("tool_events") or []):
                n += 1
        return n

    # ---------------------------------------------------------------- recall
    def recall(self, query: str, qvec: np.ndarray | None, conversation_id: str,
               in_prompt_after_turn: int) -> list[dict]:
        """Older exchanges relevant to `query` that are NOT already in the prompt.

        Chunks of the current conversation with turn > in_prompt_after_turn are still
        verbatim in the prompt and are skipped.
        """
        cfg = self.orch.config.history_recall
        if not cfg.enabled or qvec is None:
            return []
        threshold = cfg.cue_min_similarity if RECALL_CUES.search(query or "") else cfg.min_similarity
        hits = self.index.search("history", qvec, cfg.top_k * 4)
        hits = [(k, s) for k, s in hits if s >= threshold]
        if not hits:
            return []
        chunks = self.orch.db.history_chunks([k for k, _ in hits])
        out = []
        for k, sim in hits:
            c = chunks.get(k)
            if not c:
                continue
            if c["conversation_id"] == conversation_id and c["turn"] > in_prompt_after_turn:
                continue
            out.append({"key": k, "conversation_id": c["conversation_id"], "turn": c["turn"],
                        "date": (c["created_at"] or "")[:10], "similarity": round(sim, 3), "text": c["text"]})
            if len(out) >= cfg.top_k:
                break
        # Respect the token cap.
        kept, used = [], 0
        for h in out:
            cost = estimate_tokens(h["text"]) + 20
            if used + cost > cfg.max_tokens:
                break
            kept.append(h)
            used += cost
        return kept
