"""Memory retrieval behind a stable interface.

Phase 1-2: keyword scoring over the Markdown stores (no embeddings).
Phase 3 replaces KeywordRetriever with an embedding-backed implementation
of the same interface; nothing else in the app has to change.
"""
from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .markdown_store import MarkdownStore
from .schemas import Category, MemoryEntry

_STOP = set("""
a an and are as at be but by can could did do does for from had has have how i if in into is it its
me my no not of on or our so than that the their them then there these they this to too us was we
were what when where which who why will with would you your yes ok okay please just also still again
de da do das dos e em o os a as um uma para por com que não sim na no nas nos se
""".split())
_TOKEN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:+#-]*")


def tokenize(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):
        t = t.strip(".:-")
        if len(t) < 2 or t in _STOP:
            continue
        out.append(t)
        # Split compound tokens (qwen3:14b -> qwen3, 14b; open-claw -> open, claw) as extra terms.
        parts = [p for p in re.split(r"[.:_+#-]", t) if len(p) >= 2 and p not in _STOP]
        if len(parts) > 1:
            out.extend(parts)
    return out


@dataclass
class ScoredEntry:
    entry: MemoryEntry
    score: float


class MemoryRetriever(ABC):
    @abstractmethod
    def search(self, query: str, limit: int, categories: list[Category] | None = None,
               include_inactive: bool = False, query_vec=None) -> list[ScoredEntry]:
        ...


class KeywordRetriever(MemoryRetriever):
    def __init__(self, stores: dict[Category, MarkdownStore]):
        self.stores = stores

    def _corpus(self, categories, include_inactive) -> list[MemoryEntry]:
        out: list[MemoryEntry] = []
        for cat in categories or list(Category):
            out.extend(self.stores[cat].entries(active_only=not include_inactive))
        return out

    def search(self, query: str, limit: int, categories: list[Category] | None = None,
               include_inactive: bool = False, query_vec=None) -> list[ScoredEntry]:
        entries = self._corpus(categories, include_inactive)
        q_terms = set(tokenize(query))
        if not q_terms or not entries:
            return []
        docs = [(e, tokenize(e.title), tokenize(e.content)) for e in entries]
        n = len(docs)
        df: dict[str, int] = {}
        for _, t, c in docs:
            for term in set(t) | set(c):
                df[term] = df.get(term, 0) + 1
        scored: list[ScoredEntry] = []
        for e, t_terms, c_terms in docs:
            t_set, c_set = set(t_terms), set(c_terms)
            score = 0.0
            for term in q_terms:
                if term not in t_set and term not in c_set:
                    continue
                idf = math.log(1 + n / df.get(term, 1))
                score += idf * (2.0 if term in t_set else 0.0) + idf * (1.0 if term in c_set else 0.0)
            if score > 0:
                # Mild length normalisation so long entries do not dominate.
                score /= math.sqrt(1 + len(c_terms) / 50)
                if not e.active:
                    score *= 0.5
                scored.append(ScoredEntry(e, round(score, 4)))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:limit]


class HybridRetriever(KeywordRetriever):
    """Keyword + vector search merged with reciprocal rank fusion (RRF).

    Keywords are strong on what memory is full of (ports, error codes, model names,
    paths); vectors catch paraphrases and other languages. An entry found only by the
    vector side must clear `min_similarity`, so semantic near-misses can't pad the prompt.
    Without a query vector this is exactly the keyword retriever.
    """
    RRF_K = 60

    def __init__(self, stores: dict[Category, MarkdownStore], index, min_similarity: float = 0.35):
        super().__init__(stores)
        self.index = index
        self.min_similarity = min_similarity

    def search(self, query: str, limit: int, categories: list[Category] | None = None,
               include_inactive: bool = False, query_vec=None) -> list[ScoredEntry]:
        kw = super().search(query, limit * 3, categories, include_inactive)
        if query_vec is None or self.index is None:
            return kw[:limit]
        entries = {e.entry_id: e for e in self._corpus(categories, include_inactive)}
        vec = [(k, s) for k, s in self.index.search("memory", query_vec, limit * 3, allowed=set(entries))]
        kw_ids = {s.entry.entry_id for s in kw}
        fused: dict[str, float] = {}
        for rank, s in enumerate(kw):
            fused[s.entry.entry_id] = fused.get(s.entry.entry_id, 0) + 1 / (self.RRF_K + rank + 1)
        for rank, (k, sim) in enumerate(vec):
            if k not in kw_ids and sim < self.min_similarity:
                continue
            fused[k] = fused.get(k, 0) + 1 / (self.RRF_K + rank + 1)
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        return [ScoredEntry(entries[k], round(score * 1000, 3)) for k, score in ranked if k in entries]
