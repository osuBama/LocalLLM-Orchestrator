"""Builds the compact <PROJECT_MEMORY> block under a hard token budget.

Budget priority (spec §38): constraints > state > active objectives >
relevant lessons > relevant decisions > relevant environment > relevant
discoveries. Lower-priority material is dropped first; the budget is never
exceeded because a Markdown file happens to be large.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .markdown_store import MarkdownStore
from .memory_retriever import MemoryRetriever
from .schemas import Category, MemoryEntry
from .util import estimate_tokens, truncate_tokens

# (category, always-include-active?, render label)
PRIORITY: list[tuple[Category, bool]] = [
    (Category.constraint, True),
    (Category.state, True),
    (Category.objective, True),
    (Category.lesson, False),
    (Category.decision, False),
    (Category.environment, False),
    (Category.discovery, False),
]

SESSION_LABEL = "SESSION SO FAR (summary of earlier turns no longer shown verbatim)"

RENDER_ORDER: list[tuple[Category, str]] = [
    (Category.state, "CURRENT STATE"),
    (Category.objective, "ACTIVE OBJECTIVES"),
    (Category.constraint, "CONSTRAINTS"),
    (Category.decision, "RELEVANT DECISIONS"),
    (Category.lesson, "RELEVANT LESSONS"),
    (Category.environment, "RELEVANT ENVIRONMENT"),
    (Category.discovery, "RELEVANT DISCOVERIES"),
]

_TAG_RE = re.compile(r"</?\s*(PROJECT_MEMORY|EXTERNAL_MEMORY|USER_REQUEST)\s*>", re.I)


def sanitize_memory_text(text: str) -> str:
    """Hand-edited memory must not be able to close our delimiters."""
    return _TAG_RE.sub(lambda m: m.group(0).replace("<", "&lt;").replace(">", "&gt;"), text)


@dataclass
class BuiltContext:
    text: str
    token_estimate: int
    included: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.included


class ContextBuilder:
    def __init__(self, stores: dict[Category, MarkdownStore], retriever: MemoryRetriever,
                 preamble: str, max_tokens: int, max_entry_tokens: int = 600,
                 relevant_limit: int = 8):
        self.stores = stores
        self.retriever = retriever
        self.preamble = preamble.strip()
        self.max_tokens = max_tokens
        self.max_entry_tokens = max_entry_tokens
        self.relevant_limit = relevant_limit

    def _line(self, e: MemoryEntry) -> str:
        body = truncate_tokens(" ".join(e.content.split()), self.max_entry_tokens)
        return sanitize_memory_text(f"- [{e.entry_id}] {e.title}: {body}")

    def build(self, query: str, max_tokens: int | None = None,
              session_summary: str | None = None) -> BuiltContext:
        budget = max_tokens or self.max_tokens
        header = f"<PROJECT_MEMORY>\n{self.preamble}\n"
        footer = "</PROJECT_MEMORY>"
        # Section headings are paid for up front so the total can never overshoot.
        overhead = estimate_tokens(header + footer) + sum(
            estimate_tokens(f"\n{label}:\n") for _, label in RENDER_ORDER)
        remaining = budget - overhead
        chosen: dict[Category, list[str]] = {c: [] for c, _ in RENDER_ORDER}
        included: list[str] = []
        dropped: list[str] = []

        if remaining <= 0:
            return BuiltContext("", 0, [], ["<budget smaller than overhead>"])

        # The session summary replaces trimmed history, so it is paid for first,
        # capped at half the budget so durable memory always keeps room.
        session_text = ""
        if session_summary and session_summary.strip():
            label = f"\n{SESSION_LABEL}:\n"
            cap = max(0, remaining // 2 - estimate_tokens(label))
            if cap > 20:
                body = sanitize_memory_text(truncate_tokens(session_summary.strip(), cap))
                session_text = label + body + "\n"
                remaining -= estimate_tokens(session_text)
                included.append("SESSION")

        for cat, always in PRIORITY:
            if always:
                candidates = self.stores[cat].entries(active_only=True)
                # Most recently updated first: newest state wins when space is short.
                candidates.sort(key=lambda e: e.updated_at or "", reverse=True)
            else:
                candidates = [s.entry for s in
                              self.retriever.search(query, self.relevant_limit, categories=[cat])]
            for e in candidates:
                line = self._line(e)
                cost = estimate_tokens(line + "\n")
                if cost <= remaining:
                    chosen[cat].append(line)
                    included.append(e.entry_id)
                    remaining -= cost
                else:
                    dropped.append(e.entry_id)

        if not included:
            return BuiltContext("", 0, [], dropped)

        parts = [header]
        if session_text:
            parts.append(session_text)
        for cat, label in RENDER_ORDER:
            if chosen[cat]:
                parts.append(f"\n{label}:\n" + "\n".join(chosen[cat]) + "\n")
        parts.append(footer)
        text = "".join(parts)
        return BuiltContext(text, estimate_tokens(text), included, dropped)


def wrap_user_request(memory_block: str, user_text: str) -> str:
    if not memory_block:
        return user_text
    return f"{memory_block}\n\n<USER_REQUEST>\n{user_text}\n</USER_REQUEST>"
