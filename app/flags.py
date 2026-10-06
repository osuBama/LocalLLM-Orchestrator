"""Memory flags raised by the primary model.

The primary may end a reply with lines such as

    <memory_flag category="lesson">MCP route is /mcp, not /sse</memory_flag>

They are hints for the memory model, never shown to the client. Because
replies stream, a tag can be split across chunks at any character, so the
stripper holds back only the shortest tail that could still be the start of
a tag and releases everything else immediately.

Flag text is model output: it is untrusted and still goes through the
memory model and the validator like everything else.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import Category

OPEN = "<memory_flag"
CLOSE = "</memory_flag>"
_TAG_RE = re.compile(r"<memory_flag\b([^>]*)>(.*?)(?:</memory_flag>|$)", re.S)
_CAT_RE = re.compile(r'category\s*=\s*["\']?([a-z_]+)', re.I)
_VALID = {c.value for c in Category}


@dataclass
class Flag:
    category: str | None
    text: str

    def to_dict(self) -> dict:
        return {"category": self.category, "text": self.text}


def _parse(raw_attrs: str, body: str, max_chars: int) -> Flag | None:
    text = " ".join(body.split())[:max_chars]
    if not text:
        return None
    m = _CAT_RE.search(raw_attrs or "")
    cat = m.group(1).lower() if m else None
    return Flag(cat if cat in _VALID else None, text)


class FlagStripper:
    def __init__(self, max_flags: int = 3, max_chars: int = 300, max_tag_chars: int = 4000):
        self.max_flags = max_flags
        self.max_chars = max_chars
        self.max_tag_chars = max_tag_chars
        self.flags: list[Flag] = []
        self.dropped = 0
        self._pending = ""

    def _add(self, attrs: str, body: str) -> None:
        f = _parse(attrs, body, self.max_chars)
        if f is None:
            return
        if len(self.flags) < self.max_flags:
            self.flags.append(f)
        else:
            self.dropped += 1

    def feed(self, text: str) -> str:
        """Feed streamed text; return what is safe to show the client now."""
        if not text:
            return ""
        self._pending += text
        out: list[str] = []
        while True:
            i = self._pending.find(OPEN)
            if i == -1:
                # Hold back the longest tail that is a prefix of "<memory_flag".
                hold = 0
                for k in range(min(len(OPEN) - 1, len(self._pending)), 0, -1):
                    if OPEN.startswith(self._pending[-k:]):
                        hold = k
                        break
                cut = len(self._pending) - hold
                out.append(self._pending[:cut])
                self._pending = self._pending[cut:]
                break
            # Text before the tag is visible. Trailing whitespace before a flag is
            # trimmed at finish() instead, because more text may still follow.
            out.append(self._pending[:i])
            self._pending = self._pending[i:]
            nxt = self._pending[len(OPEN):len(OPEN) + 1]
            if nxt == "":
                break                      # cannot tell yet; wait for more text
            if not (nxt.isspace() or nxt == ">"):
                out.append(self._pending[0])  # e.g. "<memory_flagged": not our tag
                self._pending = self._pending[1:]
                continue
            j = self._pending.find(CLOSE)
            if j == -1:
                if len(self._pending) > self.max_tag_chars:
                    # Runaway tag: treat it as a flag and stop holding it.
                    m = _TAG_RE.match(self._pending)
                    if m:
                        self._add(m.group(1), m.group(2))
                    self._pending = ""
                break
            tag = self._pending[: j + len(CLOSE)]
            m = _TAG_RE.match(tag)
            if m:
                self._add(m.group(1), m.group(2))
            self._pending = self._pending[j + len(CLOSE):]
        return "".join(out)

    def finish(self) -> str:
        """End of stream: an unclosed tag is still a flag; any other tail is visible."""
        rest, self._pending = self._pending, ""
        if rest.startswith(OPEN):
            m = _TAG_RE.match(rest)
            if m:
                self._add(m.group(1), m.group(2))
                return ""
        return rest


def strip_flags(text: str, max_flags: int = 3, max_chars: int = 300) -> tuple[str, list[Flag]]:
    s = FlagStripper(max_flags, max_chars)
    visible = s.feed(text) + s.finish()
    return (visible.rstrip() if s.flags else visible), s.flags


def remove_flag_tags(text: str) -> str:
    """For history and summaries: drop any flag tags that slipped through."""
    return _TAG_RE.sub("", text or "").rstrip()
