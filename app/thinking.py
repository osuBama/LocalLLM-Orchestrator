"""Per-turn thinking decision for the primary model.

Qwen3-style thinking often spends hundreds to thousands of tokens before the
answer. For clearly simple turns that is pure latency. `auto` mode only ever
turns thinking OFF, and only when a turn is short and shows none of the signs
of real work; everything else keeps what the client asked for.
"""
from __future__ import annotations

import re

_FORCE = re.compile(r"\b(think|step[- ]by[- ]step|reason (it|this) (out|through)|carefully|deeply|"
                    r"pensa|passo a passo|com cuidado)\b", re.I)
_COMPLEX = re.compile(
    r"\b(why|how (does|do|can|would|should|to)|debug|diagnos\w*|design|architect\w*|implement\w*|"
    r"refactor\w*|optimi[sz]\w*|analy[sz]\w*|compare|comparison|plan|strategy|prove|calculat\w*|"
    r"explain|fix|bug|broken|review|migrat\w*|trade-?offs?|investigat\w*|root cause|"
    r"porqu[eê]|por que|como (funciona|fa[cç]o|posso)|desenh\w*|implement\w*|corrig\w*|"
    r"analis\w*|compar\w*|expli\w*|investig\w*)\b", re.I)
_ERRORISH = re.compile(r"(traceback|exception|error|erro|failed|falhou|stack trace|segfault|\bE\d{3,}\b|"
                       r"HTTP [45]\d\d|exit code)", re.I)


def classify(user_text: str, *, simple_max_words: int = 25) -> tuple[bool, str]:
    """(is_simple, reason)."""
    text = user_text or ""
    if _FORCE.search(text):
        return False, "asked to think"
    if "```" in text or "\n    " in text:
        return False, "contains code"
    if _ERRORISH.search(text):
        return False, "error/diagnostic content"
    words = len(text.split())
    if words > simple_max_words:
        return False, f"{words} words"
    m = _COMPLEX.search(text)
    if m:
        return False, f"keyword '{m.group(0).lower()}'"
    return True, "short and simple"


def decide(mode: str, client_think, user_text: str, simple_max_words: int = 25) -> tuple[object, str]:
    """Return (think value to send, or the sentinel KEEP to leave the request as-is; reason)."""
    if mode == "client":
        return KEEP, "client decides"
    if mode == "on":
        return True, "forced on"
    if mode == "off":
        return False, "forced off"
    # auto: never turn thinking ON, only off for simple turns.
    if client_think is False:
        return KEEP, "client disabled thinking"
    simple, why = classify(user_text, simple_max_words=simple_max_words)
    return (False, why) if simple else (KEEP, why)


class _Keep:
    def __repr__(self):
        return "KEEP"


KEEP = _Keep()
