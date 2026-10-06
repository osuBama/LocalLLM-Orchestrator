"""Tool-result compression (stepped, prompt-cache friendly).

In agent sessions the context fills up with tool output: file contents, logs,
command output, re-sent on every request. After a turn ends, the memory model
writes a short digest of each large tool result (stored by content hash).
Results from older user turns are then sent to the primary as digests instead
of verbatim.

Rules that protect accuracy:
  * the current turn and the last `keep_recent_user_turns` turns are never touched;
  * a result is only replaced if its digest already exists and is usable
    (actually shorter); otherwise it stays verbatim;
  * every digest is labelled as such and tells the model it can re-run the tool.

Rules that protect the prompt cache:
  * the compression boundary only moves when the trim point moves (or every
    `step_turns` turns before trimming starts), and
  * the set of results to compress is frozen when the boundary moves, so a
    digest finishing mid-window does not change the prompt prefix.
"""
from __future__ import annotations

import hashlib

from .util import estimate_tokens

MARKER = "[compressed tool output: original ~{n} tokens; re-run the tool if exact output is needed]"


def result_hash(content: str) -> str:
    return hashlib.sha1(content.encode("utf-8", "replace")).hexdigest()


def compress_boundary(user_turns: int, drop: int, *, trim_keep: int, keep_recent: int,
                      step: int, trimming: bool) -> int:
    """Highest user-turn number whose tool results may be compressed (0 = none).

    Once trimming is active, the boundary is tied to the trim point
    (drop + trim_keep - keep_recent), so it moves on exactly the same turns and
    the cache is invalidated once per step, not twice. Before that, it follows
    its own step schedule.
    """
    if trimming and drop > 0:
        b = drop + (trim_keep - keep_recent)
    else:
        b = ((user_turns - keep_recent) // step) * step if user_turns > keep_recent else 0
    return max(0, min(b, user_turns - keep_recent))


def tool_results_by_turn(messages: list[dict], first_turn: int) -> list[tuple[int, int, str]]:
    """(message index, user-turn number, content) for each tool result in `messages`.

    `first_turn` is the turn number of the first user message present
    (drop + 1 when history has been trimmed).
    """
    out = []
    turn = first_turn - 1
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "user":
            turn += 1
        elif role == "tool" and isinstance(m.get("content"), str) and turn >= first_turn:
            out.append((i, turn, m["content"]))
    return out


def candidates(messages: list[dict], first_turn: int, boundary: int, min_tokens: int,
               never_tools: set[str]) -> list[tuple[int, str]]:
    """(message index, hash) of results old and large enough to compress."""
    out = []
    for i, turn, content in tool_results_by_turn(messages, first_turn):
        if turn > boundary:
            continue
        if messages[i].get("tool_name") in never_tools or messages[i].get("name") in never_tools:
            continue
        if estimate_tokens(content) < min_tokens:
            continue
        out.append((i, result_hash(content)))
    return out


def apply(messages: list[dict], first_turn: int, boundary: int, frozen: dict[str, str],
          min_tokens: int, never_tools: set[str]) -> tuple[list[dict], int, int]:
    """Replace frozen results with their digests. Returns (messages, count, tokens_saved)."""
    if boundary <= 0 or not frozen:
        return messages, 0, 0
    out = list(messages)
    count = saved = 0
    for i, h in candidates(messages, first_turn, boundary, min_tokens, never_tools):
        digest = frozen.get(h)
        if digest is None:
            continue
        original = messages[i]["content"]
        new = MARKER.format(n=estimate_tokens(original)) + "\n" + digest
        out[i] = {**messages[i], "content": new}
        count += 1
        saved += estimate_tokens(original) - estimate_tokens(new)
    return out, count, saved
