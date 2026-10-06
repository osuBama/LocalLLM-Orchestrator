"""In-process metrics (spec §34-35), exposed on GET /metrics.

The point is to compare full-history context against retrieved-memory
context, so every primary request records how big the prompt was and how
long prefill and generation took.
"""
from __future__ import annotations

import threading
from collections import deque
from statistics import mean
from typing import Any


class Metrics:
    def __init__(self, keep: int = 500):
        self._lock = threading.Lock()
        self.requests: deque[dict[str, Any]] = deque(maxlen=keep)
        self.memory_runs: deque[dict[str, Any]] = deque(maxlen=keep)
        self.totals = {"requests": 0, "errors": 0, "memory_done": 0, "memory_retry": 0,
                       "memory_failed": 0, "memory_changes_applied": 0}

    @staticmethod
    def from_ollama(resp: dict) -> dict[str, Any]:
        """Pull timing/token fields out of a final Ollama response chunk (durations in ns)."""
        ns = 1e9
        out = {
            "prompt_tokens": resp.get("prompt_eval_count"),
            "generated_tokens": resp.get("eval_count"),
            "prefill_time": (resp.get("prompt_eval_duration") or 0) / ns or None,
            "generation_time": (resp.get("eval_duration") or 0) / ns or None,
            "load_time": (resp.get("load_duration") or 0) / ns or None,
            "ollama_total_time": (resp.get("total_duration") or 0) / ns or None,
        }
        if out["generated_tokens"] and out["generation_time"]:
            out["tokens_per_second"] = round(out["generated_tokens"] / out["generation_time"], 2)
        return out

    def record_request(self, rec: dict[str, Any]) -> None:
        with self._lock:
            self.requests.append(rec)
            self.totals["requests"] += 1
            if rec.get("error"):
                self.totals["errors"] += 1

    def record_memory(self, duration: float, status: str, applied: int) -> None:
        with self._lock:
            self.memory_runs.append({"duration": round(duration, 3), "status": status, "applied": applied})
            key = f"memory_{status}"
            if key in self.totals:
                self.totals[key] += 1
            self.totals["memory_changes_applied"] += applied

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            reqs = [r for r in self.requests if not r.get("error")]

            def avg(key):
                vals = [r[key] for r in reqs if isinstance(r.get(key), (int, float))]
                return round(mean(vals), 3) if vals else None

            by_mode: dict[str, dict] = {}
            for mode in sorted({r.get("mode", "?") for r in reqs}):
                sub = [r for r in reqs if r.get("mode") == mode]
                def a(key, sub=sub):
                    vals = [r[key] for r in sub if isinstance(r.get(key), (int, float))]
                    return round(mean(vals), 3) if vals else None
                by_mode[mode] = {"count": len(sub), "avg_prompt_tokens": a("prompt_tokens"),
                                 "avg_memory_tokens": a("memory_tokens"),
                                 "avg_prefill_time": a("prefill_time"),
                                 "avg_time_to_first_token": a("time_to_first_token"),
                                 "avg_total_request_time": a("total_request_time"),
                                 "avg_tokens_per_second": a("tokens_per_second"),
                                 "avg_tool_tokens_saved": a("tool_tokens_saved")}
            mem = list(self.memory_runs)
            return {
                "totals": dict(self.totals),
                "averages": {
                    "prompt_tokens": avg("prompt_tokens"),
                    "memory_tokens": avg("memory_tokens"),
                    "total_context_tokens": avg("total_context_tokens"),
                    "generated_tokens": avg("generated_tokens"),
                    "prefill_time": avg("prefill_time"),
                    "generation_time": avg("generation_time"),
                    "time_to_first_token": avg("time_to_first_token"),
                    "total_request_time": avg("total_request_time"),
                    "tokens_per_second": avg("tokens_per_second"),
                    "memory_update_time": round(mean(m["duration"] for m in mem), 3) if mem else None,
                },
                "by_mode": by_mode,
                "recent_requests": list(self.requests)[-20:],
            }
