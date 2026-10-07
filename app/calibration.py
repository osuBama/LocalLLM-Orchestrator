"""Learn characters-per-token for each primary model from real Ollama responses.

Ollama reports prompt_eval_count = tokens it actually processed. Cache hits only
make that smaller, never larger, so chars/processed is always >= the true
chars/token, and wildly so for cache hits. Two rules keep this safe:
  * only requests that processed close to the whole prompt are recorded (cache
    hits process far fewer tokens than any plausible ratio allows, and are ignored);
  * the MINIMUM recorded ratio is used. Anything that pushes a ratio down
    (chat-template tokens, tool schemas) only makes the estimate more conservative.
An overestimated ratio would mean underestimated prompts and silent overflow,
which is the one error this must not make.

Used by size-based trimming: a calibrated ratio lets more real history stay in
the window than the fixed ~3.5 chars/token guess, without risking overflow.
"""
from __future__ import annotations

import json
import math
import threading

DEFAULT_CPT = 3.5


class TokenCalibrator:
    def __init__(self, db=None, *, window: int = 60, min_obs: int = 5, cold_fraction: float = 0.6,
                 margin: float = 1.05, bounds: tuple[float, float] = (2.2, 5.5), min_chars: int = 400):
        self.db = db
        self.window, self.min_obs, self.cold_fraction = window, min_obs, cold_fraction
        self.margin, self.bounds, self.min_chars = margin, bounds, min_chars
        self._ratios: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _load(self, model: str) -> list[float]:
        if model not in self._ratios:
            raw = self.db.kv_get(f"calibration:{model}") if self.db else None
            try:
                self._ratios[model] = [float(x) for x in json.loads(raw)][-self.window:] if raw else []
            except (ValueError, TypeError):
                self._ratios[model] = []
        return self._ratios[model]

    def observe(self, model: str, chars: int, processed_tokens: int | None) -> None:
        if not model or not processed_tokens or processed_tokens <= 0 or chars < self.min_chars:
            return
        # Uncached-looking only: a cache hit processes far fewer tokens than the prompt has.
        if processed_tokens < self.cold_fraction * chars / DEFAULT_CPT:
            return
        with self._lock:
            r = self._load(model)
            r.append(round(chars / processed_tokens, 4))
            del r[:-self.window]
            if self.db:
                self.db.kv_set(f"calibration:{model}", json.dumps(r))

    def chars_per_token(self, model: str) -> float:
        with self._lock:
            r = sorted(self._load(model))
        if len(r) < self.min_obs:
            return DEFAULT_CPT
        lo, hi = self.bounds
        return max(lo, min(hi, r[0]))   # minimum: the conservative end

    def tokens(self, model: str, chars: int) -> int:
        if chars <= 0:
            return 0
        return math.ceil(chars / self.chars_per_token(model) * self.margin)

    def snapshot(self) -> dict:
        with self._lock:
            models = list(self._ratios)
        return {m: {"chars_per_token": round(self.chars_per_token(m), 3),
                    "observations": len(self._ratios.get(m, [])),
                    "calibrated": len(self._ratios.get(m, [])) >= self.min_obs} for m in models}
