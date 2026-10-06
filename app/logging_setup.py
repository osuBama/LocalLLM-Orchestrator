"""Structured JSON logs: orchestrator.log, primary.log, memory.log (+ console)."""
from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

_STD = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "timestamp": datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for k, v in vars(record).items():
            if k not in _STD and not k.startswith("_"):
                out[k] = v
        if record.exc_info:
            out["exception"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False, default=str)


def setup_logging(logs_dir: Path, level: str = "INFO", console: bool = True) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    fmt = JsonFormatter()
    for name, fname in (("orchestrator", "orchestrator.log"), ("primary", "primary.log"),
                        ("memory", "memory.log")):
        lg = logging.getLogger(name)
        lg.setLevel(level.upper())
        lg.propagate = False
        for h in list(lg.handlers):
            lg.removeHandler(h)
            h.close()
        fh = logging.handlers.RotatingFileHandler(logs_dir / fname, maxBytes=20_000_000,
                                                  backupCount=5, encoding="utf-8")
        fh.setFormatter(fmt)
        lg.addHandler(fh)
        if console:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
            lg.addHandler(ch)
