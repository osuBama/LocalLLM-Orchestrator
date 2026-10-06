"""Memory rebuild (spec §43). Memory is derived; raw JSONL is the source."""
from __future__ import annotations

import logging

from .schemas import InteractionTask

log = logging.getLogger("memory")


def rebuild_memory(orch, *, replay: bool = False, reset: bool = False) -> dict:
    """
    replay=False: rebuild the SQLite mirror from the Markdown files.
    replay=True : also queue every interaction in raw history for the memory model.
    reset=True  : (with replay) empty the Markdown files first, so memory is
                  regenerated purely from raw history. A snapshot is always taken first.
    """
    backup = orch.manager.snapshot("pre-rebuild")
    out = {"backup": str(backup), "replay": replay, "reset": reset}
    if replay and reset:
        orch.manager.reset_markdown()
    if replay:
        queued = skipped = 0
        for turn in orch.conv_log.iter_interactions():
            task = InteractionTask.from_dict({**turn, "source": "rebuild"})
            _, ok, _ = orch.worker.enqueue(task)
            queued += ok
            skipped += not ok
        out.update(queued=queued, skipped=skipped)
    out["db_entries"] = orch.manager.sync_db_from_markdown()
    log.info("memory rebuild", extra={"detail": str(out)})
    return out
