import asyncio
import json
from datetime import datetime, timezone

import pytest

from app import evaluation as ev
from app.eval_asof import memory_as_of, parse_ts
from app.markdown_store import MarkdownStore, all_stores
from app.schemas import Category, MemoryEntry


def utc(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def write_jsonl(cfg, records):
    cfg.conversations_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.conversations_dir / "2026-01-01.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def msg(cid, role, content, ts):
    return {"conversation_id": cid, "type": "message", "role": role, "content": content, "timestamp": ts}


# --------------------------------------------------------- point in time
def test_state_is_first_snapshot_after_t_and_future_entries_dropped(tmp_path):
    mem = tmp_path / "memory"
    s = MarkdownStore(mem, Category.lesson, history_dir=mem / "history")
    old = MemoryEntry("L-001", Category.lesson, "Route", "Use /sse.", True, "2026-01-01T09:00:00+00:00",
                      "2026-01-01T09:00:00+00:00")
    new = MemoryEntry("L-001", Category.lesson, "Route", "Use /mcp, not /sse.", True,
                      "2026-01-01T09:00:00+00:00", "2026-03-01T09:00:00+00:00")
    later = MemoryEntry("L-002", Category.lesson, "Port", "Memory is on 11435.", True,
                        "2026-04-01T09:00:00+00:00", "2026-04-01T09:00:00+00:00")
    from app.markdown_store import MarkdownDocument
    (mem / "history").mkdir(parents=True)
    snap_time = datetime(2026, 3, 1, 9, 0, 0, tzinfo=timezone.utc).astimezone()
    snap = mem / "history" / f"LESSONS.{snap_time:%Y-%m-%d_%H-%M-%S}.000001.md"
    snap.write_text(s.render(MarkdownDocument("LESSONS", [], [old])), encoding="utf-8")   # state before the March write
    s.path.write_text(s.render(MarkdownDocument("LESSONS", [], [new, later])), encoding="utf-8")

    texts, approx = memory_as_of(mem, utc("2026-02-01T00:00:00"))
    assert "Use /sse." in texts["LESSONS.md"] and "/mcp" not in texts["LESSONS.md"] and not approx
    texts, _ = memory_as_of(mem, utc("2026-03-15T00:00:00"))       # after the write, before L-002
    assert "/mcp" in texts["LESSONS.md"] and "11435" not in texts["LESSONS.md"]
    texts, _ = memory_as_of(mem, utc("2026-05-01T00:00:00"))
    assert "11435" in texts["LESSONS.md"]


def test_pruned_history_is_flagged_approximate(tmp_path):
    mem = tmp_path / "memory"
    s = MarkdownStore(mem, Category.lesson, history_dir=mem / "history")
    s.ensure_exists()
    (mem / "history").mkdir(exist_ok=True)
    for i in range(3):
        t = datetime(2026, 3, 1 + i, tzinfo=timezone.utc).astimezone()
        (mem / "history" / f"LESSONS.{t:%Y-%m-%d_%H-%M-%S}.00000{i}.md").write_text(s.path.read_text())
    _, approx = memory_as_of(mem, utc("2026-01-01T00:00:00"), versions_to_keep=3)
    assert approx
    _, approx = memory_as_of(mem, utc("2026-01-01T00:00:00"), versions_to_keep=50)
    assert not approx


# ------------------------------------------------------------- the leak
def run(cfg, fakes, **kw):
    primary, memory = fakes
    kw.setdefault("variants", ev.load_variants(None))
    kw.setdefault("golden", [])
    kw.setdefault("sessions", [])
    return asyncio.run(ev.run_eval(cfg, primary_transport=primary.transport, memory_transport=memory.transport,
                                   progress=lambda *_: None, **kw))


def asked_prompts(primary):
    return [r["messages"][-1]["content"] for r in primary.requests if r.get("options", {}).get("num_predict", 0) > 1]


def test_future_memory_does_not_leak_into_past_questions(cfg, fakes):
    write_jsonl(cfg, [msg("s-jan", "user", "where does the memory model run?", "2026-01-10T10:00:00+00:00"),
                      msg("s-jan", "assistant", "Not decided yet.", "2026-01-10T10:00:05+00:00")])
    stores = all_stores(cfg.memory_dir)
    stores[Category.environment].add("Memory instance", "The memory model listens on port 11435.")  # learned "now"
    case = ev.GoldenCase(name="port", session="s-jan", upto_turn=0,
                         question="where does the memory model run?", expect_all=["11435"])
    primary, _ = fakes
    run(cfg, fakes, golden=[case], variant_names=["full"], memory="asof")
    assert all("11435" not in p for p in asked_prompts(primary))
    sys_prompts = [r["messages"][0]["content"] for r in primary.requests if r["messages"][0]["role"] == "system"]
    assert all("11435" not in p for p in sys_prompts)
    primary.requests.clear()
    run(cfg, fakes, golden=[case], variant_names=["full"], memory="current")
    sys_prompts = [r["messages"][0]["content"] for r in primary.requests if r["messages"][0]["role"] == "system"]
    assert any("11435" in p for p in sys_prompts)             # today's memory knows it: that is the leak


def test_scripted_cases_start_empty(cfg, fakes):
    all_stores(cfg.memory_dir)[Category.environment].add("Memory instance", "Listens on port 11435.")
    case = ev.GoldenCase(name="s", turns=[ev.TurnIn(user="hello", assistant="hi")],
                         question="which port?", expect_all=["11435"])
    primary, _ = fakes
    run(cfg, fakes, golden=[case], variant_names=["full"])
    assert not any("11435" in json.dumps(r["messages"]) for r in primary.requests if "messages" in r)


def test_history_recall_limited_to_the_past_and_other_sessions(cfg, fakes):
    cfg.embeddings.enabled = True
    cfg.history_recall.min_similarity = cfg.history_recall.cue_min_similarity = 0.05
    write_jsonl(cfg, [
        msg("s-old", "user", "gateway error E1234 appeared", "2026-01-05T10:00:00+00:00"),
        msg("s-old", "assistant", "E1234 was a stale plugin cache (old fix).", "2026-01-05T10:00:05+00:00"),
        msg("s-q", "user", "gateway error E1234 again, what was the fix?", "2026-02-01T10:00:00+00:00"),
        msg("s-q", "assistant", "Not sure.", "2026-02-01T10:00:05+00:00"),
        msg("s-new", "user", "gateway error E1234 root cause", "2026-03-01T10:00:00+00:00"),
        msg("s-new", "assistant", "E1234 is really the TLS cert (future fix).", "2026-03-01T10:00:05+00:00"),
    ])
    case = ev.GoldenCase(name="e1234", session="s-q", upto_turn=0,
                         question="gateway error E1234 again, what was the fix?", expect_any=["cache"])
    primary, _ = fakes
    run(cfg, fakes, golden=[case], variant_names=["full"])
    [p] = asked_prompts(primary)
    assert "old fix" in p                    # from before the question: allowed
    assert "future fix" not in p             # learned later: excluded
    assert "Not sure." not in p              # the case's own session is not recalled


# ------------------------------------------------- order and paired stats
def test_variant_order_is_shuffled_and_reproducible(cfg, fakes):
    cases = [ev.GoldenCase(name=f"c{i}", turns=[ev.TurnIn(user="x", assistant="y")], question="q?",
                           expect_all=["z"]) for i in range(8)]
    r1 = run(cfg, fakes, golden=cases, variant_names=["baseline", "full"], seed=7)
    r2 = run(cfg, fakes, golden=cases, variant_names=["baseline", "full"], seed=7)
    firsts = [o["order"][0] for o in r1["order"]]
    assert set(firsts) == {"baseline", "full"}                     # neither is always first
    assert [o["order"] for o in r1["order"]] == [o["order"] for o in r2["order"]]


def test_mcnemar_exact():
    assert ev.mcnemar_exact(0, 0) == 1.0
    assert ev.mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert ev.mcnemar_exact(1, 5) == pytest.approx(2 * 7 / 64)
    assert ev.mcnemar_exact(3, 3) == 1.0


def test_paired_report(cfg, fakes):
    primary, _ = fakes
    primary.reply = "it is 11435"
    cases = [ev.GoldenCase(name=f"c{i}", turns=[ev.TurnIn(user="x", assistant="y")], question="q?",
                           expect_all=["11435"]) for i in range(3)]
    rep = run(cfg, fakes, golden=cases, variant_names=["baseline", "full"])
    p = rep["paired"]["full"]
    assert p["golden_pairs"] == 3 and p["both_pass"] == 3 and p["mcnemar_p"] == 1.0
    md = open(rep["files"]["markdown"], encoding="utf-8").read()
    assert "Paired comparison against baseline" in md and "no future" in md
