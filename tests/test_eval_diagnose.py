import asyncio
import json

import pytest

from app import evaluation as ev
from app.eval_diagnose import diagnose, evidence_in, split_prompt, summarize_diagnoses
from app.markdown_store import MarkdownDocument, MarkdownStore
from app.schemas import Category, MemoryEntry

CASE = ev.GoldenCase(name="port", question="Which port?", expect_all=["11435"])


def test_split_prompt_locates_each_source():
    msgs = [
        {"role": "system", "content": "client\n<PROJECT_MEMORY_BASE>\nENVIRONMENT:\n- [E-1] port 11435\n</PROJECT_MEMORY_BASE>"},
        {"role": "user", "content": "earlier: the gateway uses 8443"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Which port?\n\n<PROJECT_MEMORY>\nRELEVANT LESSONS:\n- [L-1] lesson 7777\n\n"
                                    "SESSION SO FAR (summary of earlier turns no longer shown verbatim):\nsummary 5555\n\n"
                                    "RELATED PAST EXCHANGES (verbatim excerpts):\n- past 6666\n</PROJECT_MEMORY>"},
    ]
    p = split_prompt(msgs, "Which port?")
    assert "11435" in p["memory_base"] and "7777" in p["memory_turn"]
    assert "5555" in p["session_summary"] and "6666" in p["past_exchanges"]
    assert "8443" in p["conversation"] and "Which port?" not in p["conversation"]
    assert "7777" not in p["conversation"]


@pytest.mark.parametrize("passed,parts,mem,hist,sess,expected", [
    (True, {"conversation": "port 11435"}, "", "", "", "answered_from_prompt"),
    (True, {"conversation": "nothing"}, "", "", "", "answered_without_evidence"),
    (False, {"memory_turn": "11435"}, "", "", "", "model_missed"),
    (False, {"conversation": "x"}, "port 11435", "", "", "not_retrieved"),
    (False, {"conversation": "x"}, "", "past: 11435", "", "not_retrieved"),
    (False, {"conversation": "x"}, "", "", "turn 1: 11435", "lost_from_session"),
    (False, {"conversation": "x"}, "", "", "", "never_available"),
])
def test_diagnose_categories(passed, parts, mem, hist, sess, expected):
    assert diagnose(CASE, passed, parts, memory_text=mem, history_text=hist, session_text=sess)[0] == expected


def test_forbid_only_and_regex():
    forbid_only = ev.GoldenCase(name="f", question="q", forbid=["x"])
    assert diagnose(forbid_only, True, {"conversation": "x"})[0] == "no_expectation"
    rx = ev.GoldenCase(name="r", question="q", expect_all=["/\\b114\\d\\d\\b/"])
    assert evidence_in(rx, "port 11435") and not evidence_in(rx, "port 211435")


# ------------------------------------------------------------- end to end
def run(cfg, fakes, **kw):
    primary, memory = fakes
    kw.setdefault("variants", ev.load_variants(None))
    kw.setdefault("sessions", [])
    return asyncio.run(ev.run_eval(cfg, primary_transport=primary.transport, memory_transport=memory.transport,
                                   progress=lambda *_: None, **kw))


def by_case(rep, variant="full"):
    return {g["case"]: g for g in rep["golden"] if g["variant"] == variant}


def scripted(name, turns, question, expect):
    return ev.GoldenCase(name=name, turns=[ev.TurnIn(user=u, assistant=a) for u, a in turns],
                         question=question, expect_all=[expect])


def test_each_diagnosis_end_to_end(cfg, fakes):
    primary, memory = fakes
    # Answers depend on the question, so one run produces several outcomes.
    from fastapi import Request

    @primary.app.middleware("http")
    async def answer(request: Request, call_next):
        if request.url.path == "/api/chat":
            body = await request.json()
            q = body["messages"][-1]["content"]
            primary.reply = {"Q-from-prompt": "It is 1111.", "Q-missed": "I don't know.",
                             "Q-guess": "Probably 3333.", "Q-never": "No idea."}.get(q.split("\n")[0], "ok")
        return await call_next(request)

    memory.reply = "A short summary with no numbers."
    filler = [("small talk " * 30, "reply " * 30)] * 4
    cases = [
        scripted("from-prompt", [("The code is 1111.", "Noted.")], "Q-from-prompt", "1111"),
        scripted("missed", [("The code is 2222.", "Noted.")], "Q-missed", "2222"),
        scripted("guess", [("Hello.", "Hi.")], "Q-guess", "3333"),
        scripted("lost", [("The code is 4444.", "Noted.")] + filler, "Q-lost", "4444"),
        scripted("never", [("Hello.", "Hi.")], "Q-never", "5555"),
    ]
    variants = {**ev.load_variants(None),
                "tight": {"proxy.trim_mode": "turns", "proxy.trim_trigger_user_turns": 3,
                          "proxy.trim_keep_user_turns": 2, "compression.keep_recent_user_turns": 1}}
    # 'lost' needs trimming: give the variant a summary that covers the trimmed turns.
    rep = run(cfg, fakes, golden=cases, variants=variants, variant_names=["tight"])
    d = {c: g["diagnosis"] for c, g in by_case(rep, "tight").items()}
    assert d["from-prompt"] == "answered_from_prompt"
    assert d["missed"] == "model_missed"
    assert d["guess"] == "answered_without_evidence"
    assert d["never"] == "never_available"
    assert d["lost"] == "lost_from_session"
    assert by_case(rep, "tight")["from-prompt"]["evidence_in"] == ["conversation"]
    md = open(rep["files"]["markdown"], encoding="utf-8").read()
    assert "Why answers passed or failed" in md and "Lost from the session" in md


def test_not_retrieved_and_retrieval_rate(cfg, fakes):
    primary, memory = fakes
    primary.reply = "No idea."
    # Memory from BEFORE the session: one entry the keyword search can find, one it cannot.
    st = MarkdownStore(cfg.memory_dir, Category.lesson, history_dir=cfg.history_dir)
    old = "2026-01-01T00:00:00+00:00"
    st.path.parent.mkdir(parents=True, exist_ok=True)
    st.path.write_text(st.render(MarkdownDocument("LESSONS", [], [
        MemoryEntry("L-001", Category.lesson, "Gateway port", "The gateway listens on 8443.", True, old, old),
        MemoryEntry("L-002", Category.lesson, "Backup window", "Backups run at 0230 nightly.", True, old, old)])),
        encoding="utf-8")
    cfg.conversations_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.conversations_dir / "2026-02-01.jsonl", "w") as f:
        for r in [{"conversation_id": "s", "type": "message", "role": "user", "content": "hello",
                   "timestamp": "2026-02-01T10:00:00+00:00"},
                  {"conversation_id": "s", "type": "message", "role": "assistant", "content": "hi",
                   "timestamp": "2026-02-01T10:00:01+00:00"}]:
            f.write(json.dumps(r) + "\n")
    cases = [ev.GoldenCase(name="found", session="s", upto_turn=1, question="which gateway port?",
                           expect_all=["8443"]),
             ev.GoldenCase(name="hidden", session="s", upto_turn=1, question="when does it happen?",
                           expect_all=["0230"])]
    rep = run(cfg, fakes, golden=cases, variant_names=["full"])
    d = by_case(rep)
    assert d["found"]["diagnosis"] == "model_missed" and "memory_turn" in d["found"]["evidence_in"]
    assert d["hidden"]["diagnosis"] == "not_retrieved"
    mr = rep["diagnosis"]["full"]["memory_retrieval"]
    assert mr == {"available": 2, "reached_prompt": 1, "rate": 0.5}
