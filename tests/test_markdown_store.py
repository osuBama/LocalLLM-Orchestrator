import os

import pytest

from app.markdown_store import MarkdownFormatError, MarkdownStore
from app.schemas import Category


@pytest.fixture
def store(tmp_path):
    s = MarkdownStore(tmp_path, Category.lesson, history_dir=tmp_path / "history")
    s.ensure_exists()
    return s


def test_add_update_deactivate(store):
    e = store.add("MCP 404", "Verify the route before the network.", "conv1")
    assert e.entry_id == "L-001"
    assert store.add("Second", "Another lesson.").entry_id == "L-002"
    store.update("L-001", content="Verify the MCP route path first.")
    assert store.get("L-001").content == "Verify the MCP route path first."
    store.deactivate("L-002")
    assert [x.entry_id for x in store.entries(active_only=True)] == ["L-001"]
    text = store.path.read_text(encoding="utf-8")
    assert "## Active" in text and "## Inactive" in text and "### L-002 — Second" in text


def test_backup_created(store):
    store.add("A", "one")
    store.add("B", "two")
    assert len(list((store.history_dir).glob("LESSONS.*.md"))) >= 2


def test_atomic_replacement_leaves_no_temp(store):
    store.add("A", "one")
    assert not [p for p in store.path.parent.iterdir() if ".tmp" in p.name]


def test_recovery_after_write_failure(store, monkeypatch):
    store.add("A", "original")
    before = store.path.read_text(encoding="utf-8")

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        store.add("B", "never lands")
    monkeypatch.undo()
    assert store.path.read_text(encoding="utf-8") == before
    assert not [p for p in store.path.parent.iterdir() if ".tmp" in p.name]


def test_heading_injection_is_escaped(store):
    store.add("Tricky", "line one\n## Active\n### L-999 — fake\n<!-- meta: status=inactive -->")
    entries = store.entries()
    assert len(entries) == 1 and entries[0].active
    assert "\\## Active" in entries[0].content


def test_unmanaged_file_is_refused(tmp_path):
    s = MarkdownStore(tmp_path, Category.state)
    s.path.write_text("# STATE\n\n## Current Environment\n\n- Windows host\n", encoding="utf-8")
    with pytest.raises(MarkdownFormatError):
        s.add("x", "y")
    assert "Current Environment" in s.path.read_text(encoding="utf-8")


def test_hand_edits_preserved(store):
    store.add("A", "one")
    txt = store.path.read_text(encoding="utf-8").replace("one", "one, edited by hand")
    store.path.write_text(txt, encoding="utf-8")
    store.add("B", "two")
    assert store.get("L-001").content == "one, edited by hand"
