from app.conversation_logger import ConversationLogger


def test_append_order_unicode_and_large(tmp_path):
    lg = ConversationLogger(tmp_path)
    big = "x" * 1_000_000
    lg.message("c1", "user", "Olá, porquê o 404? 🚀")
    lg.tool_call("c1", "http_get", {"url": "http://kali:8080/mcp"})
    lg.tool_result("c1", "http_get", "404 Not Found")
    lg.message("c1", "assistant", big)
    recs = list(lg.iter_records())
    assert [r["type"] for r in recs] == ["message", "tool_call", "tool_result", "message"]
    assert recs[0]["content"] == "Olá, porquê o 404? 🚀"
    assert len(recs[3]["content"]) == 1_000_000
    turns = list(lg.iter_interactions())
    assert len(turns) == 1 and len(turns[0]["tool_events"]) == 2


def test_torn_line_does_not_block_recovery(tmp_path):
    lg = ConversationLogger(tmp_path)
    lg.message("c1", "user", "hi")
    f = next(tmp_path.glob("*.jsonl"))
    with open(f, "a", encoding="utf-8") as fh:
        fh.write('{"broken": \n')
    lg.message("c1", "assistant", "hello")
    assert len(list(lg.iter_interactions())) == 1
