from uchi.hitl import format_yield, is_blocked_by_loop_guard, parse_yield
from uchi.tool_calling import ToolCallLogEntry


def test_parse_yield_extracts_question():
    text = "before <|yield_to_user|> Should I proceed? <|end_yield|> after"
    req = parse_yield(text)
    assert req is not None
    assert req.question == "Should I proceed?"


def test_parse_yield_none_when_absent():
    assert parse_yield("just plain text") is None


def test_format_yield_has_prefix():
    out = format_yield("pick one")
    assert out == "[Uchi needs input] pick one"


def test_is_blocked_by_loop_guard_true_for_blocked_error():
    entry = ToolCallLogEntry("t", {}, None, "blocked: an identical call already failed")
    assert is_blocked_by_loop_guard(entry)


def test_is_blocked_by_loop_guard_false_for_other_errors():
    entry = ToolCallLogEntry("t", {}, None, "ValueError: boom")
    assert not is_blocked_by_loop_guard(entry)


def test_is_blocked_by_loop_guard_false_on_success():
    entry = ToolCallLogEntry("t", {}, "ok", None)
    assert not is_blocked_by_loop_guard(entry)
