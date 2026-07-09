from uchi.tool_calling import ToolCall, default_registry


def test_run_python_success_is_ok(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    registry = default_registry()
    entry = registry.dispatch(ToolCall(name="run_python", args={"code": "print(2 + 2)"}, raw="x"))
    assert entry.ok
    assert "4" in entry.result


def test_run_python_failure_is_not_ok(tmp_path, monkeypatch):
    """A scratchpad run that itself errors must be classified as a failed
    dispatch (entry.error set), not a successful call whose result text
    happens to describe an error -- Item 6's loop guard and Item 10's
    HitL auto-escalation both depend on this distinction."""
    monkeypatch.chdir(tmp_path)
    registry = default_registry()
    entry = registry.dispatch(
        ToolCall(name="run_python", args={"code": "raise ValueError('boom')"}, raw="x")
    )
    assert not entry.ok
    assert "boom" in entry.error


def test_run_python_repeated_failure_is_blocked_by_loop_guard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    registry = default_registry()
    call = ToolCall(name="run_python", args={"code": "raise ValueError('boom')"}, raw="x")
    first = registry.dispatch(call)
    assert not first.ok
    assert "boom" in first.error

    second = registry.dispatch(call)
    assert not second.ok
    assert "blocked" in second.error
