from uchi.loop_guard import LoopGuard, fingerprint
from uchi.tool_calling import ToolCall, ToolRegistry


def test_fingerprint_is_stable_and_order_sensitive():
    assert fingerprint("a", "b") == fingerprint("a", "b")
    assert fingerprint("a", "b") != fingerprint("b", "a")


def test_not_penalized_initially():
    lg = LoopGuard()
    assert not lg.is_penalized("x")


def test_record_failure_penalizes():
    lg = LoopGuard()
    lg.record_failure("x")
    assert lg.is_penalized("x")
    assert lg.failure_count("x") == 1


def test_repeated_failures_increment_count():
    lg = LoopGuard()
    lg.record_failure("x")
    lg.record_failure("x")
    lg.record_failure("x")
    assert lg.failure_count("x") == 3


def test_success_clears_penalty():
    lg = LoopGuard()
    lg.record_failure("x")
    assert lg.is_penalized("x")
    lg.record_success("x")
    assert not lg.is_penalized("x")
    assert lg.failure_count("x") == 0


def test_different_signature_unaffected():
    lg = LoopGuard()
    lg.record_failure("x")
    assert not lg.is_penalized("y")


# ── ToolRegistry integration ─────────────────────────────────────────────

def test_second_identical_failing_call_is_blocked_not_reexecuted():
    calls = []
    def flaky(x):
        calls.append(x)
        raise ValueError("boom")
    registry = ToolRegistry()
    registry.register("flaky", flaky)

    first = registry.dispatch(ToolCall(name="flaky", args={"x": 1}, raw="a"))
    assert not first.ok
    assert "boom" in first.error
    assert len(calls) == 1

    second = registry.dispatch(ToolCall(name="flaky", args={"x": 1}, raw="a"))
    assert not second.ok
    assert "blocked" in second.error
    # the underlying function was NOT invoked a second time
    assert len(calls) == 1


def test_identical_call_with_different_args_is_not_blocked():
    registry = ToolRegistry()
    def flaky(x):
        if x == 1:
            raise ValueError("boom")
        return "ok"
    registry.register("flaky", flaky)

    registry.dispatch(ToolCall(name="flaky", args={"x": 1}, raw="a"))
    second = registry.dispatch(ToolCall(name="flaky", args={"x": 2}, raw="a"))
    assert second.ok
    assert second.result == "ok"


def test_success_after_failure_unblocks_future_identical_calls():
    state = {"n": 0}
    def sometimes(x):
        state["n"] += 1
        if state["n"] == 1:
            raise ValueError("boom")
        return "ok"

    registry = ToolRegistry()
    registry.register("sometimes", sometimes)

    first = registry.dispatch(ToolCall(name="sometimes", args={"x": 1}, raw="a"))
    assert not first.ok

    # Directly clear the penalty (simulating external state having changed)
    # then confirm a subsequent identical call is not permanently blocked.
    registry.loop_guard.record_success(registry._signature(
        ToolCall(name="sometimes", args={"x": 1}, raw="a")
    ))
    second = registry.dispatch(ToolCall(name="sometimes", args={"x": 1}, raw="a"))
    assert second.ok
    assert second.result == "ok"
