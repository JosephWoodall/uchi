import time

from uchi.goal_state import GoalState
from uchi.tool_calling import ToolRegistry, parse_async_tool_calls, run_async_tool_calls


def test_parse_async_tool_calls_finds_all():
    text = (
        "<|tool_call_async|> a(x=1) <|end_tool|> and "
        "<|tool_call_async|> b(y=2) <|end_tool|>"
    )
    calls = parse_async_tool_calls(text)
    assert [c.name for c in calls] == ["a", "b"]
    assert calls[0].args == {"x": 1}
    assert calls[1].args == {"y": 2}


def test_no_async_calls_is_noop():
    text = "nothing async here, just <|tool_call|> sync(x=1) <|end_tool|>"
    out = run_async_tool_calls(text, ToolRegistry())
    assert out == text  # sync marker untouched by the async runner


def test_multiple_async_calls_run_concurrently_not_serially():
    registry = ToolRegistry()

    def slow(n):
        time.sleep(0.3)
        return n

    registry.register("slow", slow)
    text = (
        "<|tool_call_async|> slow(n=1) <|end_tool|> "
        "<|tool_call_async|> slow(n=2) <|end_tool|> "
        "<|tool_call_async|> slow(n=3) <|end_tool|>"
    )
    t0 = time.time()
    out = run_async_tool_calls(text, registry)
    elapsed = time.time() - t0

    # Three 0.3s calls run serially would take ~0.9s; concurrently, ~0.3s.
    assert elapsed < 0.6, f"calls did not run concurrently (took {elapsed:.2f}s)"
    assert "<|tool_call_async|>" not in out
    assert out.count("<|tool_result|>") == 3


def test_results_land_in_original_positions():
    registry = ToolRegistry()
    registry.register("add", lambda a, b: a + b)
    text = (
        "first: <|tool_call_async|> add(a=1, b=1) <|end_tool|> "
        "second: <|tool_call_async|> add(a=10, b=10) <|end_tool|>"
    )
    out = run_async_tool_calls(text, registry)
    first_idx = out.index("first:")
    second_idx = out.index("second:")
    assert first_idx < second_idx
    # the "2" result appears before the "20" result, matching original order
    assert out.index("-> 2 ") < out.index("-> 20 ")


def test_async_errors_do_not_crash_the_batch():
    registry = ToolRegistry()
    def boom():
        raise ValueError("kaboom")
    registry.register("ok", lambda: "fine")
    registry.register("boom", boom)
    text = (
        "<|tool_call_async|> ok() <|end_tool|> "
        "<|tool_call_async|> boom() <|end_tool|>"
    )
    out = run_async_tool_calls(text, registry)
    assert "fine" in out
    assert "kaboom" in out


def test_async_calls_record_into_goal_state():
    registry = ToolRegistry()
    registry.register("add", lambda a, b: a + b)
    gs = GoalState(goal="parallel sums")
    text = (
        "<|tool_call_async|> add(a=1, b=1) <|end_tool|> "
        "<|tool_call_async|> add(a=2, b=2) <|end_tool|>"
    )
    run_async_tool_calls(text, registry, goal_state=gs)
    assert len(gs.raw_log) == 2
