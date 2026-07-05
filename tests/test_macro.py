from uchi.goal_state import GoalState
from uchi.macro import Macro, distill, load_macros, register_macro_tool, save_macro
from uchi.tool_calling import ToolCall, ToolCallLogEntry, ToolRegistry


def _entry(name, result="ok", error=None, args=None):
    return ToolCallLogEntry(name=name, args=args or {}, result=result, error=error)


def test_distill_returns_none_with_no_successful_steps():
    gs = GoalState(goal="g")
    assert distill(gs) is None


def test_distill_excludes_failures_keeps_successes():
    gs = GoalState(goal="build a report")
    gs.record(_entry("read_file", result="data", args={"path": "a.txt"}))
    gs.record(_entry("run_python", result=None, error="boom", args={"code": "bad"}))
    gs.record(_entry("write_file", result="ok", args={"path": "out.txt", "content": "x"}))

    macro = distill(gs)
    assert macro is not None
    assert macro.goal == "build a report"
    assert macro.steps == [
        ("read_file", {"path": "a.txt"}),
        ("write_file", {"path": "out.txt", "content": "x"}),
    ]


def test_macro_survives_compaction_of_raw_log():
    """Compaction (Item 5) drops raw_log's bulky text but must never drop
    the lightweight successful_steps trace macro distillation needs."""
    gs = GoalState(goal="g")
    gs.record(_entry("t1", result="x" * 100, args={"a": 1}))
    gs.maybe_compact(max_chars=10)
    assert gs.raw_log == []  # compacted away
    macro = distill(gs)
    assert macro is not None
    assert macro.steps == [("t1", {"a": 1})]


def test_macro_name_is_slugified_from_goal():
    macro = Macro(goal="Build a Financial Report!", steps=[("t", {})])
    assert macro.name == "macro_build_a_financial_report"


def test_macro_as_knowledge_mentions_goal_and_steps():
    macro = Macro(goal="sum numbers", steps=[("run_python", {"code": "print(1)"})])
    text = macro.as_knowledge()
    assert "sum numbers" in text
    assert "run_python" in text


def test_register_macro_tool_and_replay():
    registry = ToolRegistry()
    calls = []
    registry.register("step_a", lambda x: calls.append(("a", x)) or f"did {x}")
    registry.register("step_b", lambda y: calls.append(("b", y)) or f"did {y}")

    macro = Macro(goal="do both steps", steps=[("step_a", {"x": 1}), ("step_b", {"y": 2})])
    register_macro_tool(macro, registry)
    assert registry.has(macro.name)

    entry = registry.dispatch(ToolCall(name=macro.name, args={}, raw="x"))
    assert entry.ok
    assert calls == [("a", 1), ("b", 2)]


def test_replay_still_goes_through_the_real_registry_not_a_cache():
    """A macro replay must actually re-dispatch each step (so the loop
    guard, logging, etc. all still apply) -- not just replay a cached
    result string from when it first succeeded."""
    registry = ToolRegistry()
    call_count = {"n": 0}
    def counting():
        call_count["n"] += 1
        return call_count["n"]
    registry.register("counter", counting)

    macro = Macro(goal="count", steps=[("counter", {})])
    entries1 = macro.replay(registry)
    entries2 = macro.replay(registry)
    assert entries1[0].result == "1"
    assert entries2[0].result == "2"  # actually re-ran, not cached


def test_save_and_load_macro_roundtrip(tmp_path):
    macro = Macro(goal="find X", steps=[("t1", {"a": 1}), ("t2", {"b": "two"})])
    directory = str(tmp_path / "macros")
    path = save_macro(macro, directory=directory)
    assert path.endswith(".json")

    loaded = load_macros(directory=directory)
    assert len(loaded) == 1
    assert loaded[0].goal == "find X"
    assert loaded[0].steps == [("t1", {"a": 1}), ("t2", {"b": "two"})]


def test_load_macros_empty_directory_returns_empty_list(tmp_path):
    assert load_macros(directory=str(tmp_path / "nonexistent")) == []
