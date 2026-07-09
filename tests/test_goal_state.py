from uchi.goal_state import GoalState
from uchi.tool_calling import ToolCallLogEntry, ToolRegistry, run_with_tools


def _entry(name="tool", result="ok", error=None):
    return ToolCallLogEntry(name=name, args={}, result=result, error=error)


def test_context_string_with_no_notes():
    gs = GoalState(goal="find the answer")
    assert gs.context_string() == "Goal: find the answer"


def test_context_string_includes_notes():
    gs = GoalState(goal="find the answer")
    gs.notes.append("[tool] some fact")
    text = gs.context_string()
    assert "Goal: find the answer" in text
    assert "Learned so far:" in text
    assert "- [tool] some fact" in text


def test_no_compaction_below_threshold():
    gs = GoalState(goal="g")
    gs.record(_entry(result="short"))
    ran = gs.maybe_compact(max_chars=1000)
    assert not ran
    assert gs.notes == []
    assert len(gs.raw_log) == 1


def test_compaction_triggers_above_threshold():
    gs = GoalState(goal="g")
    gs.record(_entry(name="t1", result="x" * 50))
    gs.record(_entry(name="t2", result="y" * 50))
    ran = gs.maybe_compact(max_chars=60, excerpt_chars=10)
    assert ran
    assert gs.raw_log == []
    assert gs.compacted_count == 2
    assert len(gs.notes) == 2


def test_compaction_is_extractive_only_verbatim_excerpt():
    """Every note must be a literal substring of the tool's own output —
    never a paraphrase — per the North Star grounding constraint."""
    gs = GoalState(goal="g")
    gs.record(_entry(name="t1", result="the exact string the tool produced"))
    gs.maybe_compact(max_chars=0, excerpt_chars=200)
    assert len(gs.notes) == 1
    note = gs.notes[0]
    assert note.startswith("[t1] ")
    excerpt = note[len("[t1] "):]
    assert excerpt in "the exact string the tool produced"


def test_compaction_excerpt_is_truncated_with_ellipsis():
    gs = GoalState(goal="g")
    gs.record(_entry(name="t1", result="a" * 500))
    gs.maybe_compact(max_chars=0, excerpt_chars=50)
    note = gs.notes[0]
    assert note == f"[t1] {'a' * 50}…"


def test_compaction_uses_error_text_for_failed_entries():
    gs = GoalState(goal="g")
    gs.record(_entry(name="t1", result=None, error="boom"))
    gs.maybe_compact(max_chars=0)
    assert gs.notes == ["[t1] boom"]


def test_run_with_tools_records_into_goal_state():
    registry = ToolRegistry()
    registry.register("add", lambda a, b: a + b)
    gs = GoalState(goal="compute totals")
    text = "<|tool_call|> add(a=2, b=3) <|end_tool|>"
    run_with_tools(text, registry, goal_state=gs)
    assert len(gs.raw_log) == 1
    assert gs.raw_log[0].result == "5"


def test_run_with_tools_compacts_goal_state_when_threshold_exceeded():
    registry = ToolRegistry()
    registry.register("big", lambda: "z" * 100)
    gs = GoalState(goal="g")
    # tiny excerpt/threshold defaults on the GoalState side are triggered by
    # calling maybe_compact with defaults inside run_with_tools; force a low
    # threshold by pre-seeding raw_log length via repeated calls.
    for _ in range(50):
        text = "<|tool_call|> big() <|end_tool|>"
        run_with_tools(text, registry, goal_state=gs)
    assert gs.compacted_count > 0
    assert gs.raw_log_chars() < 5000
