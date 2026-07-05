from types import SimpleNamespace

from uchi.goal_state import GoalState
from uchi.tool_calling import ToolCallLogEntry, ToolRegistry
from uchi.tui.glass_brain import render_glass_brain


def _fake_core(goal_state=None, log=None, pending_yield=None):
    tools = ToolRegistry()
    tools.log = log or []
    return SimpleNamespace(goal_state=goal_state, tools=tools, pending_yield=pending_yield)


def test_no_goal_no_log_no_pending():
    text = render_glass_brain(_fake_core())
    assert "no active goal" in text
    assert "no tool calls yet" in text
    assert "awaiting input" not in text


def test_active_goal_and_notes_shown():
    gs = GoalState(goal="find the answer")
    gs.notes.append("[t] some fact")
    text = render_glass_brain(_fake_core(goal_state=gs))
    assert "find the answer" in text
    assert "1 compacted note" in text


def test_successful_entry_rendered_with_checkmark():
    entry = ToolCallLogEntry(name="add", args={}, result="5", error=None)
    text = render_glass_brain(_fake_core(log=[entry]))
    assert "✓" in text
    assert "add" in text
    assert "5" in text


def test_failed_entry_rendered_with_x():
    entry = ToolCallLogEntry(name="run_python", args={}, result=None, error="ValueError: boom")
    text = render_glass_brain(_fake_core(log=[entry]))
    assert "✗" in text
    assert "boom" in text


def test_blocked_entry_rendered_distinctly():
    entry = ToolCallLogEntry(name="run_python", args={}, result=None, error="blocked: repeat failure")
    text = render_glass_brain(_fake_core(log=[entry]))
    assert "⛔" in text
    assert "blocked (loop guard)" in text


def test_pending_yield_shown():
    text = render_glass_brain(_fake_core(pending_yield="Which quarter?"))
    assert "awaiting input" in text
    assert "Which quarter?" in text


def test_log_truncated_to_max_rows():
    entries = [ToolCallLogEntry(name=f"t{i}", args={}, result="ok", error=None) for i in range(20)]
    text = render_glass_brain(_fake_core(log=entries))
    # only the most recent 8 should be shown
    assert "t19" in text
    assert "t0" not in text


def test_renders_against_a_real_core_without_crashing():
    """The one thing this module can't get wrong: it must handle a real
    Core instance's actual attribute shapes, not just the fake stand-in."""
    from uchi import Core
    core = Core()
    text = render_glass_brain(core)
    assert "Glass Brain" in text
