import json

from uchi.checkpoint import load_into, save, to_checkpoint_dict
from uchi.episodic_memory import EpisodicMemory
from uchi.goal_state import GoalState
from uchi.tool_calling import ToolCall, ToolRegistry


class _FakeCore:
    """Minimal stand-in exposing only what checkpoint.py touches, so this
    doesn't need a full Core() (heavy FLUX/index load) just to test
    serialization round-tripping."""

    def __init__(self):
        self.pending_yield = None
        self.tools = ToolRegistry()
        self.tools.register("add", lambda a, b: a + b)
        self.goal_state = None
        self.episodic_memory = EpisodicMemory()


def test_checkpoint_dict_shape_with_no_active_goal():
    core = _FakeCore()
    core.episodic_memory.add_interaction("hi", "hello")
    data = to_checkpoint_dict(core)
    assert data["goal_state"] is None
    assert data["pending_yield"] is None
    assert data["episodic_history"][0]["user"] == "hi"


def test_save_and_load_roundtrip_tool_log_and_loop_guard(tmp_path):
    core = _FakeCore()
    core.tools.dispatch(ToolCall(name="add", args={"a": 1, "b": 2}, raw="x"))

    def boom():
        raise ValueError("boom")
    core.tools.register("boom", boom)
    core.tools.dispatch(ToolCall(name="boom", args={}, raw="x"))

    path = str(tmp_path / "ckpt.json")
    save(core, path)

    restored = _FakeCore()
    load_into(restored, path)

    assert len(restored.tools.log) == 2
    assert restored.tools.log[0].result == "3"
    assert not restored.tools.log[1].ok
    assert restored.tools.loop_guard.is_penalized(
        restored.tools._signature(ToolCall(name="boom", args={}, raw="x"))
    )


def test_save_and_load_roundtrip_goal_state(tmp_path):
    core = _FakeCore()
    core.goal_state = GoalState(goal="find X")
    core.goal_state.notes.append("[t1] some fact")
    core.goal_state.compacted_count = 1

    path = str(tmp_path / "ckpt.json")
    save(core, path)

    restored = _FakeCore()
    load_into(restored, path)

    assert restored.goal_state is not None
    assert restored.goal_state.goal == "find X"
    assert restored.goal_state.notes == ["[t1] some fact"]
    assert restored.goal_state.compacted_count == 1


def test_save_and_load_roundtrip_pending_yield(tmp_path):
    core = _FakeCore()
    core.pending_yield = "Which quarter?"
    path = str(tmp_path / "ckpt.json")
    save(core, path)

    restored = _FakeCore()
    load_into(restored, path)
    assert restored.pending_yield == "Which quarter?"


def test_save_and_load_roundtrip_episodic_memory(tmp_path):
    core = _FakeCore()
    core.episodic_memory.add_interaction("q1", "a1")
    core.episodic_memory.add_interaction("q2", "a2")
    path = str(tmp_path / "ckpt.json")
    save(core, path)

    restored = _FakeCore()
    load_into(restored, path)
    assert len(restored.episodic_memory.history) == 2
    assert restored.episodic_memory.history[1]["user"] == "q2"


def test_checkpoint_file_is_plain_readable_json(tmp_path):
    core = _FakeCore()
    core.goal_state = GoalState(goal="g")
    path = str(tmp_path / "ckpt.json")
    save(core, path)
    with open(path) as fh:
        data = json.load(fh)  # must not raise
    assert data["goal_state"]["goal"] == "g"
