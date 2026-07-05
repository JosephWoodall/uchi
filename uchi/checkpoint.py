"""checkpoint.py — Agentic State Checkpointing / Pause and Resume (0.4.0 Item 12).

Uchi's heavy state (FLUX weights, the semantic index) is already
reproducible from disk (``brain.uchi`` + the FLUX checkpoint) — there's no
need to re-serialize gigabytes of model weights. What checkpointing
actually needs to save is the *task-specific* state that would otherwise
be lost if the process stops mid-task: the active ``GoalState`` (goal +
compacted notes + pending raw log), the tool-call history and loop-guard
penalties, any pending HitL yield, and recent episodic-memory turns.
Restoring these onto a live ``Core``/``MetaUchi`` picks a paused task back
up where it left off.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from .simple import Core
    from .tool_calling import ToolCallLogEntry


def _entry_to_dict(entry: "ToolCallLogEntry") -> Dict[str, Any]:
    return {
        "name": entry.name,
        "args": entry.args,
        "result": entry.result,
        "error": entry.error,
        "timestamp": entry.timestamp,
        "duration_seconds": entry.duration_seconds,
    }


def _entry_from_dict(d: Dict[str, Any]) -> "ToolCallLogEntry":
    from .tool_calling import ToolCallLogEntry
    return ToolCallLogEntry(
        name=d["name"], args=d["args"], result=d.get("result"),
        error=d.get("error"), timestamp=d.get("timestamp", 0.0),
        duration_seconds=d.get("duration_seconds", 0.0),
    )


def to_checkpoint_dict(core: "Core") -> Dict[str, Any]:
    """Snapshot task-specific state from *core* into a JSON-serializable dict."""
    data: Dict[str, Any] = {
        "version": 1,
        "pending_yield": core.pending_yield,
        "tool_log": [_entry_to_dict(e) for e in core.tools.log],
        "loop_guard_penalized": sorted(core.tools.loop_guard._penalized),
        "loop_guard_fail_counts": dict(core.tools.loop_guard._fail_counts),
        "episodic_history": list(core.episodic_memory.history),
    }
    if core.goal_state is not None:
        data["goal_state"] = {
            "goal": core.goal_state.goal,
            "notes": list(core.goal_state.notes),
            "raw_log": [_entry_to_dict(e) for e in core.goal_state.raw_log],
            "compacted_count": core.goal_state.compacted_count,
            "successful_steps": [list(s) for s in core.goal_state.successful_steps],
        }
    else:
        data["goal_state"] = None
    return data


def save(core: "Core", path: str) -> None:
    """Serialize *core*'s task-specific state to *path* as JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_checkpoint_dict(core), fh, indent=2)


def load_into(core: "Core", path: str) -> None:
    """Restore task-specific state from *path* onto *core*, in place."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    core.pending_yield = data.get("pending_yield")
    core.tools.log = [_entry_from_dict(d) for d in data.get("tool_log", [])]
    core.tools.loop_guard._penalized = set(data.get("loop_guard_penalized", []))
    core.tools.loop_guard._fail_counts = dict(data.get("loop_guard_fail_counts", {}))
    core.episodic_memory.history = list(data.get("episodic_history", []))

    gs_data = data.get("goal_state")
    if gs_data is not None:
        from .goal_state import GoalState
        gs = GoalState(goal=gs_data["goal"])
        gs.notes = list(gs_data.get("notes", []))
        gs.raw_log = [_entry_from_dict(d) for d in gs_data.get("raw_log", [])]
        gs.compacted_count = gs_data.get("compacted_count", 0)
        gs.successful_steps = [tuple(s) for s in gs_data.get("successful_steps", [])]
        core.goal_state = gs
    else:
        core.goal_state = None
