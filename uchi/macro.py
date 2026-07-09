"""macro.py — Episodic Memory & Macro Skill Distillation (0.4.0 Item 13).

When a multi-step task succeeds, the sequence of tool calls that got there
is thrown away — the next time Uchi faces a similar goal, it reasons
through the exact same steps from scratch. This distills the *successful*
steps of a finished ``GoalState`` (errors and dead-ends already excluded —
``GoalState.successful_steps`` only ever records calls that didn't fail)
into a reusable ``Macro``: registered as a fast-path tool that replays the
whole sequence in one call, and ingested into the knowledge index so its
existence is permanently discoverable.

Persisted to ``.uchi/macros/`` (matching the ``.uchi/telemetry`` and
``.uchi/workspace`` dot-directory convention already used elsewhere) so
distilled macros survive across sessions.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .goal_state import GoalState
    from .tool_calling import ToolCallLogEntry, ToolRegistry

DEFAULT_MACRO_DIR = os.path.join(".uchi", "macros")


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (s[:40] or "macro")


@dataclass
class Macro:
    goal: str
    steps: List[Tuple[str, Dict[str, Any]]]

    @property
    def name(self) -> str:
        return f"macro_{_slugify(self.goal)}"

    def as_knowledge(self) -> str:
        step_desc = "; ".join(f"{n}({a})" for n, a in self.steps)
        return f"Macro {self.name!r} learned for goal {self.goal!r}: {step_desc}"

    def replay(self, registry: "ToolRegistry") -> List["ToolCallLogEntry"]:
        """Re-execute every step in order through *registry* — still a
        real dispatch each time (loop guard included), never blindly
        trusted just because it succeeded once before."""
        from .tool_calling import ToolCall
        return [
            registry.dispatch(ToolCall(name=n, args=a, raw=""))
            for n, a in self.steps
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {"goal": self.goal, "steps": [[n, a] for n, a in self.steps]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Macro":
        return cls(goal=d["goal"], steps=[(n, a) for n, a in d["steps"]])


def distill(goal_state: "GoalState") -> Optional[Macro]:
    """Extract the successful tool-call sequence from *goal_state* into a
    reusable Macro. Errors and dead-ends are already excluded — they were
    never added to ``successful_steps`` in the first place. Returns None
    if nothing successful was recorded."""
    if not goal_state.successful_steps:
        return None
    return Macro(goal=goal_state.goal, steps=list(goal_state.successful_steps))


def register_macro_tool(macro: Macro, registry: "ToolRegistry") -> None:
    """Register *macro* as a fast-path tool: calling it replays the whole
    distilled sequence in one shot instead of reasoning step-by-step."""
    registry.register(macro.name, lambda: [e.result for e in macro.replay(registry)])


def save_macro(macro: Macro, directory: str = DEFAULT_MACRO_DIR) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{macro.name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(macro.to_dict(), fh, indent=2)
    return path


def load_macros(directory: str = DEFAULT_MACRO_DIR) -> List[Macro]:
    if not os.path.isdir(directory):
        return []
    macros = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".json"):
            with open(os.path.join(directory, fname), encoding="utf-8") as fh:
                macros.append(Macro.from_dict(json.load(fh)))
    return macros
