"""glass_brain.py — Agentic Observability display layer (0.4.0 Item 14).

The structured trace this renders (``ToolRegistry.log``, ``GoalState``,
``pending_yield``) already exists from Items 4/5/6/10 — logging was built
in from day one rather than retrofitted. This module is only the display
layer over it: a pure function producing Rich markup (so it's testable
without a live terminal), plus a thin TUI widget that refreshes from it.
"""
from __future__ import annotations

from typing import Any

_MAX_ROWS = 8


def render_glass_brain(core: Any) -> str:
    """Render the live tool-call trace, active goal, and any pending HitL
    yield as Rich markup text — the "what is Meta-Uchi doing right now"
    view. Accepts anything duck-typed like a ``Core``/``MetaUchi``
    instance (``.goal_state``, ``.tools.log``, ``.pending_yield``), so
    it's testable with a plain stand-in object, not just a live engine.
    """
    lines = ["[bold #bb9af7]─ Glass Brain ─[/bold #bb9af7]"]

    goal_state = getattr(core, "goal_state", None)
    if goal_state is not None:
        lines.append(f"[bold #7dcfff]Goal:[/bold #7dcfff] {goal_state.goal}")
        if goal_state.notes:
            lines.append(f"[dim]{len(goal_state.notes)} compacted note(s)[/dim]")
    else:
        lines.append("[dim]no active goal[/dim]")

    tools = getattr(core, "tools", None)
    log = list(getattr(tools, "log", [])) if tools is not None else []
    if not log:
        lines.append("[dim]no tool calls yet[/dim]")
    else:
        lines.append("")
        for entry in log[-_MAX_ROWS:]:
            if entry.ok:
                icon, color, detail = "✓", "#9ece6a", (entry.result or "")
            elif entry.error and entry.error.startswith("blocked:"):
                icon, color, detail = "⛔", "#e0af68", "blocked (loop guard)"
            else:
                icon, color, detail = "✗", "#f7768e", (entry.error or "")
            detail = detail.replace("\n", " ")[:40]
            lines.append(f"[{color}]{icon}[/{color}] {entry.name}  [dim]{detail}[/dim]")

    pending = getattr(core, "pending_yield", None)
    if pending:
        lines.append("")
        lines.append(f"[bold #e0af68]⏸ awaiting input:[/bold #e0af68] {pending[:60]}")

    return "\n".join(lines)
