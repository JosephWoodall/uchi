"""hitl.py — Human-in-the-Loop (HitL) Yielding (0.4.0 Item 10).

Autonomous agents inevitably get stuck or hit high-risk ambiguity. Rather
than brute-forcing through it, Uchi can emit:

    <|yield_to_user|> clarifying question text <|end_yield|>

which pauses the response and surfaces the question to the human instead
of guessing. The next ``ask()`` call is treated as the human's answer and
folds back into context, so the task resumes rather than restarting.

Yields also trigger automatically — no explicit marker needed — when a
tool call is blocked by the Item 6 loop guard: a blocked repeat is exactly
the "stuck" signal that should escalate to a human rather than let the
pipeline silently keep returning the same blocked-error text forever.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .tool_calling import ToolCallLogEntry

_YIELD_RE = re.compile(r"<\|yield_to_user\|>\s*(.*?)\s*<\|end_yield\|>", re.DOTALL)

YIELD_PREFIX = "[Uchi needs input] "


@dataclass
class YieldRequest:
    question: str


def parse_yield(text: str) -> Optional[YieldRequest]:
    """Find an explicit ``<|yield_to_user|>`` marker in *text*."""
    m = _YIELD_RE.search(text)
    if not m:
        return None
    return YieldRequest(question=m.group(1).strip())


def is_blocked_by_loop_guard(entry: "ToolCallLogEntry") -> bool:
    """Whether *entry* failed because the Item 6 loop guard blocked an
    exact repeat of a previously-failed call — the "stuck" signal that
    should escalate to a human rather than silently repeat the same
    error forever."""
    return (not entry.ok) and bool(entry.error) and entry.error.startswith("blocked:")


def format_yield(question: str) -> str:
    return f"{YIELD_PREFIX}{question}"
