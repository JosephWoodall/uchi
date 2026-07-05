"""goal_state.py — Goal State tracking & Memory Compaction (0.4.0 Item 5).

Tool calling, web browsing, and code execution generate a lot of token
noise: raw HTML, long stack traces, large tool outputs. ``GoalState`` is
deterministic running memory of what's been accomplished so far on a task,
kept separate from the raw tool-call log so the active context doesn't
balloon with execution noise as a task runs longer.

Compaction is extractive, not abstractive: it keeps a bounded excerpt of
each tool result verbatim rather than paraphrasing it. Every note written
into ``GoalState.notes`` therefore traces back to an actual tool output —
a free-form paraphrase engine would be a confabulation backdoor into the
trie (see ``tasks/core_principle.md``'s North Star: never assert what
can't be traced back to something actually retrieved).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .tool_calling import ToolCallLogEntry

DEFAULT_MAX_LOG_CHARS = 4000     # trigger compaction once raw log exceeds this
DEFAULT_EXCERPT_CHARS = 200      # kept verbatim per compacted entry


def _entry_text(entry: "ToolCallLogEntry") -> str:
    return entry.result if entry.ok else (entry.error or "")


@dataclass
class GoalState:
    """Deterministic running memory for one goal/task."""

    goal: str
    notes: List[str] = field(default_factory=list)
    raw_log: List["ToolCallLogEntry"] = field(default_factory=list)
    compacted_count: int = 0

    def record(self, entry: "ToolCallLogEntry") -> None:
        self.raw_log.append(entry)

    def raw_log_chars(self) -> int:
        return sum(len(_entry_text(e)) for e in self.raw_log)

    def maybe_compact(
        self,
        max_chars: int = DEFAULT_MAX_LOG_CHARS,
        excerpt_chars: int = DEFAULT_EXCERPT_CHARS,
    ) -> bool:
        """If the raw tool log exceeds *max_chars*, compact it into
        ``notes`` (extractive only — a verbatim excerpt, never a
        paraphrase) and drop the raw entries. Returns whether it ran."""
        if self.raw_log_chars() <= max_chars:
            return False
        for entry in self.raw_log:
            text = _entry_text(entry)
            excerpt = text[:excerpt_chars]
            if len(text) > excerpt_chars:
                excerpt += "…"
            self.notes.append(f"[{entry.name}] {excerpt}")
            self.compacted_count += 1
        self.raw_log.clear()
        return True

    def context_string(self) -> str:
        """Render goal + accumulated notes as text for the pipeline to
        fold back into the next question, so a long task never loses the
        original intent even after its raw logs have been dropped."""
        parts = [f"Goal: {self.goal}"]
        if self.notes:
            parts.append("Learned so far:")
            parts.extend(f"- {n}" for n in self.notes)
        return "\n".join(parts)
