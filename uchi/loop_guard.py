"""loop_guard.py — Self-Healing Loop Prevention (0.4.0 Item 6).

The most common failure mode for autonomous code-writing agents: write bad
code -> get error -> write the exact same bad code -> get error. This module
tracks failure fingerprints (a stable hash of the exact call or candidate
that failed) and refuses to let an identical fingerprint execute/win again,
forcing whatever generated it (a tool call's arguments, or a CodeEngine MCTS
candidate) down a structurally different path on the next attempt.

Wired into two places:
  - ``tool_calling.ToolRegistry.dispatch`` — an exact repeat of a
    previously-failed (name, args) pair is blocked outright rather than
    re-executed.
  - ``code_engine.CodeEngine.generate_code`` — an exact repeat of a
    previously-failed candidate string is skipped rather than re-verified.

A pathway that later succeeds is un-penalized (the guard blocks *repeating
a failure*, not permanently blacklisting a shape that might legitimately
succeed once external state changes — e.g. ``read_file`` on a path that
gets created later).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Set


def fingerprint(*parts: str) -> str:
    """Stable fingerprint for a call/candidate signature."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x00")
    return h.hexdigest()


@dataclass
class LoopGuard:
    """Tracks failed pathway fingerprints and blocks exact repeats."""

    _penalized: Set[str] = field(default_factory=set)
    _fail_counts: Dict[str, int] = field(default_factory=dict)

    def record_failure(self, *parts: str) -> str:
        fp = fingerprint(*parts)
        self._fail_counts[fp] = self._fail_counts.get(fp, 0) + 1
        self._penalized.add(fp)
        return fp

    def record_success(self, *parts: str) -> None:
        fp = fingerprint(*parts)
        self._penalized.discard(fp)
        self._fail_counts.pop(fp, None)

    def is_penalized(self, *parts: str) -> bool:
        return fingerprint(*parts) in self._penalized

    def failure_count(self, *parts: str) -> int:
        return self._fail_counts.get(fingerprint(*parts), 0)
