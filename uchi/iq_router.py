"""iq_router.py — Intelligence Quotient (IQ) Task Router (0.4.0 Item 9).

Not every question needs the full swarm treatment. ``SwarmSynthesizer.
_decompose()`` burns a FLUX proposer round-trip deciding whether to split
a question into sub-questions — worth it for a genuinely multi-part
question, wasted for a simple lookup. ``SwarmSynthesizer.answer()``
previously called ``_decompose()`` unconditionally.

This is a cheap heuristic pre-check (the same regex/length pattern already
proven in ``intent_router.py``), not new infrastructure: score the
question's complexity before paying for decomposition, and skip straight
to a single pipeline pass when it's low.
"""
from __future__ import annotations

import re

# Signals that a question likely bundles multiple independent sub-problems.
_MULTI_PART = re.compile(r"\b(and|also|as well as|additionally|then|after that)\b", re.I)
_COMPARISON = re.compile(r"\b(compare|versus|vs\.?|difference between|both)\b", re.I)
_MULTI_QUESTION_MARKS = re.compile(r"\?.*\?")
# Note: no trailing \b after ")" — a word boundary can't occur between two
# non-word characters (")" followed by a space), so `\b1\)\b` would never
# match "1) " in practice. Anchor only on the leading boundary instead.
_ENUMERATION = re.compile(r"\b(first|second|third)\b|\b\d\)|\b[ab]\)", re.I)

DEFAULT_THRESHOLD = 0.3


def estimate_complexity(question: str) -> float:
    """Cheap heuristic complexity score in ``[0, 1]``.

    High score → the question likely bundles independent sub-problems
    worth decomposing. Low score → a single atomic lookup, not worth a
    decomposition round-trip.
    """
    q = question.strip()
    if not q:
        return 0.0

    score = 0.0
    word_count = len(q.split())

    if word_count > 25:
        score += 0.3
    if _MULTI_PART.search(q):
        score += 0.3
    if _COMPARISON.search(q):
        score += 0.35
    if _MULTI_QUESTION_MARKS.search(q):
        score += 0.35
    if _ENUMERATION.search(q):
        score += 0.35
    if q.count(",") >= 2:
        score += 0.15

    return min(1.0, score)


def should_decompose(question: str, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Whether *question* is worth paying for ``SwarmSynthesizer._decompose()``."""
    return estimate_complexity(question) >= threshold
