"""task_config_cache.py — ODUSP-backed vote-count recall for dynamic-N
sampling (0.4.0, follow-on to Item 17).

Not a memorization layer for specific claims (that use case is narrow --
most real corrections are about different facts each time, so exact
repeats rarely fire). This memoizes something that DOES recur broadly:
the STRUCTURAL SHAPE of a question (multi-part, comparison, enumeration,
length) and how many self-consistency votes it actually took to reach a
confident, grounded answer last time a similarly-shaped question came
through. Structural patterns repeat across completely different topics
far more often than exact factual claims do, which is what makes this a
better-fitting use of UniversalPredictor's credibility-weighted recall
than the original claim-memoization idea.

Uses UniversalPredictor's existing predict_next()/feedback() pattern
exactly as built for sequence prediction -- no changes to predictor.py.
Low credibility (novel/unseen structural pattern) means "no recommendation
yet", never a wrong one -- the caller always has a safe default to fall
back to, same graceful-degradation shape as every other optional signal
in this codebase.

Note on safety: unlike the oracle's layered vetoes (which can only REJECT
a claim, never accept one the deterministic check rejected), a
recommended vote count is a compute-budget knob, not a correctness gate.
Every candidate still goes through the full, unchanged oracle cascade
regardless of how many were generated. So this cache is allowed to move N
in either direction -- fewer votes for a reliably-easy pattern just means
less redundant sampling, not a weaker acceptance criterion.
"""
from __future__ import annotations

import re
from typing import Optional

from .predictor import UniversalPredictor

_MULTI_PART = re.compile(r"\b(and|also|as well as|additionally|then|after that)\b", re.I)
_COMPARISON = re.compile(r"\b(compare|versus|vs\.?|difference between|both)\b", re.I)
_MULTI_QUESTION_MARKS = re.compile(r"\?.*\?")
_ENUMERATION = re.compile(r"\b(first|second|third)\b|\b\d\)|\b[ab]\)", re.I)

# The N values a caller can actually be told to use -- a small, fixed
# vocabulary of compute budgets, not a continuous value (the trie predicts
# discrete symbols).
N_BUCKETS = (1, 2, 3, 5, 8)


def _nearest_bucket(n: int) -> int:
    return min(N_BUCKETS, key=lambda b: abs(b - n))


def _feature_signature(question: str) -> tuple:
    """A short, fixed-length tuple of discretized structural features --
    the "context" the trie indexes. Deliberately reuses the same signals
    as iq_router.py's estimate_complexity(), since those already capture
    the structural shape that's relevant here.
    """
    q = question.strip()
    word_count = len(q.split())
    return (
        "start",  # sentinel: gives the trie a stable anchor per signature
        "long" if word_count > 25 else "short",
        "multipart" if _MULTI_PART.search(q) else "single",
        "comparison" if _COMPARISON.search(q) else "nocompare",
        "multiq" if _MULTI_QUESTION_MARKS.search(q) else "oneq",
        "enum" if _ENUMERATION.search(q) else "noenum",
    )


class TaskConfigCache:
    """Recalls/records a recommended vote count (N) keyed by a question's
    structural signature, using UniversalPredictor's existing credibility
    mechanism -- not a new prediction algorithm, just a specific
    application of the one already built.
    """

    def __init__(self, min_confidence: float = 0.3):
        # context_length covers the signature (6 symbols) plus a little
        # slack; CTW blending across depths handles the rest, same as any
        # other use of this predictor.
        self._predictor = UniversalPredictor(context_length=8, min_confidence=min_confidence)

    def recall_n(self, question: str) -> tuple[Optional[int], float]:
        """Returns (recommended_n, confidence), or (None, 0.0) if the
        trie has no confident recommendation yet -- callers must have a
        safe default for this case, this never fails loudly."""
        sig = _feature_signature(question)
        try:
            pred = self._predictor.predict_next(list(sig))
        except Exception:
            return None, 0.0
        if pred is None or not isinstance(pred, int):
            return None, 0.0
        _, conf = self._predictor.predict()
        return pred, conf

    def record_outcome(self, question: str, actual_n_used: int) -> None:
        """Call after a task completes with however many votes it
        actually took to resolve, so future similarly-shaped questions
        benefit from this observation."""
        bucket = _nearest_bucket(actual_n_used)
        sig = _feature_signature(question)
        for item in sig:
            self._predictor.observe(item)
        self._predictor.observe(bucket)
        self._predictor.feedback(bucket)
