"""numeric_plausibility.py — Numeric plausibility checking (0.4.0 Item 17).

Flags a claimed number as statistically implausible relative to real
numeric facts pulled from the semantic index's own ingested passages --
not a separate dataset, not fabricated. This generalizes the exact-match
numeric check: a claim's number doesn't need a directly competing value in
the specific retrieved evidence to be caught, as long as it's implausible
relative to the broader distribution of real numbers already in the brain.

The original plan for this was to reuse UniversalPredictor/AnomalyDetector
(already built for the /anomaly skill). Tested empirically before wiring it
in, and it doesn't work for this specific question: that predictor is a
CTW-style *sequential* anomaly detector -- it looks for values that are
surprising given the immediately preceding context in an ORDERED series
(sensor readings, prices over time). Numeric facts pulled from scattered,
unrelated ingested passages have no such order -- feeding it an IID pool of
disconnected numbers gives it nothing learnable, and every prediction comes
back equally "surprising" regardless of the actual value (verified: 55 and
50,000 scored within 0.05 of each other). The right tool for "is this an
outlier in a static pool of facts" is a simple, deterministic statistical
check -- median + MAD (median absolute deviation), robust to skew/outliers
in the fitted distribution, unlike mean/std. Kept fully auditable and
fit-on-real-data, same as everywhere else in this codebase; just not the
sequence predictor specifically, since it's the wrong statistical tool for
a question with no temporal structure. UniversalPredictor/AnomalyDetector
remains the right tool for genuinely sequential claims (a claimed trend or
time-series value) -- that connection wasn't ruled out, just this one.

Strictly an ADDITIONAL veto layer, same principle as the entailment
classifier (verifier_model.py) -- it can only reject a claim the
deterministic check already accepted, never accept one it rejected. An
unfitted checker has "no opinion" (never vetoes), the same graceful
degradation used everywhere else in this codebase when a component isn't
available (e.g. FluxProposer.load() -> None).
"""
from __future__ import annotations

import re
import statistics
from typing import Optional

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")

# Consistency constant: makes MAD comparable to a standard deviation for
# roughly normal-ish data (1 / Phi^-1(3/4)). Standard robust-statistics
# convention, not a tuned magic number.
_MAD_TO_STD = 1.4826


def extract_numeric_facts(passages: list[str]) -> list[float]:
    """Pull real numeric values out of ingested passages. This IS the
    fitting data -- numbers actually present in the brain, nothing
    fabricated or sourced elsewhere."""
    values: list[float] = []
    for text in passages:
        for m in _NUMBER.finditer(text):
            raw = m.group().replace(",", "")
            if raw in ("", "-", "."):
                continue
            try:
                v = float(raw)
            except ValueError:
                continue
            values.append(v)
    return values


class NumericPlausibilityChecker:
    """Median + MAD outlier check for "is this claimed number plausible".

    Gracefully inert until fit() succeeds with enough real data --
    is_plausible() always returns True (no veto) when unfitted, so this
    can be wired into the oracle before it's ever been fitted without
    changing behavior at all.
    """

    def __init__(self, min_facts: int = 50, z_threshold: float = 3.5):
        self._median: Optional[float] = None
        self._mad: Optional[float] = None
        self.min_facts = min_facts
        self.z_threshold = z_threshold

    @property
    def is_fitted(self) -> bool:
        return self._median is not None

    def fit_from_passages(self, passages: list[str]) -> bool:
        """Returns whether fitting actually happened (False if there
        weren't enough real numeric facts to fit on)."""
        values = extract_numeric_facts(passages)
        if len(values) < self.min_facts:
            return False
        self._median = statistics.median(values)
        mad = statistics.median(abs(v - self._median) for v in values)
        self._mad = mad or 1.0  # avoid division by zero if all values are identical
        return True

    def is_plausible(self, value: float) -> bool:
        """True (no veto) if not an outlier, or if unfitted."""
        if not self.is_fitted:
            return True
        try:
            robust_z = abs(value - self._median) / (_MAD_TO_STD * self._mad)
            return robust_z <= self.z_threshold
        except Exception:
            return True  # fail open -- additive layer, never the sole gate
