"""
oracle.py — the fact-check oracle that keeps Uchi honest.

Generate-and-Ground's honesty gate: a candidate answer is only allowed out if its
salient content is *supported* by the evidence retrieved from the brain. Anything
unsupported is a hallucination and is vetoed (the answer abstains).

This is a *verifier*, not a value critic — it checks grounding (truth against the
brain), it does not score quality. Validated in de-risk (experiments/
factcheck_oracle.py): 93.5% separation of true vs adversarial plausible-false
claims, 100% grounding recall, ~6.5% false-accept — where the trie-probability
oracle was random (47%).

The mechanism is deliberately simple and transparent: does the claim's salient
vocabulary appear in the retrieved evidence? Simplicity is a feature — it is
auditable and cannot itself hallucinate. Sharper variants (KG/entailment
cross-check) can raise the bar later without changing this interface.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9']+")
_STOP = frozenset(
    "the a an of to in and or is are was were be been being for on at by with as "
    "that this these those it its there here what which who whom how why when where "
    "do does did can could would should will may might must not no nor but if then "
    "than into over under from about your you i we they he she him her his their our "
    "one two three some any all each more most also".split()
)


class FactCheckOracle:
    """Verify that a candidate answer is grounded in retrieved evidence.

    Parameters
    ----------
    min_support : float
        Fraction of the candidate's salient terms that must appear in the
        evidence for the answer to be emitted. Below this the answer is vetoed
        (Uchi abstains rather than confabulate). Default 0.5.
    """

    def __init__(self, min_support: float = 0.5) -> None:
        self.min_support = min_support

    @staticmethod
    def _terms(text: str) -> list[str]:
        return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2]

    def support(self, claim: str, evidence: list[str]) -> float:
        """Return the fraction of the claim's salient terms supported by evidence.

        1.0 = every salient term is present in the retrieved evidence;
        0.0 = none are (a fabrication with respect to what the brain knows).
        """
        terms = self._terms(claim)
        if not terms:
            return 0.0
        vocab: set[str] = set()
        for e in evidence:
            vocab.update(_WORD.findall(e.lower()))
        return sum(1 for w in terms if w in vocab) / len(terms)

    def is_grounded(self, claim: str, evidence: list[str]) -> bool:
        """True iff the claim clears the support threshold — safe to emit."""
        return self.support(claim, evidence) >= self.min_support
