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

No-evidence relaxation (0.4.0 finding): when *evidence* is empty, the standard
overlap check rejects every claim, including ones that never asserted anything
checkable in the first place -- "Hello! How can I help you today?" has no
evidence to check ANY word against, so it gets vetoed the same way a fabricated
fact would, and Uchi abstains on a bare greeting. ``is_grounded`` now special-
cases this: with no evidence, only a claim's *specific* content (proper nouns,
numbers) needs support; a claim built entirely from generic vocabulary has
nothing checkable in it and is safe to emit, while a claim naming something
specific with zero evidence behind it is still rejected exactly as before. This
never changes behavior when real evidence exists -- the standard strict check
is untouched for every normal factual question.
"""
from __future__ import annotations

import re
import time

_WORD = re.compile(r"[a-z0-9']+")
_STOP = frozenset(
    "the a an of to in and or is are was were be been being for on at by with as "
    "that this these those it its there here what which who whom how why when where "
    "do does did can could would should will may might must not no nor but if then "
    "than into over under from about your you i we they he she him her his their our "
    "one two three some any all each more most also".split()
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_ALNUM = re.compile(r"[^A-Za-z0-9]")


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
        # Audit trail for the no-evidence relaxation below -- every time it
        # fires, logged here so the decision is inspectable rather than a
        # silent carve-out. Feeds a future reflection/refinement pass (see
        # the module docstring); not wired into the OTel span exporter yet
        # since nothing consumes this shape as a span today.
        self.relaxed_pass_log: list[dict] = []

    @staticmethod
    def _terms(text: str) -> list[str]:
        return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2]

    @staticmethod
    def _specific_terms(text: str) -> list[str]:
        """Terms in *text* that assert something specific and checkable:
        proper nouns (capitalized, excluding a sentence's first word, since
        that's capitalized regardless of being a proper noun) and any token
        containing a digit (numbers, dates, quantities). Ordinary vocabulary
        used in generic or social speech -- "welcome", "help", "great" --
        isn't inherently checkable and is deliberately excluded: nobody
        fact-checks the word "welcome". The pronoun "I" is also excluded --
        it's the one English word always capitalized regardless of
        position, a spelling convention rather than a signal of specificity.
        """
        specific = []
        for sentence in _SENTENCE_SPLIT.split(text):
            words = sentence.split()
            for i, w in enumerate(words):
                cleaned = _ALNUM.sub("", w)
                if not cleaned or cleaned == "I":
                    continue
                has_digit = any(c.isdigit() for c in cleaned)
                is_proper_noun = i > 0 and cleaned[0].isupper()
                if has_digit or is_proper_noun:
                    specific.append(cleaned.lower())
        return specific

    @staticmethod
    def _evidence_vocab(evidence: list[str]) -> set[str]:
        vocab: set[str] = set()
        for e in evidence:
            vocab.update(_WORD.findall(e.lower()))
        return vocab

    def support(self, claim: str, evidence: list[str]) -> float:
        """Return the fraction of the claim's salient terms supported by evidence.

        1.0 = every salient term is present in the retrieved evidence;
        0.0 = none are (a fabrication with respect to what the brain knows).
        """
        terms = self._terms(claim)
        if not terms:
            return 0.0
        vocab = self._evidence_vocab(evidence)
        return sum(1 for w in terms if w in vocab) / len(terms)

    def is_grounded(self, claim: str, evidence: list[str]) -> bool:
        """True iff the claim clears the support threshold — safe to emit.

        See the module docstring for the no-evidence relaxation: when
        *evidence* contains nothing relevant, this checks only the claim's
        specific content instead of every salient word.
        """
        vocab = self._evidence_vocab(evidence)
        if not vocab:
            specific = self._specific_terms(claim)
            verdict = not specific
            self.relaxed_pass_log.append({
                "timestamp": time.time(),
                "claim": claim,
                "specific_terms": specific,
                "verdict": verdict,
            })
            return verdict
        return self.support(claim, evidence) >= self.min_support
