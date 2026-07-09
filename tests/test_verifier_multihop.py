"""Tests for verifier_train.py's synthetic multi-hop transitive examples
(0.4.0 follow-on to Item 17).

MNLI/SNLI are both single-premise; neither teaches "combine two stated
facts into a transitive conclusion" as a skill. This generates synthetic
(RuleTaker/ProofWriter-style) multi-premise examples for that specific gap.

Independently verified against RelationalTransitivityChecker (not just
trusted): a real bug was caught this way during development -- the relation
was originally being re-picked on every fact() call instead of once per
example, so premises ended up about unrelated attributes (e.g. "richer" for
one fact, "higher" for the other), silently producing wrong labels. Fixed,
then re-verified with zero mismatches across 1000+ generated examples
restricted to the vocabulary the deterministic checker actually recognizes
(the checker's own vocabulary is deliberately narrower than the
generator's, so this dataset can teach the neural classifier relation
words the checker doesn't know -- comparing against words it CAN parse is
the fair, apples-to-apples check).
"""
import re
from collections import Counter

from uchi.flux.verifier_train import generate_multihop_examples
from uchi.relational_reasoning import RelationalTransitivityChecker, _KNOWN_COMPARATIVES

_SPLIT = re.compile(r"(?<=[.!?])\s+")
_ALL_COMPARATIVE_WORDS = {
    w for pair in [
        ("taller", "shorter"), ("older", "younger"), ("faster", "slower"),
        ("heavier", "lighter"), ("bigger", "smaller"), ("richer", "poorer"),
        ("stronger", "weaker"), ("higher", "lower"), ("earlier", "later"),
        ("wider", "narrower"), ("hotter", "colder"),
    ] for w in pair
}


def _uses_only_known_words(text: str) -> bool:
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    used = words & _ALL_COMPARATIVE_WORDS
    return bool(used) and used <= set(_KNOWN_COMPARATIVES.keys())


def test_labels_are_balanced():
    examples = generate_multihop_examples(300)
    counts = Counter(e["label"] for e in examples)
    assert counts[0] > 0 and counts[1] > 0 and counts[2] > 0
    # Roughly even three-way split (i % 3), not skewed toward one label.
    assert max(counts.values()) - min(counts.values()) <= 2


def test_deterministic_with_fixed_seed():
    a = generate_multihop_examples(50, seed=42)
    b = generate_multihop_examples(50, seed=42)
    assert a == b


def test_different_seeds_produce_different_examples():
    a = generate_multihop_examples(50, seed=1)
    b = generate_multihop_examples(50, seed=2)
    assert a != b


def test_premise_contains_two_distinct_sentences():
    examples = generate_multihop_examples(20)
    for e in examples:
        sentences = _SPLIT.split(e["premise"].strip())
        assert len(sentences) == 2


def test_labels_verified_against_deterministic_checker():
    """The actual correctness check: restricted to relation words the
    checker recognizes, every generated label must agree with the
    independent, already-verified deterministic checker. This is what
    caught the per-fact-call relation-picking bug during development."""
    examples = generate_multihop_examples(1500)
    checker = RelationalTransitivityChecker()

    checked = 0
    for e in examples:
        if not _uses_only_known_words(e["premise"] + " " + e["hypothesis"]):
            continue
        sentences = _SPLIT.split(e["premise"].strip())
        contradicted = checker.is_contradicted(e["hypothesis"], sentences)
        checked += 1
        if e["label"] == 2:
            assert contradicted, f"expected contradiction, checker disagreed: {e}"
        elif e["label"] == 0:
            assert not contradicted, f"expected entailment, checker says contradicted: {e}"

    # Sanity: the filter itself must actually be exercising real examples,
    # not silently checking zero of them.
    assert checked > 200
