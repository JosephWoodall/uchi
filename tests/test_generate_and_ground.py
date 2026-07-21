"""Tests for GenerateAndGround.answer()'s extractive-fallback handling.

Real bug found investigating 0.5.0 Item 9's MMLU/ARC 0% accuracy:
`_candidates()` always yields `self._extractive(question, evidence)` as an
unconditional last candidate. Because it IS a retrieved passage, it
trivially passes `FactCheckOracle.is_grounded()` (support against itself is
1.0) regardless of whether that passage actually answers the question --
confirmed live: "What is the capital of France?" confidently returned a
passage about "New France" (the historical Quebec colony), because both
share the literal word "france". When every genuinely generated candidate
fails grounding (the common case against an undertrained proposer) and the
extractive fallback is the ONLY thing that grounds, that means no real
synthesis was ever verified -- Uchi's own stated principle is to abstain
rather than confabulate, so this must not be returned as a confident answer.
"""
import numpy as np

from uchi.generate_and_ground import _ABSTAIN, GenerateAndGround
from uchi.retrieval import SemanticIndex


class _GibberishProposer:
    """Models an undertrained proposer: never produces anything the oracle
    considers grounded, so only the extractive fallback can ever pass."""

    def propose(self, question, evidence, think=False):
        return "completely unrelated gibberish text with no support at all"

    def plan(self, question):
        return None


class _SynthesizingProposer:
    """Models a genuinely working proposer: produces a real, differently-
    phrased (not identical to the extractive fallback), grounded answer."""

    def propose(self, question, evidence, think=False):
        return "Quebec City was the capital."

    def plan(self, question):
        return None


def _build_index_with_ambiguous_entity():
    # Tiny, hand-built, orthogonal-vector index so cosine similarity is
    # exact and controllable -- no real embeddings file needed.
    w2i = {"france": 0, "capital": 1, "quebec": 2, "city": 3}
    E = np.eye(4, dtype=np.float32)
    idx = SemanticIndex(w2i, E)
    idx.add(["The capital of New France was Quebec City."])
    return idx


def test_extractive_only_answer_now_abstains_instead_of_confabulating():
    index = _build_index_with_ambiguous_entity()
    gag = GenerateAndGround(index=index, proposer=_GibberishProposer(), min_sim=0.3)

    result = gag.answer("What is the capital of France?")

    assert result == _ABSTAIN


def test_real_synthesized_grounded_answer_still_wins_normally():
    """The fix must not block genuine synthesis -- only the
    extractive-fallback-is-the-sole-survivor case should abstain."""
    index = _build_index_with_ambiguous_entity()
    gag = GenerateAndGround(index=index, proposer=_SynthesizingProposer(), min_sim=0.3)

    result = gag.answer("What is the capital of France?")

    assert result == "Quebec City was the capital."


def test_extractive_fallback_helper_itself_is_unchanged():
    """_extractive() itself is a legitimate, deliberate helper (used when
    no proposer is configured at all) -- this fix only changes whether its
    output is trusted as a confident final answer when nothing else
    grounds, not the helper's own behavior."""
    index = _build_index_with_ambiguous_entity()
    gag = GenerateAndGround(index=index, proposer=None, min_sim=0.3)
    evidence = index.retrieve("What is the capital of France?", gag.retrieve_k)

    assert gag._extractive("What is the capital of France?", evidence) == \
        "The capital of New France was Quebec City."
