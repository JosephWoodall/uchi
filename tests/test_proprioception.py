"""Tests for FluxProprioception (0.4.0 follow-on, experimental) and its
wiring into GenerateAndGround's dynamic-N voting.

NOT a substitute for the verifier -- see tasks/proprioception_experiment.md.
This checks the mechanism (unfitted degrades gracefully, state persists,
n_votes only ever increases from this signal, never decreases) rather
than re-running the real FLUX-hidden-state experiment, which needs a real
checkpoint and lives in scripts/fit_proprioception.py.
"""
import numpy as np
import torch

from uchi.oracle import FactCheckOracle
from uchi.proprioception import FluxProprioception
from uchi.generate_and_ground import GenerateAndGround
from uchi.retrieval import SemanticIndex


def test_unfitted_detector_never_flags_anything():
    prop = FluxProprioception()
    assert prop.detector.is_fitted is False


def test_state_dict_roundtrip():
    prop = FluxProprioception()
    fake_reps = torch.randn(20, 8)
    prop.detector.fit(fake_reps)
    prop.detector.threshold = 5.0

    state = prop.state_dict()
    restored = FluxProprioception()
    restored.load_state_dict(state)

    assert restored.detector.is_fitted is True
    assert restored.detector.threshold == 5.0
    v = torch.randn(8)
    assert restored.detector.distance(v) == prop.detector.distance(v)


class _FakeProprioceptionModel:
    """Stand-in FLUX model -- only needs to exist as a non-None sentinel
    for the wiring test; is_unfamiliar is mocked at the FluxProprioception
    level so this never actually runs a forward pass."""
    pass


class _FakeProprioception:
    def __init__(self, unfamiliar: bool):
        self._unfamiliar = unfamiliar
        self.calls = 0

    def is_unfamiliar(self, model, tokenizer, question):
        self.calls += 1
        return self._unfamiliar


class _FakeProposer:
    def __init__(self, answer):
        self.answer = answer
        self.propose_calls = 0

    def propose(self, prompt, evidence, think=False):
        if "Devil's Advocate" in prompt:
            return "PASS"
        self.propose_calls += 1
        return self.answer

    def plan(self, question):
        return None


def _index_with_evidence(evidence_text, vocab_words):
    w2i = {w: i for i, w in enumerate(vocab_words)}
    idx = SemanticIndex(w2i, np.eye(len(vocab_words), dtype=np.float32))
    idx.passages = [evidence_text]
    idx.retrieve = lambda q, k: [(evidence_text, 0.9)]
    return idx


def test_unfamiliar_question_raises_n_votes():
    evidence = "The Eiffel Tower is 330 meters tall."
    idx = _index_with_evidence(evidence, ["eiffel", "tower", "meters", "tall"])
    proposer = _FakeProposer("The Eiffel Tower is 330 meters tall.")
    prop = _FakeProprioception(unfamiliar=True)

    gg = GenerateAndGround(index=idx, oracle=FactCheckOracle(), proposer=proposer,
                            proprioception=prop, proprioception_model=_FakeProprioceptionModel(),
                            proprioception_tokenizer=object())
    gg.answer("What is the height of the Eiffel Tower?")
    assert prop.calls == 1
    # Simple question would otherwise get the smallest baseline (<=3);
    # flagged unfamiliar must raise it to the max bucket (8).
    assert proposer.propose_calls == 8


def test_familiar_question_does_not_raise_n_votes():
    evidence = "The Eiffel Tower is 330 meters tall."
    idx = _index_with_evidence(evidence, ["eiffel", "tower", "meters", "tall"])
    proposer = _FakeProposer("The Eiffel Tower is 330 meters tall.")
    prop = _FakeProprioception(unfamiliar=False)

    gg = GenerateAndGround(index=idx, oracle=FactCheckOracle(), proposer=proposer,
                            proprioception=prop, proprioception_model=_FakeProprioceptionModel(),
                            proprioception_tokenizer=object())
    gg.answer("What is the height of the Eiffel Tower?")
    assert prop.calls == 1
    assert proposer.propose_calls <= 3


def test_missing_proprioception_components_are_inert():
    """Only proprioception set, model/tokenizer missing -- must not crash,
    must not affect n_votes at all (matches Core.__init__'s all-or-nothing
    wiring: it only sets self.proprioception if the model also loaded)."""
    evidence = "The Eiffel Tower is 330 meters tall."
    idx = _index_with_evidence(evidence, ["eiffel", "tower", "meters", "tall"])
    proposer = _FakeProposer("The Eiffel Tower is 330 meters tall.")
    prop = _FakeProprioception(unfamiliar=True)

    gg = GenerateAndGround(index=idx, oracle=FactCheckOracle(), proposer=proposer,
                            proprioception=prop, proprioception_model=None,
                            proprioception_tokenizer=None)
    gg.answer("What is the height of the Eiffel Tower?")
    assert prop.calls == 0  # never even called -- model is None
    assert proposer.propose_calls <= 3


def test_proprioception_exception_fails_open():
    evidence = "The Eiffel Tower is 330 meters tall."
    idx = _index_with_evidence(evidence, ["eiffel", "tower", "meters", "tall"])
    proposer = _FakeProposer("The Eiffel Tower is 330 meters tall.")

    class _RaisingProprioception:
        def is_unfamiliar(self, model, tokenizer, question):
            raise RuntimeError("boom")

    gg = GenerateAndGround(index=idx, oracle=FactCheckOracle(), proposer=proposer,
                            proprioception=_RaisingProprioception(),
                            proprioception_model=_FakeProprioceptionModel(),
                            proprioception_tokenizer=object())
    result = gg.answer("What is the height of the Eiffel Tower?")
    assert "330" in result  # must not crash the whole answer path


def test_no_proprioception_supplied_is_default_and_inert():
    evidence = "The Eiffel Tower is 330 meters tall."
    idx = _index_with_evidence(evidence, ["eiffel", "tower", "meters", "tall"])
    proposer = _FakeProposer("The Eiffel Tower is 330 meters tall.")
    gg = GenerateAndGround(index=idx, oracle=FactCheckOracle(), proposer=proposer)
    assert gg.proprioception is None
    result = gg.answer("What is the height of the Eiffel Tower?")
    assert "330" in result
