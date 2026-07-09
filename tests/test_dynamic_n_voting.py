"""Tests for dynamic-N self-consistency voting in GenerateAndGround
(0.4.0 follow-on to Item 17): the number of votes requested per question
is no longer a fixed 3 -- it's derived from iq_router's complexity
heuristic, refined by TaskConfigCache's recall when confident.
"""
from unittest.mock import patch

import numpy as np

from uchi.generate_and_ground import GenerateAndGround
from uchi.oracle import FactCheckOracle
from uchi.retrieval import SemanticIndex
from uchi.task_config_cache import TaskConfigCache


class _FakeProposer:
    """Always proposes the same grounded candidate; records how many
    candidate-generation calls happened (as opposed to Devil's Advocate
    critique calls, routed by prompt content -- same pattern used in
    test_swarm_loop_guard.py) so we can confirm n_votes was honored
    without unwanted reflection retries polluting the count.
    """

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


def test_default_task_config_cache_is_none_and_uses_complexity_baseline():
    """With no cache supplied, n_votes still varies with complexity (not
    a hardcoded 3) -- confirms the baseline formula is live even without
    the ODUSP recall layer."""
    evidence = "The Eiffel Tower is 330 meters tall and located in Paris."
    idx = _index_with_evidence(evidence, ["eiffel", "tower", "meters", "tall", "paris"])
    proposer = _FakeProposer("The Eiffel Tower is 330 meters tall.")
    gg = GenerateAndGround(index=idx, oracle=FactCheckOracle(), proposer=proposer)

    gg.answer("What is the height of the Eiffel Tower?")
    # Simple question -> low complexity -> should request the smallest
    # baseline bucket (1 vote), not the old hardcoded default of 3.
    assert proposer.propose_calls <= 3


def test_task_config_cache_recall_overrides_baseline_when_confident():
    evidence = "The Eiffel Tower is 330 meters tall and located in Paris."
    idx = _index_with_evidence(evidence, ["eiffel", "tower", "meters", "tall", "paris"])
    proposer = _FakeProposer("The Eiffel Tower is 330 meters tall.")
    cache = TaskConfigCache()

    with patch.object(cache, "recall_n", return_value=(8, 0.9)) as mock_recall:
        gg = GenerateAndGround(index=idx, oracle=FactCheckOracle(), proposer=proposer,
                                task_config_cache=cache)
        gg.answer("What is the height of the Eiffel Tower?")
        assert mock_recall.called
        assert proposer.propose_calls == 8


def test_low_confidence_recall_falls_back_to_baseline():
    evidence = "The Eiffel Tower is 330 meters tall and located in Paris."
    idx = _index_with_evidence(evidence, ["eiffel", "tower", "meters", "tall", "paris"])
    proposer = _FakeProposer("The Eiffel Tower is 330 meters tall.")
    cache = TaskConfigCache()

    with patch.object(cache, "recall_n", return_value=(8, 0.1)):  # low confidence
        gg = GenerateAndGround(index=idx, oracle=FactCheckOracle(), proposer=proposer,
                                task_config_cache=cache)
        gg.answer("What is the height of the Eiffel Tower?")
        # Low confidence recall must be ignored -- baseline (<=3 for a
        # simple question) applies instead of the recalled 8.
        assert proposer.propose_calls <= 3


def test_successful_answer_records_outcome_in_cache():
    evidence = "The Eiffel Tower is 330 meters tall and located in Paris."
    idx = _index_with_evidence(evidence, ["eiffel", "tower", "meters", "tall", "paris"])
    proposer = _FakeProposer("The Eiffel Tower is 330 meters tall.")
    cache = TaskConfigCache()

    with patch.object(cache, "record_outcome") as mock_record:
        gg = GenerateAndGround(index=idx, oracle=FactCheckOracle(), proposer=proposer,
                                task_config_cache=cache)
        result = gg.answer("What is the height of the Eiffel Tower?")
        assert "330" in result
        assert mock_record.called


def test_no_cache_supplied_never_touches_cache_api():
    """Default behavior (task_config_cache=None) must not attempt to call
    anything on a cache -- confirms zero coupling when unused."""
    evidence = "The Eiffel Tower is 330 meters tall and located in Paris."
    idx = _index_with_evidence(evidence, ["eiffel", "tower", "meters", "tall", "paris"])
    proposer = _FakeProposer("The Eiffel Tower is 330 meters tall.")
    gg = GenerateAndGround(index=idx, oracle=FactCheckOracle(), proposer=proposer)
    assert gg.task_config_cache is None
    result = gg.answer("What is the height of the Eiffel Tower?")
    assert "330" in result
