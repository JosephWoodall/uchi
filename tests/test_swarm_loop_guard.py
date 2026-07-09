"""Tests for the 0.4.0 Item 6 gap closure: self-healing loop prevention
wired into SwarmSynthesizer's delegation path, not just tool calls.
"""
from uchi.swarm import SwarmSynthesizer

COMPLEX_QUESTION = "Compare the GDP of Germany versus France, and also their population, and also area?"


class _FakeProposer:
    """Routes propose() calls by prompt shape: the decomposition prompt
    vs. the final aggregation prompt, so a single fake can drive both
    call sites SwarmSynthesizer makes to self.proposer.propose()."""

    def __init__(self, decompose_response, aggregate_response=None, aggregate_raises=False):
        self.decompose_response = decompose_response
        self.aggregate_response = aggregate_response
        self.aggregate_raises = aggregate_raises
        self.decompose_calls = 0
        self.aggregate_calls = 0

    def propose(self, prompt, evidence):
        if "Break the following" in prompt:
            self.decompose_calls += 1
            return self.decompose_response
        self.aggregate_calls += 1
        if self.aggregate_raises:
            raise RuntimeError("aggregation exploded")
        return self.aggregate_response


class _FakeQA:
    """Stands in for GenerateAndGround. answer() is called both for each
    sub-question (during dispatch) and for the single-pipeline fallback --
    sub_answers maps sub-question text to a canned result; anything else
    (including the original question, for the fallback path) gets
    fallback_answer."""

    def __init__(self, proposer, sub_answers=None, fallback_answer="fallback answer"):
        self.proposer = proposer
        self.sub_answers = sub_answers or {}
        self.fallback_answer = fallback_answer
        self.calls = []

    def answer(self, question, callback=None):
        self.calls.append(question)
        return self.sub_answers.get(question, self.fallback_answer)


def test_successful_delegation_is_not_penalized():
    proposer = _FakeProposer(
        decompose_response='["What is Germany GDP?", "What is France GDP?"]',
        aggregate_response="Germany and France GDP compared: ...",
    )
    qa = _FakeQA(proposer, sub_answers={
        "What is Germany GDP?": "Germany GDP is 4T.",
        "What is France GDP?": "France GDP is 3T.",
    })
    swarm = SwarmSynthesizer(qa)

    result = swarm.answer(COMPLEX_QUESTION)
    assert result == "Germany and France GDP compared: ..."
    assert not swarm.loop_guard.is_penalized(COMPLEX_QUESTION)


def test_all_subanswers_failing_records_failure_and_falls_back():
    proposer = _FakeProposer(
        decompose_response='["What is Germany GDP?", "What is France GDP?"]',
    )
    qa = _FakeQA(proposer, sub_answers={
        "What is Germany GDP?": "I don't have grounded knowledge to answer that.",
        "What is France GDP?": "I don't have grounded knowledge to answer that.",
    })
    swarm = SwarmSynthesizer(qa)

    result = swarm.answer(COMPLEX_QUESTION)
    assert result == "fallback answer"
    assert swarm.loop_guard.is_penalized(COMPLEX_QUESTION)
    assert swarm.loop_guard.failure_count(COMPLEX_QUESTION) == 1


def test_second_identical_failure_skips_redecomposition_entirely():
    """The core fix: a repeat of an already-failed question must not pay
    for _decompose()'s FLUX round-trip again -- it should fall straight
    to the single pipeline."""
    proposer = _FakeProposer(
        decompose_response='["What is Germany GDP?", "What is France GDP?"]',
    )
    qa = _FakeQA(proposer, sub_answers={
        "What is Germany GDP?": "I don't have grounded knowledge to answer that.",
        "What is France GDP?": "I don't have grounded knowledge to answer that.",
    })
    swarm = SwarmSynthesizer(qa)

    first = swarm.answer(COMPLEX_QUESTION)
    assert first == "fallback answer"
    assert proposer.decompose_calls == 1

    second = swarm.answer(COMPLEX_QUESTION)
    assert second == "fallback answer"
    assert proposer.decompose_calls == 1, "retry re-ran _decompose() instead of being blocked by the loop guard"


def test_aggregation_failure_records_failure():
    proposer = _FakeProposer(
        decompose_response='["What is Germany GDP?", "What is France GDP?"]',
        aggregate_raises=True,
    )
    qa = _FakeQA(proposer, sub_answers={
        "What is Germany GDP?": "Germany GDP is 4T.",
        "What is France GDP?": "France GDP is 3T.",
    })
    swarm = SwarmSynthesizer(qa)

    result = swarm.answer(COMPLEX_QUESTION)
    assert result == "I am sorry, the swarm failed to aggregate a final answer."
    assert swarm.loop_guard.is_penalized(COMPLEX_QUESTION)


def test_different_question_is_unaffected_by_another_questions_failure():
    other_question = "What is the tallest mountain, and also the deepest ocean, and also the longest river?"
    proposer = _FakeProposer(
        decompose_response='["What is Germany GDP?", "What is France GDP?"]',
    )
    qa = _FakeQA(proposer, sub_answers={
        "What is Germany GDP?": "I don't have grounded knowledge to answer that.",
        "What is France GDP?": "I don't have grounded knowledge to answer that.",
    })
    swarm = SwarmSynthesizer(qa)

    swarm.answer(COMPLEX_QUESTION)
    assert swarm.loop_guard.is_penalized(COMPLEX_QUESTION)
    assert not swarm.loop_guard.is_penalized(other_question)


def test_success_after_a_prior_failure_clears_the_penalty():
    proposer = _FakeProposer(
        decompose_response='["What is Germany GDP?", "What is France GDP?"]',
        aggregate_response="synthesized answer",
    )
    qa = _FakeQA(proposer, sub_answers={
        "What is Germany GDP?": "I don't have grounded knowledge to answer that.",
        "What is France GDP?": "I don't have grounded knowledge to answer that.",
    })
    swarm = SwarmSynthesizer(qa)

    swarm.answer(COMPLEX_QUESTION)
    assert swarm.loop_guard.is_penalized(COMPLEX_QUESTION)

    # Simulate underlying data changing (e.g. web search/ingest happened)
    # so the same sub-questions now resolve successfully.
    qa.sub_answers["What is Germany GDP?"] = "Germany GDP is 4T."
    qa.sub_answers["What is France GDP?"] = "France GDP is 3T."
    swarm.loop_guard.record_success(COMPLEX_QUESTION)  # what a retry-after-fix path would do

    result = swarm.answer(COMPLEX_QUESTION)
    assert result == "synthesized answer"
