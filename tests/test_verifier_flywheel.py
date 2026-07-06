"""Tests for the 0.4.0 Item 17 verifier flywheel: detecting when a user's
next turn contradicts Uchi's own prior answer (reusing the entailment
checker, no new model), logging it, and exporting it as new, real,
verified training data for the next verifier training run.
"""
import json

from uchi.verifier_flywheel import VerifierFlywheel, load_flywheel_examples


class _FakeChecker:
    """Predictable is_contradiction for testing, no real model needed."""
    def __init__(self, contradicts_on=None):
        self.contradicts_on = contradicts_on or set()
        self.calls = []

    def is_contradiction(self, premise, hypothesis):
        self.calls.append((premise, hypothesis))
        return hypothesis in self.contradicts_on


def test_inactive_without_a_checker():
    fw = VerifierFlywheel()
    assert fw.is_active is False
    assert fw.check_for_correction("Q", "The tower is 330m tall.", "No, it's 500m.") is False
    assert fw.corrections == []


def test_detects_and_records_a_real_correction():
    checker = _FakeChecker(contradicts_on={"No, it's 500m."})
    fw = VerifierFlywheel(entailment_checker=checker)
    assert fw.is_active is True

    result = fw.check_for_correction("How tall is the tower?", "The tower is 330m tall.", "No, it's 500m.")
    assert result is True
    assert len(fw.corrections) == 1
    assert fw.corrections[0]["prior_answer"] == "The tower is 330m tall."
    assert fw.corrections[0]["correction"] == "No, it's 500m."


def test_non_correction_is_not_recorded():
    checker = _FakeChecker(contradicts_on=set())  # never contradicts
    fw = VerifierFlywheel(entailment_checker=checker)
    result = fw.check_for_correction("How tall is the tower?", "The tower is 330m tall.", "Thanks!")
    assert result is False
    assert fw.corrections == []


def test_empty_prior_answer_is_never_checked():
    """Nothing to correct if there was no prior answer -- must not call
    the checker at all (e.g. the very first turn of a conversation)."""
    checker = _FakeChecker(contradicts_on={"anything"})
    fw = VerifierFlywheel(entailment_checker=checker)
    result = fw.check_for_correction("Q", "", "anything")
    assert result is False
    assert checker.calls == []


def test_checker_exception_fails_open():
    class _BrokenChecker:
        def is_contradiction(self, premise, hypothesis):
            raise RuntimeError("boom")

    fw = VerifierFlywheel(entailment_checker=_BrokenChecker())
    # Must not raise -- an observational loop can't break the conversation.
    result = fw.check_for_correction("Q", "answer", "correction")
    assert result is False


def test_export_writes_real_examples_and_clears_buffer(tmp_path):
    checker = _FakeChecker(contradicts_on={"No, it's 500m."})
    fw = VerifierFlywheel(entailment_checker=checker)
    fw.check_for_correction("Q", "The tower is 330m tall.", "No, it's 500m.")

    out_path = str(tmp_path / "flywheel_corrections.jsonl")
    n = fw.export_training_examples(out_path)
    assert n == 1
    assert fw.corrections == []  # buffer cleared after export

    with open(out_path) as fh:
        row = json.loads(fh.readline())
    assert row["premise"] == "The tower is 330m tall."
    assert row["hypothesis"] == "No, it's 500m."
    assert row["label"] == 2  # contradiction


def test_export_with_no_corrections_writes_nothing(tmp_path):
    fw = VerifierFlywheel(entailment_checker=_FakeChecker())
    out_path = str(tmp_path / "empty.jsonl")
    n = fw.export_training_examples(out_path)
    assert n == 0
    import os
    assert not os.path.exists(out_path)


def test_load_flywheel_examples_reads_back_same_shape(tmp_path):
    checker = _FakeChecker(contradicts_on={"correction 1", "correction 2"})
    fw = VerifierFlywheel(entailment_checker=checker)
    fw.check_for_correction("Q1", "answer 1", "correction 1")
    fw.check_for_correction("Q2", "answer 2", "correction 2")

    path = str(tmp_path / "corrections.jsonl")
    fw.export_training_examples(path)

    examples = load_flywheel_examples(path)
    assert len(examples) == 2
    assert all(ex["label"] == 2 for ex in examples)
    assert {ex["hypothesis"] for ex in examples} == {"correction 1", "correction 2"}


def test_load_flywheel_examples_missing_file_returns_empty():
    assert load_flywheel_examples("/tmp/definitely_does_not_exist_flywheel.jsonl") == []


def test_load_flywheel_examples_respects_num_examples_limit(tmp_path):
    checker = _FakeChecker(contradicts_on={f"c{i}" for i in range(10)})
    fw = VerifierFlywheel(entailment_checker=checker)
    for i in range(10):
        fw.check_for_correction(f"Q{i}", f"a{i}", f"c{i}")
    path = str(tmp_path / "many.jsonl")
    fw.export_training_examples(path)

    limited = load_flywheel_examples(path, num_examples=3)
    assert len(limited) == 3
