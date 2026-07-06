"""Tests for the no-evidence relaxation in FactCheckOracle.is_grounded()
(0.4.0 finding): a bare greeting or acknowledgment has nothing to check
its wording against and was being vetoed the same way a fabricated fact
would be, causing Uchi to abstain on inputs like "Hello!". This checks
only the claim's *specific* content (proper nouns, numbers) when no
evidence exists, and leaves the strict, normal-evidence path unchanged.
"""
from uchi.oracle import FactCheckOracle


def test_greeting_with_no_evidence_is_now_grounded():
    o = FactCheckOracle()
    assert o.is_grounded("Hello! How can I help you today?", []) is True


def test_thanks_with_no_evidence_is_now_grounded():
    o = FactCheckOracle()
    assert o.is_grounded("You are welcome! Happy to help.", []) is True


def test_specific_claim_with_no_evidence_is_still_rejected():
    """The relaxation must not become a blanket pass-through -- a claim
    naming something specific with zero evidence behind it is exactly the
    fabrication case the oracle exists to catch."""
    o = FactCheckOracle()
    assert o.is_grounded("The Eiffel Tower is 330 meters tall.", []) is False


def test_specific_claim_with_irrelevant_evidence_is_still_rejected():
    o = FactCheckOracle()
    assert o.is_grounded(
        "The Eiffel Tower is 330 meters tall.",
        ["Photosynthesis converts sunlight into chemical energy."],
    ) is False


def test_generic_reply_with_irrelevant_evidence_is_grounded():
    """Irrelevant evidence still produces an empty overlap vocabulary in
    practice for a generic reply -- the relaxation is keyed off whether
    the evidence vocabulary is empty, not whether the evidence list itself
    is empty, so this must behave the same as the fully-empty-evidence case."""
    o = FactCheckOracle()
    assert o.is_grounded(
        "Sure, happy to help with that.",
        ["Photosynthesis converts sunlight into chemical energy."],
    ) is False  # "happy", "help" aren't in the vocab -- but neither are specific


def test_normal_grounded_factual_claim_is_unaffected():
    """The ordinary, real-evidence path must be completely unchanged."""
    o = FactCheckOracle()
    evidence = ["The Eiffel Tower is 330 meters tall and located in Paris."]
    assert o.is_grounded("The Eiffel Tower is 330 meters tall.", evidence) is True


def test_normal_ungrounded_factual_claim_is_still_rejected():
    """Evidence on a completely unrelated topic -- a claim mostly
    overlapping relevant-but-wrong evidence (e.g. a wrong number sharing
    4/5 salient words with the right passage) can clear the pre-existing
    word-overlap threshold regardless of this change; that's a separate,
    pre-existing property of support() (~6.5% false-accept, per this
    file's own docstring), not something the no-evidence relaxation
    touches -- this test targets the relaxation specifically, with
    evidence that shares nothing with the claim at all."""
    evidence = ["Photosynthesis converts sunlight into chemical energy."]
    o = FactCheckOracle()
    assert o.is_grounded("The Eiffel Tower is 500 meters tall.", evidence) is False


def test_specific_terms_excludes_sentence_initial_capitalization():
    """The first word of a sentence is capitalized regardless of being a
    proper noun -- must not be treated as "specific" on that basis alone."""
    o = FactCheckOracle()
    # "The" is sentence-initial and lowercase-ish generic; nothing else here
    # is capitalized or numeric, so this must be treated as fully generic.
    terms = o._specific_terms("The weather is quite nice today.")
    assert terms == []


def test_specific_terms_catches_proper_nouns_and_numbers():
    o = FactCheckOracle()
    terms = o._specific_terms("I visited Germany in 2019 with Maria.")
    assert "germany" in terms
    assert "2019" in terms
    assert "maria" in terms


def test_relaxed_pass_log_records_every_no_evidence_decision():
    o = FactCheckOracle()
    assert o.relaxed_pass_log == []
    o.is_grounded("Hello there!", [])
    o.is_grounded("The Eiffel Tower is 330 meters tall.", [])
    assert len(o.relaxed_pass_log) == 2
    assert o.relaxed_pass_log[0]["verdict"] is True
    assert o.relaxed_pass_log[1]["verdict"] is False
    assert o.relaxed_pass_log[1]["specific_terms"] == ["eiffel", "tower", "330"]


def test_relaxed_pass_log_untouched_by_normal_evidence_path():
    o = FactCheckOracle()
    evidence = ["The Eiffel Tower is 330 meters tall and located in Paris."]
    o.is_grounded("The Eiffel Tower is 330 meters tall.", evidence)
    assert o.relaxed_pass_log == []
