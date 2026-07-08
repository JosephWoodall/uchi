"""Tests for RelationalTransitivityChecker (0.4.0 follow-on to Item 17).

The gap this closes was demonstrated empirically before writing any code:
FactCheckOracle's word-overlap check (and the entailment classifier) accept
a claim purely on term-overlap/surface consistency, with zero notion of
relational direction. Given evidence "A is taller than B" and "B is taller
than C", it accepted "A is taller than C" (valid), "C is taller than A"
(the reversed, wrong claim), and "A is taller than A" (nonsense) equally.
"""
from uchi.oracle import FactCheckOracle
from uchi.relational_reasoning import (
    RelationalTransitivityChecker,
    extract_attribute_value,
    extract_comparative,
)


def test_extract_comparative_basic():
    fact = extract_comparative("Building A is taller than Building B.")
    assert fact.subject == "building a"
    assert fact.relation == "height"
    assert fact.obj == "building b"
    assert fact.direction == 1


def test_extract_comparative_inverse_word():
    fact = extract_comparative("Building B is shorter than Building A.")
    assert fact.relation == "height"
    assert fact.direction == -1


def test_extract_comparative_generic_more_less():
    fact = extract_comparative("Widget X is more expensive than Widget Y.")
    assert fact.subject == "widget x"
    assert fact.relation == "expensive"
    assert fact.direction == 1


def test_extract_comparative_temporal():
    fact = extract_comparative("The battle happened before the treaty.")
    assert fact.relation == "time"
    assert fact.direction == -1


def test_non_comparative_sentence_does_not_parse():
    assert extract_comparative("The building has a red roof.") is None


def test_valid_transitive_conclusion_not_contradicted():
    checker = RelationalTransitivityChecker()
    evidence = ["Building A is taller than Building B.", "Building B is taller than Building C."]
    assert checker.is_contradicted("Building A is taller than Building C.", evidence) is False


def test_reversed_conclusion_is_contradicted():
    checker = RelationalTransitivityChecker()
    evidence = ["Building A is taller than Building B.", "Building B is taller than Building C."]
    assert checker.is_contradicted("Building C is taller than Building A.", evidence) is True


def test_self_comparison_is_contradicted():
    checker = RelationalTransitivityChecker()
    evidence = ["Building A is taller than Building B."]
    assert checker.is_contradicted("Building A is taller than Building A.", evidence) is True


def test_unconnected_entities_abstain_not_falsely_veto():
    checker = RelationalTransitivityChecker()
    evidence = ["Building A is taller than Building B."]
    assert checker.is_contradicted("Building D is taller than Building E.", evidence) is False


def test_unknown_relation_abstains():
    checker = RelationalTransitivityChecker()
    evidence = ["Building A is taller than Building B."]
    assert checker.is_contradicted("Building A is older than Building B.", evidence) is False


def test_mixed_direction_chain_still_resolves():
    checker = RelationalTransitivityChecker()
    # B is SHORTER than A (i.e. A > B), and B is taller than C (B > C) -> A > C
    evidence = ["Building B is shorter than Building A.", "Building B is taller than Building C."]
    assert checker.is_contradicted("Building A is taller than Building C.", evidence) is False
    assert checker.is_contradicted("Building C is taller than Building A.", evidence) is True


# ── Numeric attribute-value extraction (no comparative wording needed) ────

def test_extract_attribute_value_basic():
    assert extract_attribute_value("Building A is 442 meters tall.") == ("building a", "height", 442.0)


def test_extract_attribute_value_age():
    assert extract_attribute_value("Alice is 30 years old.") == ("alice", "age", 30.0)


def test_extract_attribute_value_non_matching_sentence():
    assert extract_attribute_value("The building has a red roof.") is None


def test_extract_attribute_value_unknown_attribute_word():
    # "shiny" isn't in _ATTR_TO_RELATION -- must not fabricate a relation.
    assert extract_attribute_value("The car is 5 years shiny.") is None


def test_pure_numeric_evidence_no_comparative_wording_at_all():
    """The real-world common case: evidence states measurements, not
    explicit comparisons -- this must still derive the relation."""
    checker = RelationalTransitivityChecker()
    evidence = ["Building A is 442 meters tall.", "Building B is 330 meters tall."]
    assert checker.is_contradicted("Building A is taller than Building B.", evidence) is False
    assert checker.is_contradicted("Building B is taller than Building A.", evidence) is True


def test_numeric_and_comparative_evidence_compose_through_one_closure():
    """Three-entity chain where the A-B edge comes from numeric facts and
    the B-C edge comes from an explicit comparative sentence -- both
    sources must feed the same transitive closure."""
    checker = RelationalTransitivityChecker()
    evidence = [
        "Building A is 442 meters tall.",
        "Building B is 330 meters tall.",
        "Building B is taller than Building C.",
    ]
    assert checker.is_contradicted("Building A is taller than Building C.", evidence) is False
    assert checker.is_contradicted("Building C is taller than Building A.", evidence) is True


def test_equal_numeric_values_derive_no_direction():
    checker = RelationalTransitivityChecker()
    evidence = ["Building A is 442 meters tall.", "Building B is 442 meters tall."]
    assert checker.is_contradicted("Building A is taller than Building B.", evidence) is False


# ── Trailing-clause and "Compared to" phrasing (found via cross-checking
# synthetic multi-hop training data against this checker -- see
# verifier_train.py's generate_multihop_examples) ─────────────────────────

def test_trailing_clause_does_not_get_swallowed_into_entity_name():
    """'Building A' and 'Building A, based on the available data' must
    resolve to the same node, not two different, unconnected ones."""
    fact = extract_comparative("Grace is younger than the black car, based on the available data.")
    assert fact.obj == "the black car"


def test_compared_to_phrasing_without_the_word_than():
    fact = extract_comparative("Compared to Grace, Kilimanjaro is younger.")
    assert fact is not None
    assert fact.subject == "kilimanjaro"
    assert fact.obj == "grace"
    assert fact.relation == "age"
    assert fact.direction == -1


def test_compared_to_and_trailing_clause_compose_through_closure():
    """The exact scenario that surfaced both gaps: a trailing-clause
    sentence and a "Compared to" sentence, chained together."""
    checker = RelationalTransitivityChecker()
    evidence = [
        "Grace is younger than the black car, based on the available data.",
        "Compared to Grace, Kilimanjaro is younger.",
    ]
    assert checker.is_contradicted("Compared to Kilimanjaro, the black car is younger.", evidence) is True
    assert checker.is_contradicted("Compared to the black car, Kilimanjaro is younger.", evidence) is False


# ── Integration: FactCheckOracle wiring ────────────────────────────────────

def test_oracle_vetoes_reversed_transitive_claim():
    oracle = FactCheckOracle(relational_checker=RelationalTransitivityChecker())
    evidence = ["Building A is taller than Building B.", "Building B is taller than Building C."]
    assert oracle.is_grounded("Building A is taller than Building C.", evidence) is True
    assert oracle.is_grounded("Building C is taller than Building A.", evidence) is False
    assert oracle.is_grounded("Building A is taller than Building A.", evidence) is False


def test_oracle_without_relational_checker_behaves_as_before():
    """Zero behavior change when relational_checker isn't supplied -- same
    discipline as every other optional veto layer in this file."""
    oracle = FactCheckOracle()
    evidence = ["Building A is taller than Building B.", "Building B is taller than Building C."]
    # Without the checker, the reversed claim passes (the original,
    # demonstrated gap) -- confirms the checker is additive, not baked in.
    assert oracle.is_grounded("Building C is taller than Building A.", evidence) is True


def test_relational_checker_cannot_override_a_deterministic_rejection():
    """Core safety property, tested directly: even if the relational
    checker would raise no objection, a claim the deterministic
    word-overlap check already rejected stays rejected."""
    oracle = FactCheckOracle(relational_checker=RelationalTransitivityChecker())
    evidence = ["Building A is taller than Building B."]
    # Completely unrelated claim -- fails word-overlap, and the relational
    # checker has no opinion on it (different entities) -- must stay rejected.
    assert oracle.is_grounded("The moon is made of cheese and unicorns dance nightly.", evidence) is False
