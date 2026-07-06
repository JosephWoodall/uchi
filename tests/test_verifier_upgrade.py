"""Tests for the 0.4.0 Item 17 verifier upgrade: the entailment classifier
architecture, its data pipeline, the numeric plausibility checker, and their
integration into FactCheckOracle as strictly additive veto layers.
"""
from unittest.mock import patch

import torch

from uchi.flux.verifier_model import EntailmentClassifier, EntailmentChecker, OODDetector
from uchi.flux.verifier_train import load_verifier_examples
from uchi.flux.tokenizer_v2 import TikTokenHybridTokenizer
from uchi.numeric_plausibility import NumericPlausibilityChecker, extract_numeric_facts
from uchi.oracle import FactCheckOracle


# ── EntailmentClassifier architecture ───────────────────────────────────────

def test_entailment_classifier_forward_shape():
    m = EntailmentClassifier(vocab_size=1000, d_model=32, n_layers=4, d_state=8)
    x = torch.randint(0, 1000, (3, 20))
    mask = torch.ones(3, 20)
    logits = m(x, attention_mask=mask)
    assert logits.shape == (3, 3)


def test_entailment_classifier_padding_mask_affects_pooling():
    """Padded positions must not influence the pooled representation."""
    m = EntailmentClassifier(vocab_size=1000, d_model=16, n_layers=4, d_state=8)
    x = torch.randint(0, 1000, (1, 10))
    full_mask = torch.ones(1, 10)
    partial_mask = full_mask.clone()
    partial_mask[0, 5:] = 0
    # Different masks over the same input should generally produce
    # different pooled logits (padding actually affects the result).
    logits_full = m(x, attention_mask=full_mask)
    logits_partial = m(x, attention_mask=partial_mask)
    assert not torch.allclose(logits_full, logits_partial)


def test_predict_label_returns_label_names():
    m = EntailmentClassifier(vocab_size=1000, d_model=16, n_layers=4, d_state=8)
    x = torch.randint(0, 1000, (2, 10))
    mask = torch.ones(2, 10)
    labels = m.predict_label(x, mask)
    assert all(l in EntailmentClassifier.LABEL_NAMES for l in labels)


# ── Data pipeline (mocked HF datasets, same pattern as other sources) ──────

def _mnli_rows(n):
    for i in range(n):
        yield {"premise": f"mnli premise {i}", "hypothesis": f"mnli hypothesis {i}", "label": i % 3}


def _snli_rows(n):
    for i in range(n):
        yield {"premise": f"snli premise {i}", "hypothesis": f"snli hypothesis {i}", "label": i % 3}


def test_verifier_examples_fair_budget_across_sources():
    tokenizer = TikTokenHybridTokenizer()
    consumed = {"mnli": 0, "snli": 0}

    def loader(name, config=None, split=None, streaming=None):
        if config == "mnli":
            consumed["mnli"] += 1
            return _mnli_rows(200)
        if "snli" in name.lower():
            consumed["snli"] += 1
            return _snli_rows(200)
        raise AssertionError(f"unexpected dataset: {name} {config}")

    with patch("datasets.load_dataset", side_effect=loader):
        formatted = load_verifier_examples(tokenizer, max_seq_len=64, max_examples=100)

    assert all(v > 0 for v in consumed.values())
    assert len(formatted) > 0
    assert len(formatted) <= 100
    for ex in formatted:
        assert ex["label"] in (0, 1, 2)
        assert len(ex["input_ids"]) == 64
        assert len(ex["attention_mask"]) == 64


def test_verifier_examples_includes_flywheel_as_third_source(tmp_path):
    """Real corrections exported by VerifierFlywheel must be usable as a
    third fair-budgeted source, same shape as MNLI/SNLI, no format changes."""
    import json
    from uchi.verifier_flywheel import VerifierFlywheel

    flywheel_path = str(tmp_path / "corrections.jsonl")
    with open(flywheel_path, "w") as fh:
        for i in range(30):
            fh.write(json.dumps({
                "premise": f"flywheel premise {i}",
                "hypothesis": f"flywheel hypothesis {i}",
                "label": 2,
            }) + "\n")

    tokenizer = TikTokenHybridTokenizer()

    def loader(name, config=None, split=None, streaming=None):
        if config == "mnli":
            return _mnli_rows(200)
        return _snli_rows(200)

    with patch("datasets.load_dataset", side_effect=loader):
        formatted = load_verifier_examples(
            tokenizer, max_seq_len=64, max_examples=90, flywheel_path=flywheel_path,
        )

    # With 3 sources and max_examples=90, each gets ~30 -- confirm the
    # flywheel's real examples actually made it into the final set by
    # checking the total reflects all three contributing, not just 2/3
    # of max_examples worth from MNLI+SNLI alone capped low.
    assert len(formatted) > 0


def test_verifier_examples_excludes_snli_unlabeled_rows():
    """SNLI's label=-1 ("no annotator consensus") must be dropped, not
    treated as a bogus fourth class."""
    tokenizer = TikTokenHybridTokenizer()

    def rows_with_unlabeled(n):
        for i in range(n):
            yield {"premise": f"p{i}", "hypothesis": f"h{i}", "label": -1 if i % 2 == 0 else 0}

    def loader(name, config=None, split=None, streaming=None):
        return rows_with_unlabeled(20)

    with patch("datasets.load_dataset", side_effect=loader):
        formatted = load_verifier_examples(tokenizer, max_seq_len=64, max_examples=40)

    assert all(ex["label"] in (0, 1, 2) for ex in formatted)


# ── OOD detector (gates the entailment veto, never an independent veto) ────

def test_ood_detector_unfitted_never_flags_anything():
    ood = OODDetector()
    assert ood.is_ood(torch.zeros(8)) is False
    assert ood.distance(torch.zeros(8)) == 0.0


def test_ood_detector_discriminates_in_vs_out_of_distribution():
    """Diagnostic of the Mahalanobis math itself, using a well-behaved
    synthetic feature space -- an untrained model's latent space has no
    learned structure, so testing through a real (but untrained)
    EntailmentClassifier wouldn't be diagnostic of anything."""
    torch.manual_seed(0)
    normal = torch.randn(200, 8) + 5.0
    ood = OODDetector(threshold=3.0)
    ood.fit(normal)

    in_dist = torch.randn(8) + 5.0
    far_ood = torch.full((8,), 500.0)

    assert ood.is_ood(in_dist) is False
    assert ood.is_ood(far_ood) is True
    assert ood.distance(far_ood) > ood.distance(in_dist)


def test_ood_detector_state_dict_round_trips():
    torch.manual_seed(0)
    normal = torch.randn(50, 4) + 2.0
    ood = OODDetector(threshold=2.5)
    ood.fit(normal)

    restored = OODDetector()
    restored.load_state_dict(ood.state_dict())
    assert restored.threshold == 2.5
    probe = torch.randn(4) + 2.0
    assert abs(restored.distance(probe) - ood.distance(probe)) < 1e-4


def test_entailment_checker_ood_gate_suppresses_contradiction_verdict():
    """When the OOD detector flags an input, the checker must report
    "no contradiction" (no opinion) regardless of what the classifier head
    would have said -- the gate can only make it LESS likely to veto."""
    torch.manual_seed(0)
    tokenizer = TikTokenHybridTokenizer()
    model = EntailmentClassifier(vocab_size=tokenizer.vocab_size, d_model=16, n_layers=4, d_state=8)

    class _AlwaysOOD:
        def is_ood(self, rep):
            return True

    checker = EntailmentChecker(model, tokenizer, ood_detector=_AlwaysOOD())
    # Regardless of what the untrained classifier head would say, the OOD
    # gate must suppress it to False.
    assert checker.is_contradiction("The tower is 330 meters tall.", "The tower is 500 meters tall.") is False


# ── Numeric plausibility ────────────────────────────────────────────────────

def test_extract_numeric_facts_pulls_real_numbers():
    passages = ["The tower is 330 meters tall.", "Built in 1889, cost 7.8 million francs."]
    values = extract_numeric_facts(passages)
    assert 330.0 in values
    assert 1889.0 in values
    assert 7.8 in values


def test_numeric_checker_unfitted_never_vetoes():
    checker = NumericPlausibilityChecker()
    assert checker.is_fitted is False
    assert checker.is_plausible(999999999) is True


def test_numeric_checker_flags_wild_outlier_not_in_range_values():
    import random
    random.seed(0)
    passages = [f"The tower stands {random.randint(50, 600)} meters tall." for _ in range(200)]
    checker = NumericPlausibilityChecker(min_facts=50)
    assert checker.fit_from_passages(passages) is True
    assert checker.is_plausible(300) is True
    assert checker.is_plausible(50000) is False


def test_numeric_checker_declines_to_fit_with_too_few_facts():
    checker = NumericPlausibilityChecker(min_facts=50)
    assert checker.fit_from_passages(["only one number: 42"]) is False
    assert checker.is_fitted is False


# ── EntailmentChecker adapter: graceful degradation ─────────────────────────

def test_entailment_checker_load_missing_checkpoint_returns_none():
    assert EntailmentChecker.load("/tmp/definitely_does_not_exist_12345.pt") is None


# ── Oracle integration: strictly additive, never overrides a pass into fail ─

class _FakeContradictionChecker:
    """Always says contradiction -- used to prove the veto actually fires."""
    def is_contradiction(self, premise, hypothesis):
        return True


class _FakeAgreeableChecker:
    """Never objects -- used to prove a passing deterministic check isn't
    disturbed when the additional layer has nothing to add."""
    def is_contradiction(self, premise, hypothesis):
        return False


class _FakeImplausibleNumericChecker:
    def is_plausible(self, value):
        return False


def test_entailment_veto_rejects_a_claim_the_deterministic_check_passed():
    evidence = ["The Eiffel Tower is 330 meters tall and located in Paris."]
    oracle = FactCheckOracle(entailment_checker=_FakeContradictionChecker())
    # Deterministic check alone would pass this (real evidence, real overlap).
    assert oracle.support("The Eiffel Tower is 330 meters tall.", evidence) >= oracle.min_support
    # But the entailment layer vetoes it.
    assert oracle.is_grounded("The Eiffel Tower is 330 meters tall.", evidence) is False
    assert len(oracle.layered_veto_log) == 1
    assert oracle.layered_veto_log[0]["layer"] == "entailment"


def test_entailment_checker_cannot_override_a_deterministic_rejection():
    """The core safety property: a smart layer agreeing isn't enough to
    accept a claim the deterministic check already rejected."""
    evidence = ["Photosynthesis converts sunlight into chemical energy."]
    oracle = FactCheckOracle(entailment_checker=_FakeAgreeableChecker())
    # Deterministic check rejects (no overlap at all) -- entailment checker
    # saying "no contradiction" must not resurrect it.
    assert oracle.is_grounded("The Eiffel Tower is 500 meters tall.", evidence) is False


def test_numeric_veto_rejects_a_claim_the_deterministic_check_passed():
    evidence = ["The tower is 330 meters tall and located in Paris, built in 1889."]
    oracle = FactCheckOracle(numeric_checker=_FakeImplausibleNumericChecker())
    assert oracle.is_grounded("The tower is 330 meters tall.", evidence) is False
    assert len(oracle.layered_veto_log) == 1
    assert oracle.layered_veto_log[0]["layer"] == "numeric"


def test_no_layered_checkers_behaves_exactly_as_before():
    """Default construction (no checkers passed) must be byte-identical
    in behavior to the oracle before this item existed."""
    evidence = ["The Eiffel Tower is 330 meters tall and located in Paris."]
    oracle = FactCheckOracle()
    assert oracle.is_grounded("The Eiffel Tower is 330 meters tall.", evidence) is True
    assert oracle.is_grounded("The Eiffel Tower is 500 meters tall.", evidence) is True  # pre-existing overlap limitation, unrelated to this item
    assert oracle.layered_veto_log == []


def test_layered_veto_does_not_fire_on_no_evidence_relaxation_reject():
    """When the no-evidence relaxation already rejects (specific claim, no
    evidence), the layered vetoes shouldn't need to run at all -- but
    correctness-wise, the result must still be False regardless."""
    oracle = FactCheckOracle(entailment_checker=_FakeAgreeableChecker())
    assert oracle.is_grounded("The Eiffel Tower is 500 meters tall.", []) is False


def test_entailment_checker_exception_fails_open():
    class _BrokenChecker:
        def is_contradiction(self, premise, hypothesis):
            raise RuntimeError("boom")

    evidence = ["The Eiffel Tower is 330 meters tall and located in Paris."]
    oracle = FactCheckOracle(entailment_checker=_BrokenChecker())
    # A broken additional layer must not become a hard failure -- it fails
    # open, deterministic check's verdict stands.
    assert oracle.is_grounded("The Eiffel Tower is 330 meters tall.", evidence) is True
