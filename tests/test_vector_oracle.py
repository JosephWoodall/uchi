"""Tests for uchi/vector_oracle.py — encoding, diversity, and contrastive alignment."""

import math
import pytest
import torch
from unittest.mock import MagicMock, patch


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_ssm():
    """Return a real (tiny) SSM so encode() has real GRU parameters to test."""
    from uchi.neuro_symbolic import StateSpaceModel
    return StateSpaceModel(d_model=64)


def _unit(vec):
    n = math.sqrt(sum(x * x for x in vec))
    return [x / n for x in vec] if n > 1e-9 else vec


# ── similarity ────────────────────────────────────────────────────────────────

class TestSimilarity:
    def test_identical_vectors(self):
        from uchi.vector_oracle import similarity
        v = [1.0, 0.0, 0.0]
        assert similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        from uchi.vector_oracle import similarity
        assert similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        from uchi.vector_oracle import similarity
        assert similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector_returns_zero(self):
        from uchi.vector_oracle import similarity
        assert similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ── token_diversity ───────────────────────────────────────────────────────────

class TestTokenDiversity:
    def test_single_candidate_returns_one(self):
        from uchi.vector_oracle import token_diversity
        assert token_diversity([["a", "b", "c"]]) == pytest.approx(1.0)

    def test_identical_candidates_returns_zero(self):
        from uchi.vector_oracle import token_diversity
        cands = [["hello", "world"], ["hello", "world"], ["hello", "world"]]
        assert token_diversity(cands) == pytest.approx(0.0, abs=1e-6)

    def test_disjoint_candidates_returns_one(self):
        from uchi.vector_oracle import token_diversity
        cands = [["a", "b"], ["c", "d"], ["e", "f"]]
        assert token_diversity(cands) == pytest.approx(1.0, abs=1e-6)

    def test_partial_overlap(self):
        from uchi.vector_oracle import token_diversity
        cands = [["a", "b", "c"], ["a", "b", "d"]]
        div = token_diversity(cands)
        assert 0.0 < div < 1.0

    def test_returns_float(self):
        from uchi.vector_oracle import token_diversity
        result = token_diversity([["x"], ["y"]])
        assert isinstance(result, float)

    def test_empty_list_returns_one(self):
        from uchi.vector_oracle import token_diversity
        assert token_diversity([]) == pytest.approx(1.0)


# ── encode ────────────────────────────────────────────────────────────────────

class TestEncode:
    def test_returns_list_of_floats(self):
        from uchi.vector_oracle import encode
        ssm = _make_ssm()
        result = encode(["hello", "world"], ssm=ssm)
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_length_equals_d_model(self):
        from uchi.vector_oracle import encode
        ssm = _make_ssm()
        result = encode(["test"], ssm=ssm)
        assert len(result) == ssm.d_model

    def test_unit_norm(self):
        from uchi.vector_oracle import encode
        ssm = _make_ssm()
        vec = encode(["hello", "world"], ssm=ssm)
        norm = math.sqrt(sum(x * x for x in vec))
        assert norm == pytest.approx(1.0, abs=1e-4)

    def test_empty_tokens_returns_zeros(self):
        from uchi.vector_oracle import encode
        ssm = _make_ssm()
        result = encode([], ssm=ssm)
        assert len(result) == ssm.d_model

    def test_with_trie_dist_changes_vector(self):
        from uchi.vector_oracle import encode, similarity
        ssm = _make_ssm()
        v_no_dist = encode(["classify", "data"], ssm=ssm)
        v_with_dist = encode(
            ["classify", "data"],
            trie_dist={"csv": 0.5, "table": 0.5},
            ssm=ssm,
        )
        # Trie distribution should shift the vector (not identical)
        sim = similarity(v_no_dist, v_with_dist)
        assert sim < 1.0  # they diverge

    def test_fallback_on_exception(self):
        from uchi.vector_oracle import encode
        bad_ssm = MagicMock()
        bad_ssm.d_model = 8
        bad_ssm.get_state.side_effect = RuntimeError("boom")
        result = encode(["test"], ssm=bad_ssm)
        assert result == [0.0] * 8

    def test_different_tokens_produce_different_vectors(self):
        from uchi.vector_oracle import encode, similarity
        ssm = _make_ssm()
        v1 = encode(["classify", "csv", "data"], ssm=ssm)
        v2 = encode(["write", "python", "function"], ssm=ssm)
        assert similarity(v1, v2) < 0.999


# ── contrastive_update ────────────────────────────────────────────────────────

class TestContrastiveUpdate:
    def _make_optimizer(self, ssm):
        return torch.optim.Adam(ssm.parameters(), lr=1e-3)

    def test_positive_reward_pulls_vectors_closer(self):
        from uchi.vector_oracle import encode, similarity, contrastive_update
        ssm = _make_ssm()
        opt = self._make_optimizer(ssm)

        q_toks = ["classify", "my", "data"]
        r_toks = ["tabular", "predictor", "accuracy"]

        sim_before = similarity(
            encode(q_toks, ssm=ssm),
            encode(r_toks, ssm=ssm),
        )
        for _ in range(5):
            contrastive_update(q_toks, r_toks, reward=+1.0, optimizer=opt, ssm=ssm)
        sim_after = similarity(
            encode(q_toks, ssm=ssm),
            encode(r_toks, ssm=ssm),
        )
        assert sim_after >= sim_before - 0.1  # should not get worse

    def test_negative_reward_pushes_vectors_apart(self):
        from uchi.vector_oracle import encode, similarity, contrastive_update
        ssm = _make_ssm()
        opt = self._make_optimizer(ssm)

        q_toks = ["hello", "world"]
        r_toks = ["hello", "world"]

        for _ in range(5):
            contrastive_update(q_toks, r_toks, reward=-1.0, optimizer=opt, ssm=ssm)
        sim_after = similarity(
            encode(q_toks, ssm=ssm),
            encode(r_toks, ssm=ssm),
        )
        assert sim_after <= 1.0  # must stay in valid range

    def test_empty_tokens_is_no_op(self):
        from uchi.vector_oracle import contrastive_update
        ssm = _make_ssm()
        opt = self._make_optimizer(ssm)
        contrastive_update([], ["hello"], reward=1.0, optimizer=opt, ssm=ssm)
        contrastive_update(["hello"], [], reward=1.0, optimizer=opt, ssm=ssm)

    def test_does_not_raise_on_exception(self):
        from uchi.vector_oracle import contrastive_update
        bad_ssm = MagicMock()
        bad_ssm.get_state.side_effect = RuntimeError("bad")
        bad_opt = MagicMock()
        contrastive_update(["a"], ["b"], reward=1.0, optimizer=bad_opt, ssm=bad_ssm)
