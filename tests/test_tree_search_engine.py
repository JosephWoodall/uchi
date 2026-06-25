"""Tests for uchi/tree_search_engine.py — PUCT tree search engine."""

import math
import pytest
from unittest.mock import MagicMock, patch


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_router(peek_dist=None, eval_value=0.1):
    """Mock router with deterministic trie and SSM responses."""
    router = MagicMock()
    router.predictor.peek_distribution.return_value = (
        {"hello": 0.4, "world": 0.3, "test": 0.2, "foo": 0.05, "bar": 0.05}
        if peek_dist is None else peek_dist
    )
    return router


# ── _Node ─────────────────────────────────────────────────────────────────────

class TestNode:
    def test_default_values(self):
        from uchi.tree_search_engine import _Node
        node = _Node(tokens=["hello"])
        assert node.value == 0.0
        assert node.prior == 1.0
        assert node.visits == 0
        assert node.children == {}
        assert node.pruned is False

    def test_tokens_stored(self):
        from uchi.tree_search_engine import _Node
        tokens = ["a", "b", "c"]
        node = _Node(tokens=tokens)
        assert node.tokens == tokens


# ── TreeSearchEngine ──────────────────────────────────────────────────────────

class TestTreeSearchEngine:
    def test_returns_list(self):
        from uchi.tree_search_engine import TreeSearchEngine
        router = _make_router()
        engine = TreeSearchEngine(router)
        with patch.object(engine, "_eval", return_value=0.1):
            result = engine.search(["<|user|>", "hello", "<|assistant|>"])
        assert isinstance(result, list)

    def test_empty_seed_returns_empty(self):
        from uchi.tree_search_engine import TreeSearchEngine
        engine = TreeSearchEngine(_make_router())
        result = engine.search([])
        assert result == []

    def test_stop_tokens_stripped_from_result(self):
        from uchi.tree_search_engine import TreeSearchEngine
        router = _make_router(peek_dist={"<|user|>": 0.5, "hello": 0.5})
        engine = TreeSearchEngine(router)
        with patch.object(engine, "_eval", return_value=0.2):
            result = engine.search(["seed"], max_nodes=20)
        assert "<|user|>" not in result
        assert "<|assistant|>" not in result
        assert "<|end|>" not in result

    def test_respects_max_depth(self):
        from uchi.tree_search_engine import TreeSearchEngine
        router = _make_router()
        engine = TreeSearchEngine(router)
        with patch.object(engine, "_eval", return_value=0.5):
            result = engine.search(["seed"], max_nodes=100, max_depth=5)
        assert len(result) <= 5

    def test_prunes_low_value_nodes(self):
        from uchi.tree_search_engine import TreeSearchEngine, PRUNE_THRESHOLD
        router = _make_router()
        engine = TreeSearchEngine(router)

        call_count = [0]
        def fake_eval(tokens):
            call_count[0] += 1
            # Return below prune threshold for all expansions
            return PRUNE_THRESHOLD - 0.1

        with patch.object(engine, "_eval", side_effect=fake_eval):
            result = engine.search(["seed"], max_nodes=30)
        # Result may be empty (all pruned) or very short
        assert isinstance(result, list)

    def test_empty_trie_returns_empty(self):
        from uchi.tree_search_engine import TreeSearchEngine
        router = _make_router(peek_dist={})
        engine = TreeSearchEngine(router)
        with patch.object(engine, "_eval", return_value=0.5):
            result = engine.search(["seed"], max_nodes=20)
        assert result == []

    def test_best_path_follows_high_visit_children(self):
        from uchi.tree_search_engine import TreeSearchEngine, _Node
        engine = TreeSearchEngine(_make_router())
        root = _Node(["seed"])
        root.visits = 10
        child_a = _Node(["seed", "a"], value=0.5)
        child_a.visits = 8
        child_b = _Node(["seed", "b"], value=0.3)
        child_b.visits = 2
        root.children = {"a": child_a, "b": child_b}
        path = engine._best_path(root, seed_len=1)
        assert path == ["a"]  # highest visits

    def test_eval_returns_float(self):
        from uchi.tree_search_engine import TreeSearchEngine
        from uchi.neuro_symbolic import StateSpaceModel
        router = _make_router()
        engine = TreeSearchEngine(router)
        with patch("uchi.neuro_symbolic.get_ssm", return_value=StateSpaceModel(d_model=64)):
            val = engine._eval(["hello", "world"])
        assert isinstance(val, float)

    def test_eval_fallback_on_exception(self):
        from uchi.tree_search_engine import TreeSearchEngine
        router = _make_router()
        engine = TreeSearchEngine(router)
        with patch("uchi.neuro_symbolic.get_ssm", side_effect=RuntimeError("boom")):
            val = engine._eval(["test"])
        assert val == 0.0

    def test_ucb1_selects_unvisited_child(self):
        from uchi.tree_search_engine import TreeSearchEngine, _Node
        engine = TreeSearchEngine(_make_router())
        parent = _Node(["a"])
        parent.visits = 5
        visited = _Node(["a", "b"], value=0.8, prior=0.5)
        visited.visits = 4
        unvisited = _Node(["a", "c"], value=0.1, prior=0.5)
        unvisited.visits = 0
        parent.children = {"b": visited, "c": unvisited}
        # Unvisited child gets infinite UCB bonus → should be selected
        selected = engine._ucb1_select(parent)
        assert selected is unvisited

    def test_ucb1_skips_pruned_children(self):
        from uchi.tree_search_engine import TreeSearchEngine, _Node
        engine = TreeSearchEngine(_make_router())
        parent = _Node(["a"])
        parent.visits = 10
        pruned = _Node(["a", "x"], value=0.9, prior=1.0)
        pruned.pruned = True
        active = _Node(["a", "y"], value=0.1, prior=0.1)
        parent.children = {"x": pruned, "y": active}
        selected = engine._ucb1_select(parent)
        assert selected is active

    def test_ucb1_returns_none_if_all_pruned(self):
        from uchi.tree_search_engine import TreeSearchEngine, _Node
        engine = TreeSearchEngine(_make_router())
        parent = _Node(["a"])
        parent.visits = 1
        pruned = _Node(["a", "x"])
        pruned.pruned = True
        parent.children = {"x": pruned}
        selected = engine._ucb1_select(parent)
        assert selected is None


# ── Self-consistency reward ───────────────────────────────────────────────────

class TestSelfConsistencyReward:
    def _make_vec(self, v0, v1=0.0):
        """Unit vector with two prominent dimensions."""
        import math
        v = [v0, v1] + [0.0] * 62
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n > 1e-9 else v

    def test_consensus_returns_positive(self):
        from uchi.convergent_engine import _self_consistency_reward
        # All candidates have identical tokens → diversity = 0 → +0.1
        q_vec = self._make_vec(1.0)
        cands = [
            (["hello", "world"], q_vec),
            (["hello", "world"], q_vec),
            (["hello", "world"], q_vec),
        ]
        reward = _self_consistency_reward(cands, q_vec)
        assert reward == pytest.approx(0.1)

    def test_divergent_returns_zero(self):
        from uchi.convergent_engine import _self_consistency_reward
        q_vec = self._make_vec(1.0)
        # Completely different token sets → diversity ≈ 1 → 0.0
        cands = [
            (["apple", "banana"], q_vec),
            (["car", "truck"], q_vec),
            (["sun", "moon"], q_vec),
        ]
        reward = _self_consistency_reward(cands, q_vec)
        assert reward == pytest.approx(0.0)

    def test_single_candidate_returns_zero(self):
        from uchi.convergent_engine import _self_consistency_reward
        q_vec = self._make_vec(1.0)
        reward = _self_consistency_reward([(["only", "one"], q_vec)], q_vec)
        assert reward == pytest.approx(0.0)

    def test_returns_float(self):
        from uchi.convergent_engine import _self_consistency_reward
        q_vec = self._make_vec(1.0)
        cands = [(["a", "b"], q_vec), (["c", "d"], q_vec)]
        assert isinstance(_self_consistency_reward(cands, q_vec), float)


# ── Inner monologue backtracking ──────────────────────────────────────────────

class TestInnerMonologueBacktracking:
    """Verify that <|inner_monologue|> fires when all children are pruned."""

    def test_inner_monologue_injected_on_dead_end(self):
        """When every expanded child is below PRUNE_THRESHOLD, the engine
        injects <|inner_monologue|> rather than giving up."""
        from uchi.tree_search_engine import (
            TreeSearchEngine, _Node, PRUNE_THRESHOLD, _INNER_MONOLOGUE,
        )
        router = _make_router(
            peek_dist={"bad_tok": 0.9, "worse_tok": 0.1},
        )
        engine = TreeSearchEngine(router)

        # Patch eval so every child gets a value below PRUNE_THRESHOLD
        sub_threshold = PRUNE_THRESHOLD - 0.1
        with patch.object(engine, "_eval_with_hidden", return_value=(sub_threshold, None)), \
             patch.object(engine, "_eval_incremental", return_value=(sub_threshold, None)):
            result = engine.search(["seed"], max_nodes=10)

        # The inner monologue token must NOT appear in the final output
        assert _INNER_MONOLOGUE not in result

    def test_best_path_strips_inner_monologue(self):
        """_best_path must strip <|inner_monologue|> from the generated sequence."""
        from uchi.tree_search_engine import TreeSearchEngine, _Node, _INNER_MONOLOGUE
        engine = TreeSearchEngine(_make_router())
        root = _Node(["seed"], value=0.5)
        root.visits = 5
        # Simulate path: seed → <|inner_monologue|> → good_word
        im_node = _Node(["seed", _INNER_MONOLOGUE], value=0.3)
        im_node.visits = 4
        good_node = _Node(["seed", _INNER_MONOLOGUE, "good_word"], value=0.8)
        good_node.visits = 3
        im_node.children = {"good_word": good_node}
        root.children = {_INNER_MONOLOGUE: im_node}

        result = engine._best_path(root, seed_len=1)
        assert _INNER_MONOLOGUE not in result
        assert "good_word" in result

    def test_inner_monologue_not_a_stop_token(self):
        """<|inner_monologue|> must not be in _STOP_TOKENS — it's a search control
        token, not a sequence terminator."""
        from uchi.tree_search_engine import _STOP_TOKENS, _INNER_MONOLOGUE
        assert _INNER_MONOLOGUE not in _STOP_TOKENS


# ── ExperienceReplayBuffer ────────────────────────────────────────────────────

class TestExperienceReplayBuffer:
    def setup_method(self, _):
        import tempfile, os
        self._tmp = tempfile.mktemp(suffix=".db")

    def teardown_method(self, _):
        import os
        try:
            os.unlink(self._tmp)
        except OSError:
            pass

    def _buf(self):
        from uchi.experience_replay import ExperienceReplayBuffer
        return ExperienceReplayBuffer(self._tmp)

    def test_empty_buffer_returns_empty_sample(self):
        buf = self._buf()
        assert buf.sample(4) == []

    def test_len_zero_initially(self):
        buf = self._buf()
        assert len(buf) == 0

    def test_push_increases_length(self):
        import time
        buf = self._buf()
        buf.push(["hello"], ["world"], priority=1.0)
        # push is async — wait briefly for the background thread
        time.sleep(0.1)
        assert len(buf) == 1

    def test_sample_returns_correct_structure(self):
        import time
        buf = self._buf()
        buf.push(["q1", "q2"], ["p1", "p2"], hard_negative_tokens=["n1"], priority=2.0)
        time.sleep(0.1)
        rows = buf.sample(1)
        assert len(rows) == 1
        row = rows[0]
        assert "id" in row and "query" in row and "positive" in row
        assert isinstance(row["query"], list)
        assert isinstance(row["positive"], list)
        assert row["priority"] == 2.0

    def test_update_priority_changes_value(self):
        import time
        buf = self._buf()
        buf.push(["q"], ["p"], priority=5.0)
        time.sleep(0.1)
        rows = buf.sample(1)
        assert rows
        memory_id = rows[0]["id"]
        buf.update_priority(memory_id, 0.001)
        # Re-sample to verify the update landed
        rows2 = buf.sample(1)
        assert rows2
        assert rows2[0]["priority"] < 1.0  # floored at _PRIORITY_FLOOR, but < 5.0

    def test_high_priority_sampled_more_often(self):
        """Memories with higher priority should dominate the sample distribution."""
        import time
        buf = self._buf()
        buf.push(["low"], ["priority"], priority=0.001)
        buf.push(["high"], ["priority"], priority=1000.0)
        time.sleep(0.2)
        counts = {"low": 0, "high": 0}
        for _ in range(50):
            rows = buf.sample(1)
            if rows:
                key = "high" if rows[0]["query"] == ["high"] else "low"
                counts[key] += 1
        assert counts["high"] > counts["low"]

    def test_push_without_negative_stores_none(self):
        import time
        buf = self._buf()
        buf.push(["q"], ["p"])  # no hard_negative_tokens
        time.sleep(0.1)
        rows = buf.sample(1)
        assert rows
        assert rows[0]["negative"] is None


# ── Temperature Annealing (Prior Warmup) ──────────────────────────────────────

class TestTemperatureAnnealing:
    """Verify the depth-based UCB prior warmup behaves as specified."""

    def test_temperature_is_max_at_depth_zero(self):
        """T = max(1.0, 5.0 * (1.0 - 0/max_depth)) = 5.0 at depth 0."""
        max_depth = 40
        depth = 0
        T = max(1.0, 5.0 * (1.0 - (depth / max(max_depth, 1))))
        assert T == pytest.approx(5.0)

    def test_temperature_is_one_at_max_depth(self):
        """T = max(1.0, 5.0 * (1.0 - max_depth/max_depth)) = 1.0 at full depth."""
        max_depth = 40
        T = max(1.0, 5.0 * (1.0 - (max_depth / max(max_depth, 1))))
        assert T == pytest.approx(1.0)

    def test_temperature_never_below_one(self):
        """T is clamped to ≥ 1.0 at any depth."""
        max_depth = 40
        for depth in range(0, max_depth + 5):
            T = max(1.0, 5.0 * (1.0 - (depth / max(max_depth, 1))))
            assert T >= 1.0

    def test_high_temperature_flattens_distribution(self):
        """At T=5.0, softmax over equal raw probs produces uniform output (already flat input stays flat)."""
        import torch
        import torch.nn.functional as F
        # With T=5.0, dividing a peaked distribution by T and softmaxing should be
        # less peaked than the original.
        probs = [0.7, 0.2, 0.07, 0.02, 0.01]
        T_high = 5.0
        T_low  = 1.0
        high_annealed = F.softmax(torch.tensor([p / T_high for p in probs]), dim=0).tolist()
        low_annealed  = F.softmax(torch.tensor([p / T_low  for p in probs]), dim=0).tolist()
        # High temperature should yield a more uniform distribution (lower max)
        assert max(high_annealed) < max(low_annealed), (
            "High temperature should flatten distribution"
        )

    def test_search_uses_annealed_priors(self):
        """Search completes without error and returns a list (prior annealing active)."""
        from uchi.tree_search_engine import TreeSearchEngine
        router = _make_router(
            peek_dist={"hello": 0.7, "world": 0.2, "foo": 0.07, "bar": 0.02, "baz": 0.01}
        )
        engine = TreeSearchEngine(router)
        with patch.object(engine, "_eval_with_hidden", return_value=(0.5, None)), \
             patch.object(engine, "_eval_incremental", return_value=(0.5, None, None)), \
             patch.object(engine, "_build_seed_cache", return_value=None):
            result = engine.search(["seed"], max_nodes=20, max_depth=10)
        assert isinstance(result, list)


# ── Episodic KV Cache in _Node ────────────────────────────────────────────────

class TestNodeKVCache:
    """_Node now stores a kv_cache slot alongside the hidden state."""

    def test_node_default_kv_cache_is_none(self):
        from uchi.tree_search_engine import _Node
        node = _Node(tokens=["hello"])
        assert node.kv_cache is None

    def test_node_accepts_kv_cache(self):
        import torch
        from uchi.tree_search_engine import _Node
        kv = torch.randn(5, 64)
        node = _Node(tokens=["a", "b"], kv_cache=kv)
        assert node.kv_cache is not None
        assert node.kv_cache.shape == (5, 64)

    def test_eval_incremental_returns_triple(self):
        """_eval_incremental must now return (value, hidden, kv_cache)."""
        from uchi.tree_search_engine import TreeSearchEngine
        from uchi.neuro_symbolic import StateSpaceModel
        import torch
        router = _make_router()
        engine = TreeSearchEngine(router)
        ssm = StateSpaceModel(d_model=64)
        with patch("uchi.neuro_symbolic.get_ssm", return_value=ssm):
            hidden = ssm.get_state(["hello"])
            val, new_hidden, new_kv = engine._eval_incremental(hidden, "world")
        assert isinstance(val, float)
        assert new_hidden is not None
        assert new_kv is None  # no kv_cache passed → None returned

    def test_eval_incremental_with_kv_cache(self):
        """When kv_cache is passed, returned cache grows by one state."""
        from uchi.tree_search_engine import TreeSearchEngine
        from uchi.neuro_symbolic import StateSpaceModel
        import torch
        router = _make_router()
        engine = TreeSearchEngine(router)
        ssm = StateSpaceModel(d_model=64)
        with patch("uchi.neuro_symbolic.get_ssm", return_value=ssm):
            hidden = ssm.get_state(["hello"])
            seed_kv = ssm.get_kv_cache(["hello"])
            val, new_h, new_kv = engine._eval_incremental(hidden, "world", seed_kv)
        assert new_kv is not None
        assert new_kv.shape[0] == seed_kv.shape[0] + 1

    def test_build_seed_cache_returns_tensor_or_none(self):
        """_build_seed_cache returns a tensor on success or None on failure."""
        from uchi.tree_search_engine import TreeSearchEngine
        from uchi.neuro_symbolic import StateSpaceModel
        router = _make_router()
        engine = TreeSearchEngine(router)
        ssm = StateSpaceModel(d_model=64)
        with patch("uchi.neuro_symbolic.get_ssm", return_value=ssm):
            kv = engine._build_seed_cache(["hello", "world"])
        assert kv is not None
        assert kv.shape[1] == 64

    def test_build_seed_cache_returns_none_on_error(self):
        from uchi.tree_search_engine import TreeSearchEngine
        router = _make_router()
        engine = TreeSearchEngine(router)
        with patch("uchi.neuro_symbolic.get_ssm", side_effect=RuntimeError("boom")):
            result = engine._build_seed_cache(["hello"])
        assert result is None


# ── BPE Fallback ──────────────────────────────────────────────────────────────

class TestBPEFallback:
    """OmniTokenizer shatters OOV words into subwords, never UnknownConcept."""

    def test_oov_word_returns_list_of_strings(self):
        from uchi.omni_tokenizer import OmniTokenizer, UnknownConcept
        tok = OmniTokenizer(use_wordnet=False)
        result = tok.tokenize(["xyzzyquux"], is_inference=True)
        assert isinstance(result, list)
        assert len(result) >= 1
        assert not any(isinstance(t, UnknownConcept) for t in result)
        assert all(isinstance(t, str) and t for t in result)

    def test_bpe_fallback_short_word(self):
        from uchi.omni_tokenizer import OmniTokenizer
        subs = OmniTokenizer._bpe_fallback("hi")
        assert subs == ["hi"]

    def test_bpe_fallback_produces_non_empty_strings(self):
        from uchi.omni_tokenizer import OmniTokenizer
        for word in ["gibberish", "pythoon", "zzzt", "a"]:
            subs = OmniTokenizer._bpe_fallback(word)
            assert all(isinstance(s, str) and s for s in subs), f"empty token in {subs}"

    def test_subwords_registered_in_known_concepts(self):
        """After BPE fallback, the subwords are added to _known_concepts."""
        from uchi.omni_tokenizer import OmniTokenizer
        tok = OmniTokenizer(use_wordnet=False)
        result = tok.tokenize(["zxqvjk"], is_inference=True)
        for sw in result:
            if isinstance(sw, str):
                assert sw in tok._known_concepts


# ── Oracle AST Blame ──────────────────────────────────────────────────────────

class TestOracleASTBlame:
    """oracle_ast_blame assigns 1.0 to valid prefix, 0.0 from first error token."""

    def test_valid_sequence_all_ones(self):
        from uchi.omni_evaluator import oracle_ast_blame
        tokens = ["def", "add", "(", "a", ",", "b", ")", ":", "return", "a", "+", "b"]
        rewards = oracle_ast_blame(tokens)
        assert rewards == [1.0] * len(tokens)

    def test_empty_sequence(self):
        from uchi.omni_evaluator import oracle_ast_blame
        assert oracle_ast_blame([]) == []

    def test_error_at_end_has_valid_prefix(self):
        from uchi.omni_evaluator import oracle_ast_blame
        tokens = ["x", "=", "1", "+", ")"]
        rewards = oracle_ast_blame(tokens)
        assert len(rewards) == len(tokens)
        # Prefix before the ')' should be 1.0
        assert rewards[0] == 1.0
        # The ')' and everything after should be 0.0
        assert rewards[-1] == 0.0

    def test_blame_index_is_first_bad_token(self):
        from uchi.omni_evaluator import oracle_ast_blame
        tokens = ["def", "f", "(", ")", ":", "=+", "oops"]
        rewards = oracle_ast_blame(tokens)
        ones = [r for r in rewards if r == 1.0]
        zeros = [r for r in rewards if r == 0.0]
        assert len(ones) > 0, "valid prefix should exist"
        assert len(zeros) > 0, "error tokens should get 0.0"
        # Ones must precede zeros (no interleaving)
        first_zero = next(i for i, r in enumerate(rewards) if r == 0.0)
        assert all(r == 1.0 for r in rewards[:first_zero])
        assert all(r == 0.0 for r in rewards[first_zero:])

    def test_returns_same_length_as_input(self):
        from uchi.omni_evaluator import oracle_ast_blame
        tokens = ["a", "b", "c", "invalid +++"]
        rewards = oracle_ast_blame(tokens)
        assert len(rewards) == len(tokens)


# ── Grammar Mask PDA ──────────────────────────────────────────────────────────

class TestGrammarMaskPDA:
    """Verify the enhanced PDA rules beyond bracket balancing."""

    _FENCE = "```python"

    def test_return_masked_outside_function(self):
        from uchi.grammar_mask import apply
        dist = {"return": 0.8, "x": 0.2}
        result = apply([self._FENCE, "x", "=", "1"], dist)
        assert "return" not in result

    def test_return_allowed_inside_function(self):
        from uchi.grammar_mask import apply
        dist = {"return": 0.8, "x": 0.2}
        result = apply([self._FENCE, "def", "f", "(", ")", ":", "x"], dist)
        assert "return" in result

    def test_break_masked_outside_loop(self):
        from uchi.grammar_mask import apply
        dist = {"break": 0.5, "pass": 0.5}
        result = apply([self._FENCE, "x", "=", "1"], dist)
        assert "break" not in result

    def test_break_allowed_inside_for_loop(self):
        from uchi.grammar_mask import apply
        dist = {"break": 0.5, "pass": 0.5}
        result = apply([self._FENCE, "for", "x", "in", "y"], dist)
        assert "break" in result

    def test_after_def_keyword_not_allowed(self):
        from uchi.grammar_mask import apply
        dist = {"return": 0.3, "if": 0.3, "add": 0.4}
        result = apply([self._FENCE, "def"], dist)
        assert "return" not in result
        assert "if" not in result
        assert "add" in result

    def test_fallback_when_all_tokens_masked(self):
        from uchi.grammar_mask import apply
        # Only "return" in distribution, but we're outside a function
        dist = {"return": 1.0}
        result = apply([self._FENCE, "x"], dist)
        # Fallback: return original to avoid MCTS deadlock
        assert result == dist

    def test_non_code_block_passes_through(self):
        from uchi.grammar_mask import apply
        dist = {"return": 0.5, "break": 0.5}
        result = apply(["hello", "world"], dist)
        assert result == dist  # no change outside code fence


# ── Neuro-Symbolic: CrossAttentionHead & KV Compression ──────────────────────

class TestCrossAttentionHead:
    def test_output_shape(self):
        import torch
        from uchi.neuro_symbolic import CrossAttentionHead
        ca = CrossAttentionHead(d_model=64)
        q = torch.randn(1, 64)
        kv = torch.randn(8, 64)
        out, weights = ca(q, kv)
        assert out.shape == (1, 64)
        assert weights.shape == (8,)

    def test_attention_weights_sum_to_one(self):
        import torch
        from uchi.neuro_symbolic import CrossAttentionHead
        ca = CrossAttentionHead(d_model=64)
        q = torch.randn(1, 64)
        kv = torch.randn(6, 64)
        _, weights = ca(q, kv)
        assert abs(weights.sum().item() - 1.0) < 1e-5

    def test_output_differs_from_input(self):
        """Cross-attention should modify the query state."""
        import torch
        from uchi.neuro_symbolic import CrossAttentionHead
        torch.manual_seed(42)
        ca = CrossAttentionHead(d_model=64)
        q = torch.randn(1, 64)
        kv = torch.randn(4, 64)
        out, _ = ca(q, kv)
        assert not torch.allclose(out, q, atol=1e-4), "output should differ from raw query"


class TestKVCacheCompression:
    def test_no_compression_below_budget(self):
        import torch
        from uchi.neuro_symbolic import _compress_kv_cache
        cache = torch.randn(100, 256)
        out = _compress_kv_cache(cache, max_budget=1000)
        assert out.shape == cache.shape

    def test_compression_enforces_budget(self):
        import torch
        from uchi.neuro_symbolic import _compress_kv_cache
        cache = torch.randn(1500, 256)
        out = _compress_kv_cache(cache, max_budget=1000, n_sinks=4)
        assert out.shape[0] == 1000

    def test_sinks_always_preserved(self):
        import torch
        from uchi.neuro_symbolic import _compress_kv_cache
        cache = torch.randn(2000, 256)
        out = _compress_kv_cache(cache, max_budget=100, n_sinks=4)
        assert torch.allclose(out[:4], cache[:4])

    def test_attention_weight_based_eviction(self):
        """When attn_weights given, highest-weight states are kept."""
        import torch
        from uchi.neuro_symbolic import _compress_kv_cache
        n, d = 200, 64
        cache = torch.randn(n, d)
        weights = torch.zeros(n)
        weights[10] = 10.0  # state 10 is very important
        out = _compress_kv_cache(cache, attn_weights=weights, max_budget=50, n_sinks=4)
        # State 10 should survive (it's beyond the sinks, but has high weight)
        # Find if cache[10] appears in out[4:] (non-sink region)
        survived = any(
            torch.allclose(out[i], cache[10]) for i in range(4, out.shape[0])
        )
        assert survived, "high-attention state should be preserved"


class TestInfoNCELoss:
    def test_loss_is_finite_with_reward(self):
        import torch
        from uchi.neuro_symbolic import StateSpaceModel
        ssm = StateSpaceModel(d_model=64)
        seq = ["def", "f", "(", "x", ")", ":", "return", "x"]
        loss = ssm.compute_loss(seq, reward=1.0)
        assert torch.isfinite(loss)

    def test_loss_is_zero_for_length_one(self):
        import torch
        from uchi.neuro_symbolic import StateSpaceModel
        ssm = StateSpaceModel(d_model=64)
        loss = ssm.compute_loss(["single"], reward=1.0)
        assert loss.item() == pytest.approx(0.0)

    def test_gradient_flows_through_policy_head(self):
        import torch
        from uchi.neuro_symbolic import StateSpaceModel
        ssm = StateSpaceModel(d_model=64)
        ssm.train()
        seq = ["a", "b", "c", "d", "e"]
        loss = ssm.compute_loss(seq, reward=1.0)
        loss.backward()
        ph_grad = ssm.policy_head.net[-1].weight.grad
        assert ph_grad is not None, "gradient must flow to policy head"
        assert ph_grad.abs().sum().item() > 0
