"""Tests for v0.3.0 deliverables 6–11."""

import tempfile
import os


# ── Item 6: Confidence Calibration ───────────────────────────────────────────

class TestTemperatureCalibrator:
    def test_identity_temperature(self):
        from uchi.calibration import TemperatureCalibrator
        cal = TemperatureCalibrator(temperature=1.0)
        assert abs(cal.predict(0.0) - 0.5) < 0.01

    def test_calibrate_separates_classes(self):
        from uchi.calibration import TemperatureCalibrator
        cal = TemperatureCalibrator()
        raw    = [3.0, 3.5, -3.0, -3.5] * 15
        labels = [1.0, 1.0,  0.0,  0.0] * 15
        cal.calibrate(raw, labels)
        assert cal.predict(3.0) > 0.5
        assert cal.predict(-3.0) < 0.5

    def test_predict_returns_probability(self):
        from uchi.calibration import TemperatureCalibrator
        cal = TemperatureCalibrator()
        p = cal.predict(1.0)
        assert 0.0 < p < 1.0

    def test_save_load_roundtrip(self):
        from uchi.calibration import TemperatureCalibrator
        cal = TemperatureCalibrator(temperature=2.5)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cal.save(path)
            cal2 = TemperatureCalibrator.load(path)
            assert abs(cal2.temperature.item() - 2.5) < 1e-4
        finally:
            os.unlink(path)

    def test_load_missing_file_returns_default(self):
        from uchi.calibration import TemperatureCalibrator
        cal = TemperatureCalibrator.load("/nonexistent/path.json")
        assert abs(cal.temperature.item() - 1.0) < 1e-4

    def test_too_few_samples_skips_calibration(self):
        from uchi.calibration import TemperatureCalibrator
        cal = TemperatureCalibrator(temperature=1.5)
        T = cal.calibrate([1.0, 0.0], [1.0, 0.0])  # only 2 samples
        assert abs(T - 1.5) < 1e-4  # unchanged


# ── Item 7: HRR Analogical Fallback ──────────────────────────────────────────

class TestHRRFallback:
    def test_hrr_fallback_method_exists(self):
        from uchi.convergent_engine import ConvergentEngine
        assert hasattr(ConvergentEngine, "_hrr_analogy_fallback")

    def test_hrr_fallback_returns_none_without_memory(self):
        """When the router has no memory, fallback should return None gracefully."""
        from uchi.convergent_engine import ConvergentEngine
        from unittest.mock import MagicMock

        router = MagicMock()
        router.memory = None
        engine = ConvergentEngine.__new__(ConvergentEngine)
        engine._router = router

        result = engine._hrr_analogy_fallback(["hello", "world"], [])
        assert result is None


# ── Item 8: GoalState ─────────────────────────────────────────────────────────

class TestGoalState:
    def test_update_increments_turn_count_on_related_queries(self):
        from uchi.goal_state import GoalState
        gs = GoalState()
        gs.update(["python", "sorting", "list", "algorithm"], "code")
        gs.update(["sort", "list", "python", "function"], "code")
        assert gs.turn_count > 1

    def test_update_resets_on_unrelated_query(self):
        from uchi.goal_state import GoalState
        gs = GoalState()
        gs.update(["python", "sorting", "list"], "code")
        # Completely unrelated query
        gs.update(["weather", "tomorrow", "rain", "forecast"], "factual")
        assert gs.turn_count == 1

    def test_short_term_intent_updated(self):
        from uchi.goal_state import GoalState
        gs = GoalState()
        gs.update(["what", "is", "two", "plus", "two"], "math")
        assert gs.short_term_intent == "math"

    def test_objective_tokens_returns_list(self):
        from uchi.goal_state import GoalState
        gs = GoalState()
        for _ in range(4):
            gs.update(["machine", "learning", "model", "training"], "code")
        toks = gs.objective_tokens()
        assert isinstance(toks, list)

    def test_is_new_thread_first_turn(self):
        from uchi.goal_state import GoalState
        gs = GoalState()
        gs.update(["hello", "world"], None)
        assert gs.is_new_thread()

    def test_summary_returns_string(self):
        from uchi.goal_state import GoalState
        gs = GoalState()
        gs.update(["python", "code"], "code")
        assert isinstance(gs.summary(), str)


# ── Item 9: Inner Monologue ───────────────────────────────────────────────────

class TestInnerMonologue:
    def test_inner_monologue_method_exists(self):
        from uchi.convergent_engine import ConvergentEngine
        assert hasattr(ConvergentEngine, "_run_inner_monologue")

    def test_skips_factual_queries(self):
        from uchi.convergent_engine import ConvergentEngine
        from unittest.mock import MagicMock

        engine = ConvergentEngine.__new__(ConvergentEngine)
        engine._router = MagicMock()
        # factual_short queries should return [] immediately
        result = engine._run_inner_monologue(
            ["paris", "france"], initial_value=0.9,
            query_type="factual_short", bias_context=None
        )
        assert result == []

    def test_skips_when_bias_context_set(self):
        from uchi.convergent_engine import ConvergentEngine
        from unittest.mock import MagicMock

        engine = ConvergentEngine.__new__(ConvergentEngine)
        engine._router = MagicMock()
        result = engine._run_inner_monologue(
            ["explain", "how", "sorting", "works", "in", "python"],
            initial_value=0.3, query_type="generative",
            bias_context="already have bias"
        )
        assert result == []


# ── Item 10: Experience Replay ────────────────────────────────────────────────

class TestExperienceReplay:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        os.unlink(self.tmp.name)  # let ExperienceReplayBuffer create fresh
        self.db_path = self.tmp.name

    def teardown_method(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_push_and_sample(self):
        from uchi.experience_replay import ExperienceReplayBuffer
        buf = ExperienceReplayBuffer(db_path=self.db_path)
        import time
        for i in range(10):
            buf.push(["q", str(i)], ["r", str(i)], priority=1.0)
        time.sleep(0.1)  # let async writes complete
        batch = buf.sample(batch_size=4)
        assert len(batch) <= 4

    def test_update_priority(self):
        from uchi.experience_replay import ExperienceReplayBuffer
        import time
        buf = ExperienceReplayBuffer(db_path=self.db_path)
        buf.push(["q"], ["r"], priority=1.0)
        time.sleep(0.1)
        batch = buf.sample(batch_size=1)
        if batch:
            mem_id = batch[0]["id"]
            buf.update_priority(mem_id, 2.0)  # should not raise

    def test_replay_buffer_wired_to_router(self):
        """OmniRouter should have replay_buffer attribute after __init__."""
        from uchi.experience_replay import ExperienceReplayBuffer
        from uchi.omni_router import OmniRouter
        OmniRouter._bootstrap_knowledge = lambda self, *a, **kw: None
        OmniRouter._bootstrap_persona   = lambda self, *a, **kw: None
        router = OmniRouter(use_bpe=False)
        assert hasattr(router, "replay_buffer")
        assert isinstance(router.replay_buffer, ExperienceReplayBuffer)


# ── Item 11: Format Reward ────────────────────────────────────────────────────

class TestFormatReward:
    def test_good_response_positive_score(self):
        from uchi.grpo import format_reward
        score = format_reward(
            ["the", "capital", "of", "france", "is", "paris"],
            ["what", "is", "the", "capital", "of", "france"],
        )
        assert score > 0

    def test_empty_response_max_negative(self):
        from uchi.grpo import format_reward
        assert format_reward([], ["what"]) == -1.0

    def test_very_long_response_penalised(self):
        from uchi.grpo import format_reward
        long_resp = ["word"] * 100
        short_resp = ["word"] * 10
        assert format_reward(long_resp, []) < format_reward(short_resp, [])

    def test_echo_response_penalised(self):
        from uchi.grpo import format_reward
        query = ["what", "is", "the", "meaning", "of", "life"]
        echo  = query[:]  # identical tokens
        good  = ["forty", "two", "is", "the", "answer"]
        assert format_reward(echo, query) < format_reward(good, query)

    def test_repeated_trigrams_penalised(self):
        from uchi.grpo import format_reward
        degenerate = ["the", "cat", "sat"] * 10
        normal     = ["the", "cat", "sat", "on", "a", "mat", "near", "a", "bat"]
        assert format_reward(degenerate, []) < format_reward(normal, [])

    def test_code_intent_rewards_code_keywords(self):
        from uchi.grpo import format_reward
        code_resp    = ["def", "sort", "return", "sorted", "list"]
        generic_resp = ["yes", "sure", "here", "you", "go"]
        assert (
            format_reward(code_resp, [], intent_key="code")
            > format_reward(generic_resp, [], intent_key="code")
        )

    def test_output_bounded(self):
        from uchi.grpo import format_reward
        for tokens in [[], ["a"], ["x"] * 5, ["the", "quick", "brown"] * 20]:
            score = format_reward(tokens, ["hello"])
            assert -1.0 <= score <= 1.0

    def test_synset_tokens_penalised(self):
        """Naturalness penalty: synset-heavy responses score lower."""
        from uchi.grpo import format_reward
        synset_resp  = ["water.n.01", "is.v.01", "boiling.n.01", "at.r.01", "100.n.01"]
        clean_resp   = ["water", "is", "boiling", "at", "100", "degrees"]
        assert format_reward(synset_resp, []) < format_reward(clean_resp, [])

    def test_control_tokens_penalised(self):
        from uchi.grpo import format_reward
        control_resp = ["<|assistant|>", "paris", "is", "the", "capital", "<|user|>"]
        clean_resp   = ["paris", "is", "the", "capital", "of", "france"]
        assert format_reward(control_resp, []) < format_reward(clean_resp, [])


# ── Item 14: Response Normalizer ─────────────────────────────────────────────

class TestResponseNormalizer:
    def test_strips_synset_markers(self):
        from uchi.response_normalizer import normalize
        result = normalize("the water.n.01 boil.v.01 at 100 degree.n.01")
        assert ".n.01" not in result
        assert ".v.01" not in result
        assert "water" in result

    def test_strips_control_tokens(self):
        from uchi.response_normalizer import normalize
        result = normalize("<|assistant|> paris is the capital <|user|>")
        assert "<|" not in result
        assert "paris" in result.lower()

    def test_strips_uncertain_marker(self):
        from uchi.response_normalizer import normalize
        result = normalize("[Uncertain] the answer may be photosynthesis.n.01")
        assert "[Uncertain]" not in result
        assert "photosynthesis" in result

    def test_capitalises_first_letter(self):
        from uchi.response_normalizer import normalize
        result = normalize("the answer is 42")
        assert result[0].isupper()

    def test_expands_underscore_compounds(self):
        from uchi.response_normalizer import normalize
        result = normalize("united_states is a country")
        assert "_" not in result
        assert "united states" in result.lower()

    def test_terminal_punctuation_added(self):
        from uchi.response_normalizer import normalize
        result = normalize("paris is the capital of france")
        assert result[-1] in ".!?"

    def test_empty_string_passthrough(self):
        from uchi.response_normalizer import normalize
        assert normalize("") == ""
        assert normalize("  ") == "  "

    def test_already_clean_unchanged(self):
        from uchi.response_normalizer import normalize
        clean = "The capital of France is Paris."
        result = normalize(clean)
        assert result == clean

    def test_combined_artifacts(self):
        from uchi.response_normalizer import normalize
        raw = "<|assistant|> the capital.n.01 of france.n.01 is paris.n.01 <|user|>"
        result = normalize(raw)
        assert "<|" not in result
        assert ".n.01" not in result
        assert result[0].isupper()
        assert "paris" in result.lower()
