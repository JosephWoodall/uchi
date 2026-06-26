"""Tests for the routing layer components added in v0.3.0."""
import pytest
from uchi.procedural_memory import ProceduralMemory
from uchi.grpo import AgenticBaseline, grpo_agentic_advantage
from uchi.cpu_memory import CPUVectorMemory
import numpy as np
import tempfile
import os


class TestProceduralMemory:
    def setup_method(self):
        # Use a temp path so tests don't pollute the working directory
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        self.tmp_path = tmp.name
        # Remove the empty file so ProceduralMemory creates it fresh with defaults
        os.unlink(self.tmp_path)
        self.mem = ProceduralMemory(path=self.tmp_path)

    def teardown_method(self):
        if os.path.exists(self.tmp_path):
            os.unlink(self.tmp_path)

    def test_code_intent_detected(self):
        result = self.mem.retrieve("write a python function to sort a list")
        assert result is not None
        assert "code" in result.lower()

    def test_math_intent_detected(self):
        result = self.mem.retrieve("calculate the sum of two integers")
        assert result is not None
        assert "math" in result.lower()

    def test_unknown_intent_returns_none(self):
        result = self.mem.retrieve("hello how are you doing today")
        assert result is None

    def test_update_persists(self):
        self.mem.update("cooking", "identify the recipe")
        result = self.mem.retrieve("cooking pasta")
        assert result is not None


class TestAgenticBaseline:
    def test_initial_advantage_is_finite(self):
        baseline = AgenticBaseline()
        adv = baseline.advantage(1.0)
        assert isinstance(adv, float)
        assert adv == pytest.approx(1.0, abs=1.0)

    def test_update_shifts_mean(self):
        baseline = AgenticBaseline(alpha=0.5)
        baseline.update(10.0)
        assert baseline.mean > 0

    def test_advantage_normalizes(self):
        baseline = AgenticBaseline()
        for _ in range(20):
            baseline.update(1.0)
        # After many updates of 1.0, advantage of 1.0 should be near 0
        adv = baseline.advantage(1.0)
        assert abs(adv) < 2.0

    def test_grpo_agentic_advantage(self):
        adv = grpo_agentic_advantage(reward=1.0, running_mean=0.0, running_std=1.0)
        assert adv == pytest.approx(1.0)


class TestCPUVectorMemory:
    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_mem")
        self.mem = CPUVectorMemory(db_path=self.db_path)

    def test_add_and_retrieve(self):
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.mem.add_memory("test text", vec)
        results = self.mem.retrieve(vec, top_k=1)
        assert len(results) == 1
        assert results[0] == "test text"

    def test_retrieve_with_scores(self):
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.mem.add_memory("exact match", vec)
        results = self.mem.retrieve_with_scores(vec, top_k=1)
        assert len(results) == 1
        text, score = results[0]
        assert text == "exact match"
        assert score == pytest.approx(1.0, abs=0.01)

    def test_cosine_similarity_ordering(self):
        self.mem.add_memory("close", np.array([1.0, 0.1, 0.0], dtype=np.float32))
        self.mem.add_memory("far", np.array([0.0, 0.0, 1.0], dtype=np.float32))
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = self.mem.retrieve_with_scores(query, top_k=2)
        assert results[0][0] == "close"

    def test_persistence(self):
        vec = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        self.mem.add_memory("persisted", vec)
        # Re-open from same path
        mem2 = CPUVectorMemory(db_path=self.db_path)
        assert "persisted" in mem2.records
