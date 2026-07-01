"""Tests for the routing layer components added in v0.3.0."""
import pytest
from uchi.procedural_memory import ProceduralMemory
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


class TestIncrementalBrainBuilder:
    """Unit tests for the incremental brain builder (item 5)."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.brain_path = os.path.join(self.tmp_dir, "test_brain.uchi")
        self.checkpoint_path = os.path.join(self.tmp_dir, "checkpoint.json")

    def test_checkpoint_roundtrip(self):
        from uchi.incremental_builder import IncrementalBrainBuilder
        builder = IncrementalBrainBuilder(
            brain_path=self.brain_path,
            checkpoint_path=self.checkpoint_path,
        )
        builder._save_checkpoint(current_source="mmlu", offset=42, total=100)
        assert os.path.exists(self.checkpoint_path)

        builder2 = IncrementalBrainBuilder(
            brain_path=self.brain_path,
            checkpoint_path=self.checkpoint_path,
        )
        assert builder2._checkpoint["current_source"] == "mmlu"
        assert builder2._checkpoint["current_offset"] == 42
        assert builder2._checkpoint["total_ingested"] == 100

    def test_mark_source_complete(self):
        from uchi.incremental_builder import IncrementalBrainBuilder
        builder = IncrementalBrainBuilder(
            brain_path=self.brain_path,
            checkpoint_path=self.checkpoint_path,
        )
        builder._mark_source_complete("openhermes")
        assert "openhermes" in builder._checkpoint["completed_sources"]
        assert os.path.exists(self.checkpoint_path)

    def test_clear_checkpoint(self):
        from uchi.incremental_builder import IncrementalBrainBuilder
        builder = IncrementalBrainBuilder(
            brain_path=self.brain_path,
            checkpoint_path=self.checkpoint_path,
        )
        builder._save_checkpoint(current_source="mmlu", offset=1, total=1)
        assert os.path.exists(self.checkpoint_path)
        builder._clear_checkpoint()
        assert not os.path.exists(self.checkpoint_path)

    def test_completed_sources_skipped(self):
        from uchi.incremental_builder import IncrementalBrainBuilder
        builder = IncrementalBrainBuilder(
            brain_path=self.brain_path,
            checkpoint_path=self.checkpoint_path,
        )
        # Mark a source complete before the run starts
        builder._checkpoint["completed_sources"] = ["openhermes", "wikipedia",
                                                     "mmlu", "gsm8k",
                                                     "swebench", "humaneval"]
        builder._save_checkpoint(total=0)
        # run() with all sources pre-completed should not raise
        # and should exit quickly without ingesting anything
        builder2 = IncrementalBrainBuilder(
            brain_path=self.brain_path,
            checkpoint_path=self.checkpoint_path,
        )
        # Patch _train_ssm_delta to no-op to avoid torch cost in tests
        builder2._train_ssm_delta = lambda router, seqs: None
        builder2.run(limit=0, sources=["openhermes"])
