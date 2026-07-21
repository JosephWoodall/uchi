"""Tests for benchmarks/truthfulqa_benchmark.py (0.5.0 Item 9).

Same no-network-download shape as tests/test_humaneval_benchmark.py:
direct function tests, no dataset fetch. `score_question` is exercised
against a stub policy (same pattern as tests/test_grpo.py's `_StubPolicy`/
`_StubTokenizer`) -- an untrained stub can't be expected to prefer the
"correct" choice (that's Item 9's real, honest result to measure, not
something to fake here), so this only asserts the mechanics: a valid index
is returned and scoring is deterministic given a frozen model.
"""
import torch

from benchmarks.truthfulqa_benchmark import score_question


class _StubTokenizer:
    vocab_size = 64

    def encode_text(self, text: str, max_length: int = 1024) -> list[int]:
        return [ord(c) % self.vocab_size for c in text][:max_length] or [0]


class _StubPolicy(torch.nn.Module):
    def __init__(self, vocab_size: int = 64, d_model: int = 8):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.proj = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        return self.proj(h), None


def test_score_question_returns_valid_index():
    model = _StubPolicy()
    model.eval()
    tokenizer = _StubTokenizer()
    choices = ["Paris.", "London.", "Berlin.", "Madrid."]

    idx = score_question(model, tokenizer, "cpu", "What is the capital of France?", choices)
    assert 0 <= idx < len(choices)


def test_score_question_is_deterministic_for_a_frozen_model():
    model = _StubPolicy()
    model.eval()
    tokenizer = _StubTokenizer()
    choices = ["Yes.", "No.", "Maybe."]

    idx1 = score_question(model, tokenizer, "cpu", "Is the sky blue?", choices)
    idx2 = score_question(model, tokenizer, "cpu", "Is the sky blue?", choices)
    assert idx1 == idx2


def test_score_question_single_choice_is_trivially_correct():
    model = _StubPolicy()
    model.eval()
    tokenizer = _StubTokenizer()
    assert score_question(model, tokenizer, "cpu", "Q?", ["only choice"]) == 0
