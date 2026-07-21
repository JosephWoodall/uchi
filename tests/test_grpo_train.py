"""Tests for uchi/grpo_train.py (0.5.0 Item 6 launcher).

Covers only the non-training helper logic -- curriculum iteration with a
cap, and trajectory JSONL serialization -- same scope discipline as
tests/test_grpo.py (real training itself needs a real live checkpoint,
exercised via a live dry-run, not a unit test with a stub policy).
"""
import json

from uchi.grpo import Branch
from uchi.grpo_train import _branch_trajectory_lines, _iter_curriculum


def _diff(n_changed_lines: int) -> str:
    body = "\n".join(f"+line{i}" for i in range(n_changed_lines))
    return f"diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,{n_changed_lines} @@\n{body}\n"


def _write_curriculum(tmp_path, n_small=2, n_large=2):
    records = []
    for i in range(n_small):
        records.append({
            "repo": "a/a", "instance_id": f"small-{i}", "problem_statement": "p",
            "patch": _diff(3), "base_commit": "x", "fail_to_pass": [], "pass_to_pass": [],
        })
    for i in range(n_large):
        records.append({
            "repo": "b/b", "instance_id": f"large-{i}", "problem_statement": "p",
            "patch": _diff(100), "base_commit": "x", "fail_to_pass": [], "pass_to_pass": [],
        })
    path = tmp_path / "curriculum.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return str(path)


class _StubTokenizer:
    vocab_size = 64

    def encode_text(self, text: str, max_length: int = 1024) -> list[int]:
        return [ord(c) % self.vocab_size for c in text][:max_length] or [0]


def test_iter_curriculum_respects_max_instances_and_bucket_order(tmp_path):
    path = _write_curriculum(tmp_path, n_small=2, n_large=2)

    # No cap: all 4, small bucket (easy) before large (hard).
    all_ids = [r["instance_id"] for r in _iter_curriculum(path, max_instances=None)]
    assert all_ids == ["small-0", "small-1", "large-0", "large-1"]

    # Capped: stops mid-run, not just mid-bucket.
    capped_ids = [r["instance_id"] for r in _iter_curriculum(path, max_instances=3)]
    assert capped_ids == ["small-0", "small-1", "large-0"]


def test_iter_curriculum_zero_cap_yields_nothing(tmp_path):
    path = _write_curriculum(tmp_path, n_small=1, n_large=0)
    assert list(_iter_curriculum(path, max_instances=0)) == []


def _make_branch(transcript: list[str], reward: float, resolved: bool) -> Branch:
    class _FakeEvalResult:
        def __init__(self, resolved):
            self._resolved = resolved

        @property
        def resolved(self):
            return self._resolved

    import torch
    return Branch(
        transcript=transcript, patch="", eval_result=_FakeEvalResult(resolved),
        reward=reward, log_prob=torch.tensor(0.0),
    )


def test_branch_trajectory_lines_round_trip():
    tokenizer = _StubTokenizer()
    branches = [
        _make_branch(["Thought: fix it", "Action: apply_patch"], reward=1.0, resolved=True),
        _make_branch(["Thought: guess", "Action: apply_patch"], reward=0.0, resolved=False),
    ]

    lines = _branch_trajectory_lines(branches, tokenizer, "the bug report", "inst-1")
    assert len(lines) == 2

    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["instance_id"] == "inst-1"
    assert parsed[0]["reward"] == 1.0
    assert parsed[0]["resolved"] is True
    assert parsed[1]["reward"] == 0.0
    assert parsed[1]["resolved"] is False
    # Real, non-empty token sequences -- what Item 8's train_dynamics needs.
    assert all(isinstance(t, int) for t in parsed[0]["tokens"])
    assert len(parsed[0]["tokens"]) > 0


def test_branch_trajectory_lines_handles_missing_eval_result():
    tokenizer = _StubTokenizer()
    import torch
    branch = Branch(transcript=["Thought: no patch produced"], patch="",
                     eval_result=None, reward=0.0, log_prob=torch.tensor(0.0))
    lines = _branch_trajectory_lines([branch], tokenizer, "p", "inst-2")
    parsed = json.loads(lines[0])
    assert parsed["resolved"] is False
