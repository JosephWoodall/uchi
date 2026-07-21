"""Tests for uchi/flux/react_warmup_train.py.

Real-throwaway-repo test (same pattern as tests/test_grpo.py's `buggy_repo`
fixture: real git repo + real ExecutionSandbox, real gold patch) verifying
`build_react_example` produces real, PER-TURN (prompt, target) pairs -- the
v2 fix over v1's whole-trace-in-one-shot approach (see module docstring for
why that was wrong). Every Action/Observation comes from a real tool call
against a real repo; only the Thought/Final Answer English is templated.
No live network/GPU needed.
"""
import subprocess

import pytest

from uchi.agentic_repair import (
    _ACTION_PATTERN,
    _FINAL_PATTERN,
    _PATCH_ACTION_PATTERN,
    _THOUGHT_PATTERN,
)
from uchi.execution_sandbox import ExecutionSandbox
from uchi.flux.react_warmup_train import build_react_example, load_react_examples


def _run(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, f"{cmd} failed: {result.stderr}"
    return result


@pytest.fixture
def fixed_repo(tmp_path):
    """A real throwaway repo with a real bug and a real, correct fix --
    same shape as test_grpo.py's buggy_repo, but only needs the correct
    patch (this module doesn't compare fix vs. wrong branches)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init"], cwd=repo)
    _run(["git", "config", "user.email", "test@example.com"], cwd=repo)
    _run(["git", "config", "user.name", "Test"], cwd=repo)

    (repo / "mathutils.py").write_text("def add(a, b):\n    return a - b\n")
    (repo / "test_bug.py").write_text(
        "from mathutils import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    (repo / "test_sanity.py").write_text("def test_sanity():\n    assert 1 == 1\n")
    _run(["git", "add", "-A"], cwd=repo)
    _run(["git", "commit", "-m", "buggy base"], cwd=repo)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()

    (repo / "mathutils.py").write_text("def add(a, b):\n    return a + b\n")
    fix_patch = subprocess.run(["git", "diff"], cwd=repo, capture_output=True, text=True).stdout
    _run(["git", "checkout", "--", "."], cwd=repo)

    return {"repo_path": str(repo), "base_commit": base_commit, "fix_patch": fix_patch}


def _record(fixed_repo):
    return {
        "repo": fixed_repo["repo_path"],  # local path stands in for "owner/name" in offline tests
        "instance_id": "test-instance",
        "problem_statement": "add() is subtracting instead of adding.",
        "patch": fixed_repo["fix_patch"],
        "base_commit": fixed_repo["base_commit"],
        "fail_to_pass": ["test_bug.py::test_add"],
        "pass_to_pass": ["test_sanity.py::test_sanity"],
    }


def test_build_react_example_returns_multiple_real_turns(fixed_repo, monkeypatch):
    monkeypatch.setattr(
        "uchi.repo_fetch.ensure_local_clone",
        lambda repo, commit: fixed_repo["repo_path"],
    )
    sandbox = ExecutionSandbox(timeout=30.0)
    record = _record(fixed_repo)

    turns = build_react_example(record, sandbox)

    assert turns is not None
    # read_file, apply_patch, run_tests, final -- 4 discrete turns, not 1
    # flat trace (the whole point of the v2 fix).
    assert len(turns) == 4
    for turn in turns:
        assert "prompt" in turn and "target" in turn

    # Turn 1: bare prompt (no prior history yet).
    assert "add() is subtracting" in turns[0]["prompt"]
    assert turns[1]["prompt"] != turns[0]["prompt"]
    # Turn 2's prompt must include turn 1's real history (the read_file
    # action + its real Observation) -- proof context accumulates exactly
    # like run_react_episode's own memory list.
    assert "Action: read_file[mathutils.py]" in turns[1]["prompt"]
    assert "return a - b" in turns[1]["prompt"]  # real pre-patch file content


def test_each_turn_target_parses_as_exactly_one_react_step(fixed_repo, monkeypatch):
    # The real, objective check: each turn's target must be parseable by
    # the ACTUAL production regexes agentic_repair.run_react_episode uses
    # (not a hand-rolled approximation) -- this is what "the format is
    # trainable" actually means.
    monkeypatch.setattr(
        "uchi.repo_fetch.ensure_local_clone",
        lambda repo, commit: fixed_repo["repo_path"],
    )
    sandbox = ExecutionSandbox(timeout=30.0)
    record = _record(fixed_repo)

    turns = build_react_example(record, sandbox)
    assert turns is not None

    # Turn 1 (read_file): Thought + bracket-form Action.
    assert _THOUGHT_PATTERN.search(turns[0]["target"])
    assert _ACTION_PATTERN.search(turns[0]["target"])

    # Turn 2 (apply_patch): Thought + the special <<<PATCH>>> block form,
    # containing the REAL gold diff verbatim -- not a placeholder.
    assert _THOUGHT_PATTERN.search(turns[1]["target"])
    patch_match = _PATCH_ACTION_PATTERN.search(turns[1]["target"])
    assert patch_match
    assert "return a + b" in patch_match.group(1)
    # The full diff must NOT appear again in turn 3's prompt -- only the
    # abbreviated `<N-char diff>` memory form does, matching
    # run_react_episode's own abbreviation exactly.
    assert "return a + b" not in turns[2]["prompt"]
    assert "-char diff>" in turns[2]["prompt"]

    # Turn 3 (run_tests): Thought + bracket-form Action.
    assert _THOUGHT_PATTERN.search(turns[2]["target"])
    assert _ACTION_PATTERN.search(turns[2]["target"])

    # Turn 4 (final): Thought + Final Answer, no Action.
    assert _THOUGHT_PATTERN.search(turns[3]["target"])
    assert _FINAL_PATTERN.search(turns[3]["target"])
    assert not _ACTION_PATTERN.search(turns[3]["target"])


def test_build_react_example_returns_none_without_a_touched_file(fixed_repo, monkeypatch):
    monkeypatch.setattr(
        "uchi.repo_fetch.ensure_local_clone",
        lambda repo, commit: fixed_repo["repo_path"],
    )
    sandbox = ExecutionSandbox(timeout=30.0)
    record = _record(fixed_repo)
    record["patch"] = ""  # no diff --git header -> extract_patch_files finds nothing

    assert build_react_example(record, sandbox) is None


class _StubTokenizer:
    vocab_size = 128
    syntax_vocab_size = 8
    eos_token_id = 1
    pad_token_id = 2

    def encode_text(self, text: str, max_length: int | None = None) -> list[int]:
        ids = [ord(c) % self.vocab_size for c in text]
        return ids[:max_length] if max_length else ids

    def encode_special(self, name: str) -> int:
        return {"<|user|>": 3, "<|think|>": 4}[name]


def test_load_react_examples_yields_more_examples_than_instances(fixed_repo, monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(
        "uchi.repo_fetch.ensure_local_clone",
        lambda repo, commit: fixed_repo["repo_path"],
    )
    record = _record(fixed_repo)
    curriculum_path = tmp_path / "curriculum.jsonl"
    curriculum_path.write_text(json.dumps(record))

    # Generous seq_len: char-level stub tokenizer has no BPE compression,
    # so real prompt+target text needs far more "tokens" under it than a
    # real tokenizer would use -- this test is about the per-turn
    # multiplication and masking, not the length-drop path.
    tokenizer = _StubTokenizer()
    formatted = load_react_examples(tokenizer, max_seq_len=4096, max_examples=1,
                                     curriculum_path=str(curriculum_path))

    # 1 real instance -> 4 real per-turn examples -- the whole point of the
    # v2 fix (more real training signal from the same real curated data).
    assert len(formatted) == 4
    for ex in formatted:
        assert len(ex["input_ids"]) == 4096
        assert len(ex["loss_mask"]) == 4096
        assert ex["loss_mask"][0] == 0
        assert 1 in ex["loss_mask"]
