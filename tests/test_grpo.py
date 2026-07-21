"""Tests for uchi/grpo.py (0.5.0 Item 6).

Pure-math tests against hand-computed values for the ported GRPO
functions, plus a real-throwaway-repo test (same pattern as
tests/test_agentic_repair.py: real git repo + real ExecutionSandbox,
scripted fake `generate_fn`) verifying GRPOTrainer runs a full multi-branch
step, produces a non-degenerate advantage vector, and actually
backpropagates a gradient into the (stub) policy's parameters. No trained
FLUX checkpoint is used or required -- that's the whole point of this
scope, see grpo.py's module docstring.
"""
import json
import subprocess

import pytest
import torch

from uchi.execution_sandbox import ExecutionSandbox
from uchi.grpo import (
    AgenticBaseline,
    GRPOTrainer,
    TrajectoryStep,
    _patch_line_count,
    derive_test_lists,
    grpo_advantage,
    grpo_agentic_advantage,
    grpo_loss,
    load_curriculum,
    parse_transcript,
    sequence_log_prob,
)


# ── Pure math ──────────────────────────────────────────────────────────────


def test_grpo_advantage_hand_computed():
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0])
    adv = grpo_advantage(rewards)
    # mean=0.5, std=0.5773502692 (population std with ddof=1 via torch default)
    expected_std = rewards.std().item()
    assert adv[0].item() == pytest.approx((0.0 - 0.5) / expected_std, abs=1e-4)
    assert adv[1].item() == pytest.approx((1.0 - 0.5) / expected_std, abs=1e-4)


def test_grpo_advantage_uniform_rewards_near_zero():
    # All-equal rewards -> zero-variance advantage, the exact degenerate
    # case the module docstring warns about (needs curriculum to avoid).
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
    adv = grpo_advantage(rewards)
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-3)


def test_grpo_loss_rewards_above_average_branches():
    advantage = torch.tensor([1.0, -1.0])
    log_probs = torch.tensor([-0.5, -0.5], requires_grad=True)
    loss = grpo_loss(advantage, log_probs)
    # L = -mean(adv * logp) = -mean([-0.5, 0.5]) = -0.0
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_grpo_agentic_advantage():
    assert grpo_agentic_advantage(reward=1.0, running_mean=0.0, running_std=1.0) == pytest.approx(1.0)


def test_agentic_baseline_updates_toward_reward():
    baseline = AgenticBaseline(alpha=0.5)
    baseline.update(1.0)
    assert baseline.mean > 0.0
    adv = baseline.advantage(1.0)
    assert adv > 0.0  # reward above the (still-low) running mean


# ── Trajectory schema ──────────────────────────────────────────────────────


def test_parse_transcript_tags_each_line():
    lines = [
        "Thought: I should look at the file first.",
        "Action: read_file[mathutils.py]",
        "Observation: def add(a, b): return a - b",
        "Final Answer: fixed the subtraction bug.",
    ]
    steps = parse_transcript(lines)
    assert [s.kind for s in steps] == ["thought", "action", "observation", "final"]
    assert steps[0].text == "I should look at the file first."
    assert all(isinstance(s, TrajectoryStep) for s in steps)


# ── Curriculum bucketing ───────────────────────────────────────────────────


def _diff(n_changed_lines: int) -> str:
    body = "\n".join(f"+line{i}" for i in range(n_changed_lines))
    return f"diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,{n_changed_lines} @@\n{body}\n"


def test_patch_line_count_ignores_hunk_headers():
    diff = _diff(5)
    assert _patch_line_count(diff) == 5


def test_load_curriculum_buckets_easy_to_hard(tmp_path):
    records = [
        {"repo": "a/a", "instance_id": "small", "problem_statement": "p", "patch": _diff(3),
         "base_commit": "x", "fail_to_pass": [], "pass_to_pass": []},
        {"repo": "b/b", "instance_id": "medium", "problem_statement": "p", "patch": _diff(30),
         "base_commit": "x", "fail_to_pass": [], "pass_to_pass": []},
        {"repo": "c/c", "instance_id": "large", "problem_statement": "p", "patch": _diff(100),
         "base_commit": "x", "fail_to_pass": [], "pass_to_pass": []},
    ]
    jsonl_path = tmp_path / "corpus.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(r) for r in records))

    small, medium, large = load_curriculum(str(jsonl_path))
    assert [r["instance_id"] for r in small] == ["small"]
    assert [r["instance_id"] for r in medium] == ["medium"]
    assert [r["instance_id"] for r in large] == ["large"]


# ── sequence_log_prob + GRPOTrainer, against a stub policy ────────────────


class _StubTokenizer:
    """Deterministic, dependency-free stand-in for TikTokenHybridTokenizer
    -- GRPOTrainer's sequence_log_prob only needs `encode_text(text,
    max_length=...) -> list[int]` within a small vocab."""

    vocab_size = 64

    def encode_text(self, text: str, max_length: int = 1024) -> list[int]:
        return [ord(c) % self.vocab_size for c in text][:max_length] or [0]


class _StubPolicy(torch.nn.Module):
    """Minimal trainable stand-in for HybridTSSM -- only needs to be
    callable as `model(x) -> (lang_logits, syntax_logits)` with real
    gradients, same call shape `sequence_log_prob` uses against the real
    model. `.embedding` aliases `.embed` -- `sequence_log_prob`'s
    empty-response path reads `model.embedding.weight` directly (matching
    the real HybridTSSM's attribute name), so the stub needs to expose it
    under both names to stand in for that path too."""

    def __init__(self, vocab_size: int = 64, d_model: int = 8):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.embedding = self.embed
        self.proj = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        return self.proj(h), None


def test_sequence_log_prob_is_finite_and_differentiable():
    model = _StubPolicy()
    tok = _StubTokenizer()
    lp = sequence_log_prob(model, tok, prompt="Issue: fix the bug", response="Thought: ok")
    assert torch.isfinite(lp)
    lp.backward()
    assert model.embed.weight.grad is not None


def test_sequence_log_prob_empty_response_is_still_differentiable():
    # Real bug found running a live GRPO calibration: a branch whose
    # response leaves no room in the token budget (e.g. a long prompt with
    # a tight max_length, or -- the real, observed case -- a ReAct episode
    # that produced nothing against an undertrained proposer) hits the
    # "no room for a response" early-return path. The old code returned
    # torch.zeros(()) there -- a fresh tensor disconnected from the model's
    # graph. If EVERY branch in a GRPOTrainer group hits this
    # simultaneously, grpo_loss's stacked log_probs ends up with no graph
    # at all, and .backward() crashes with "element 0 of tensors does not
    # require grad and does not have a grad_fn". max_length=1 forces the
    # prompt alone to consume the whole budget, deterministically
    # triggering the same n_response<=0 path.
    model = _StubPolicy()
    tok = _StubTokenizer()
    lp = sequence_log_prob(model, tok, prompt="Issue: fix the bug", response="Thought: ok", max_length=1)
    assert torch.isfinite(lp)
    assert lp.item() == pytest.approx(0.0)
    lp.backward()
    assert model.embed.weight.grad is not None


def test_grpo_loss_backward_survives_an_all_degenerate_group():
    # The exact failure mode: every branch in a group hits the degenerate
    # case above, so log_probs is built entirely from sequence_log_prob's
    # differentiable-zero path, not real per-token log-probs. grpo_loss's
    # backward() must not crash even here.
    model = _StubPolicy()
    tok = _StubTokenizer()
    log_probs = torch.stack([
        sequence_log_prob(model, tok, prompt="Issue: fix the bug", response="Thought: ok", max_length=1)
        for _ in range(4)
    ])
    rewards = torch.tensor([0.0, 0.0, 0.0, 0.0])
    advantage = grpo_advantage(rewards)
    loss = grpo_loss(advantage, log_probs)
    assert torch.isfinite(loss)
    loss.backward()  # must not raise


def _run(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, f"{cmd} failed: {result.stderr}"
    return result


@pytest.fixture
def buggy_repo(tmp_path):
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

    (repo / "mathutils.py").write_text("def add(a, b):\n    return a * b\n")  # still wrong
    wrong_patch = subprocess.run(["git", "diff"], cwd=repo, capture_output=True, text=True).stdout
    _run(["git", "checkout", "--", "."], cwd=repo)

    return {
        "repo_path": str(repo), "base_commit": base_commit,
        "fix_patch": fix_patch, "wrong_patch": wrong_patch,
    }


def _patch_action(diff: str) -> str:
    return f"Thought: I'll apply the fix.\nAction: apply_patch\n<<<PATCH>>>\n{diff}\n<<<END_PATCH>>>"


FAIL_TO_PASS = ["test_bug.py::test_add"]
PASS_TO_PASS = ["test_sanity.py::test_sanity"]


def test_grpo_trainer_step_produces_nondegenerate_advantage_and_updates_policy(buggy_repo):
    # Alternates fix/wrong across branches so the group's rewards actually
    # differ -- the real-world condition GRPO's normalization needs to
    # produce a non-zero gradient at all.
    call_count = {"n": 0}

    def fake_generate(prompt, max_tokens=300, think=True):
        call_count["n"] += 1
        diff = buggy_repo["fix_patch"] if call_count["n"] % 2 == 1 else buggy_repo["wrong_patch"]
        return _patch_action(diff)

    model = _StubPolicy()
    tokenizer = _StubTokenizer()
    sandbox = ExecutionSandbox(timeout=30.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = GRPOTrainer(
        model=model, tokenizer=tokenizer, sandbox=sandbox, generate_fn=fake_generate,
        optimizer=optimizer, n_branches=4, max_iterations=3, max_tokens=300,
    )

    params_before = [p.detach().clone() for p in model.parameters()]

    branches, loss = trainer.train_step(
        repo_path=buggy_repo["repo_path"], problem_statement="add() is subtracting",
        fail_to_pass=FAIL_TO_PASS, pass_to_pass=PASS_TO_PASS,
        base_commit=buggy_repo["base_commit"],
    )

    assert len(branches) == 4
    rewards = [b.reward for b in branches]
    assert len(set(rewards)) > 1, "branches should have differing rewards, not all-equal"
    assert any(b.reward == pytest.approx(1.0) for b in branches), "at least one branch should fully resolve"
    assert any(b.reward < 1.0 for b in branches), "at least one branch should not fully resolve"
    assert torch.isfinite(loss)

    params_after = list(model.parameters())
    changed = any(
        not torch.allclose(before, after.detach())
        for before, after in zip(params_before, params_after)
    )
    assert changed, "optimizer step should have actually moved the policy's parameters"

    # Each branch's trajectory should parse into structured steps -- the
    # exact shape Item 8's action space needs.
    for b in branches:
        assert any(s.kind == "thought" for s in b.steps)


# ── derive_test_lists (0.5.0 Item 5 Stage 2 fallback) ─────────────────────


def test_derive_test_lists_splits_real_baseline_pass_fail(tmp_path):
    # File-level granularity, not per-test-function -- impacted_tests
    # resolves whole files via the import graph/naming convention;
    # pytest can run a whole file as a node id just as well as a single
    # function, and derive_test_lists's docstring is explicit this is
    # coarser than SWE-bench's own function-level annotation.
    #
    # Deliberately not buggy_repo here: its test_sanity.py doesn't import
    # mathutils at all, so it's correctly *not* impacted by a change to
    # mathutils.py -- real impacted_tests behavior, but it means that
    # fixture can't exercise the pass_to_pass side of the split. This repo
    # has a second, already-passing test that genuinely does depend on
    # the changed file, so both sides of the split are real.
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init"], cwd=repo)
    _run(["git", "config", "user.email", "t@t.com"], cwd=repo)
    _run(["git", "config", "user.name", "T"], cwd=repo)
    (repo / "mathutils.py").write_text("def add(a, b):\n    return a - b\n")
    (repo / "test_bug.py").write_text(
        "from mathutils import add\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    (repo / "test_other.py").write_text(
        "from mathutils import add\ndef test_zero_identity():\n    assert add(10, 0) == 10\n"
    )
    _run(["git", "add", "-A"], cwd=repo)
    _run(["git", "commit", "-m", "base"], cwd=repo)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()

    (repo / "mathutils.py").write_text("def add(a, b):\n    return a + b\n")
    fix_patch = subprocess.run(["git", "diff"], cwd=repo, capture_output=True, text=True).stdout
    _run(["git", "checkout", "--", "."], cwd=repo)

    sandbox = ExecutionSandbox(timeout=30.0)
    fail_to_pass, pass_to_pass = derive_test_lists(sandbox, str(repo), base_commit, fix_patch)
    assert "test_bug.py" in fail_to_pass
    assert "test_other.py" in pass_to_pass


def test_derive_test_lists_empty_for_patch_touching_untested_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init"], cwd=repo)
    _run(["git", "config", "user.email", "t@t.com"], cwd=repo)
    _run(["git", "config", "user.name", "T"], cwd=repo)
    (repo / "untested.py").write_text("VALUE = 1\n")
    _run(["git", "add", "-A"], cwd=repo)
    _run(["git", "commit", "-m", "base"], cwd=repo)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()

    patch = (
        "diff --git a/untested.py b/untested.py\n"
        "--- a/untested.py\n+++ b/untested.py\n"
        "@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"
    )
    sandbox = ExecutionSandbox(timeout=30.0)
    fail_to_pass, pass_to_pass = derive_test_lists(sandbox, str(repo), base_commit, patch)
    assert fail_to_pass == []
    assert pass_to_pass == []


def test_grpo_trainer_falls_back_to_derived_test_lists_when_unannotated(buggy_repo):
    """A curriculum record with no fail_to_pass/pass_to_pass (broader-
    corpus records past curated SWE-Gym) should still get real, non-zero
    grading via the derive_test_lists fallback, not silently score zero."""
    call_count = {"n": 0}

    def fake_generate(prompt, max_tokens=300, think=True):
        call_count["n"] += 1
        return _patch_action(buggy_repo["fix_patch"])

    model = _StubPolicy()
    tokenizer = _StubTokenizer()
    sandbox = ExecutionSandbox(timeout=30.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = GRPOTrainer(
        model=model, tokenizer=tokenizer, sandbox=sandbox, generate_fn=fake_generate,
        optimizer=optimizer, n_branches=2, max_iterations=3, max_tokens=300,
    )

    branches, loss = trainer.train_step(
        repo_path=buggy_repo["repo_path"], problem_statement="add() is subtracting",
        fail_to_pass=[], pass_to_pass=[],  # deliberately unannotated
        base_commit=buggy_repo["base_commit"],
    )

    assert len(branches) == 2
    assert all(b.reward == pytest.approx(1.0) for b in branches), \
        "derived test lists should correctly grade the real fix as fully resolved"
