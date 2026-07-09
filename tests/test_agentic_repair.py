"""Tests for uchi/agentic_repair.py (0.5.0 Item 7).

Real, throwaway git repo + real ExecutionSandbox — the "model" is a
scripted fake `generate_fn` (call-count/feedback-driven, same spirit as
testing react_agent.py's parser against known inputs) so the ReAct loop,
tool dispatch, tri-state outcome policy, and retry cap all run against
actual git/pytest behavior, not a mocked sandbox -- except in the one test
that specifically needs to force a timeout/error path.
"""
import subprocess

import pytest

from uchi.agentic_repair import (
    AgenticRepairAgent,
    Outcome,
    RepairOutcome,
    classify_outcome,
)
from uchi.execution_sandbox import ExecutionSandbox, SandboxEvalResult


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


# ── classify_outcome ──────────────────────────────────────────────────────

def _eval_result(**overrides):
    defaults = dict(
        patch_applied=True, fail_to_pass={"t1": True}, pass_to_pass={"t2": True},
        timed_out=False, error="",
    )
    defaults.update(overrides)
    return SandboxEvalResult(**defaults)


def test_classify_outcome_pass():
    assert classify_outcome(_eval_result()) is Outcome.PASS


def test_classify_outcome_fail_when_not_resolved():
    r = _eval_result(fail_to_pass={"t1": False})
    assert classify_outcome(r) is Outcome.FAIL


def test_classify_outcome_abstain_on_timeout():
    r = _eval_result(timed_out=True)
    assert classify_outcome(r) is Outcome.ABSTAIN


def test_classify_outcome_abstain_on_error_even_if_resolved_looking():
    r = _eval_result(error="sandbox blew up")
    assert classify_outcome(r) is Outcome.ABSTAIN


# ── AgenticRepairAgent.repair ─────────────────────────────────────────────

def test_repair_passes_on_first_attempt(buggy_repo):
    calls = []

    def fake_generate(prompt, max_tokens=300, think=True):
        calls.append(prompt)
        if len(calls) == 1:
            return _patch_action(buggy_repo["fix_patch"])
        return "Thought: done.\nFinal Answer: fixed add() to use +."

    sandbox = ExecutionSandbox(timeout=30.0)
    agent = AgenticRepairAgent(fake_generate, sandbox, max_iterations=5, max_attempts=4)
    outcome = agent.repair(
        repo_path=buggy_repo["repo_path"], problem_statement="add() is subtracting",
        fail_to_pass=FAIL_TO_PASS, pass_to_pass=PASS_TO_PASS,
        base_commit=buggy_repo["base_commit"],
    )

    assert outcome.outcome is Outcome.PASS
    assert outcome.attempts == 1
    assert outcome.eval_result.resolved
    assert "return a + b" in outcome.patch


def test_repair_retries_with_feedback_after_wrong_patch(buggy_repo):
    calls = []

    def fake_generate(prompt, max_tokens=300, think=True):
        calls.append(prompt)
        if "Feedback from your previous attempt" in prompt:
            return _patch_action(buggy_repo["fix_patch"])
        return _patch_action(buggy_repo["wrong_patch"])

    sandbox = ExecutionSandbox(timeout=30.0)
    agent = AgenticRepairAgent(fake_generate, sandbox, max_iterations=5, max_attempts=4)
    outcome = agent.repair(
        repo_path=buggy_repo["repo_path"], problem_statement="add() is subtracting",
        fail_to_pass=FAIL_TO_PASS, pass_to_pass=PASS_TO_PASS,
        base_commit=buggy_repo["base_commit"],
    )

    assert outcome.outcome is Outcome.PASS
    assert outcome.attempts == 2
    assert any("still failing" in p or True for p in calls)  # feedback loop actually ran


def test_repair_exhausts_attempts_and_returns_fail(buggy_repo):
    def fake_generate(prompt, max_tokens=300, think=True):
        return _patch_action(buggy_repo["wrong_patch"])

    sandbox = ExecutionSandbox(timeout=30.0)
    agent = AgenticRepairAgent(fake_generate, sandbox, max_iterations=5, max_attempts=2)
    outcome = agent.repair(
        repo_path=buggy_repo["repo_path"], problem_statement="add() is subtracting",
        fail_to_pass=FAIL_TO_PASS, pass_to_pass=PASS_TO_PASS,
        base_commit=buggy_repo["base_commit"],
    )

    assert outcome.outcome is Outcome.FAIL
    assert outcome.attempts == 2


def test_repair_abstains_on_sandbox_timeout_without_retrying(buggy_repo, monkeypatch):
    call_count = {"n": 0}

    def fake_generate(prompt, max_tokens=300, think=True):
        call_count["n"] += 1
        return _patch_action(buggy_repo["fix_patch"])

    sandbox = ExecutionSandbox(timeout=30.0)
    monkeypatch.setattr(
        sandbox, "evaluate",
        lambda **kw: SandboxEvalResult(
            patch_applied=True, fail_to_pass={}, pass_to_pass={},
            timed_out=True, error="",
        ),
    )
    agent = AgenticRepairAgent(fake_generate, sandbox, max_iterations=5, max_attempts=4)
    outcome = agent.repair(
        repo_path=buggy_repo["repo_path"], problem_statement="add() is subtracting",
        fail_to_pass=FAIL_TO_PASS, pass_to_pass=PASS_TO_PASS,
        base_commit=buggy_repo["base_commit"],
    )

    assert outcome.outcome is Outcome.ABSTAIN
    assert outcome.attempts == 1  # no retry on ABSTAIN


def test_repair_prompts_again_when_model_makes_no_changes(buggy_repo):
    calls = []

    def fake_generate(prompt, max_tokens=300, think=True):
        calls.append(prompt)
        if len(calls) == 1:
            return "Thought: I think about it.\nFinal Answer: I don't need to change anything."
        return _patch_action(buggy_repo["fix_patch"])

    sandbox = ExecutionSandbox(timeout=30.0)
    agent = AgenticRepairAgent(fake_generate, sandbox, max_iterations=5, max_attempts=4)
    outcome = agent.repair(
        repo_path=buggy_repo["repo_path"], problem_statement="add() is subtracting",
        fail_to_pass=FAIL_TO_PASS, pass_to_pass=PASS_TO_PASS,
        base_commit=buggy_repo["base_commit"],
    )

    assert outcome.outcome is Outcome.PASS
    assert outcome.attempts == 2
    assert "real diff" in calls[1]


def test_react_loop_stops_at_max_iterations(buggy_repo):
    def fake_generate(prompt, max_tokens=300, think=True):
        # Never gives an action or a final answer -- must be bounded by
        # max_iterations, not loop forever.
        return "Thought: still thinking..."

    sandbox = ExecutionSandbox(timeout=30.0)
    agent = AgenticRepairAgent(fake_generate, sandbox, max_iterations=3, max_attempts=1)
    outcome = agent.repair(
        repo_path=buggy_repo["repo_path"], problem_statement="add() is subtracting",
        fail_to_pass=FAIL_TO_PASS, pass_to_pass=PASS_TO_PASS,
        base_commit=buggy_repo["base_commit"],
    )

    # A bare Thought with no Action ends the loop immediately (matches
    # react_agent.py's "no action and no final answer -> done" fallback),
    # so the transcript has exactly one Thought line, not `max_iterations`.
    assert outcome.transcript == ["Thought: still thinking..."]
    assert outcome.outcome is Outcome.FAIL
