"""Tests for uchi/execution_sandbox.py (0.5.0 Item 5).

Builds a real, throwaway git repo per test — a buggy function plus a
failing test (FAIL_TO_PASS) and an always-passing test (PASS_TO_PASS) —
so the sandbox's patch-apply and per-test grading are exercised against
actual git/pytest behavior, not mocks.
"""
import subprocess

import pytest

from uchi.execution_sandbox import ExecutionSandbox


def _run(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, f"{cmd} failed: {result.stderr}"
    return result


@pytest.fixture
def buggy_repo(tmp_path):
    """A git repo with add() implemented as subtraction (bug), one test that
    exercises the bug (FAIL_TO_PASS) and one unrelated passing test (PASS_TO_PASS).
    """
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

    # Generate a real fix patch via git diff, then revert the working tree
    # so the repo directory itself stays exactly at base_commit.
    (repo / "mathutils.py").write_text("def add(a, b):\n    return a + b\n")
    fix_patch = subprocess.run(
        ["git", "diff"], cwd=repo, capture_output=True, text=True
    ).stdout
    _run(["git", "checkout", "--", "."], cwd=repo)

    # A patch that fixes the bug but also breaks the sanity test — used to
    # confirm PASS_TO_PASS regressions are caught, not just FAIL_TO_PASS.
    (repo / "mathutils.py").write_text("def add(a, b):\n    return a + b\n")
    (repo / "test_sanity.py").write_text("def test_sanity():\n    assert 1 == 2\n")
    regressing_patch = subprocess.run(
        ["git", "diff"], cwd=repo, capture_output=True, text=True
    ).stdout
    _run(["git", "checkout", "--", "."], cwd=repo)

    return {
        "repo_path": str(repo),
        "base_commit": base_commit,
        "fix_patch": fix_patch,
        "regressing_patch": regressing_patch,
    }


def test_correct_patch_resolves(buggy_repo):
    sandbox = ExecutionSandbox(timeout=30.0)
    result = sandbox.evaluate(
        repo_path=buggy_repo["repo_path"],
        patch_text=buggy_repo["fix_patch"],
        fail_to_pass=["test_bug.py::test_add"],
        pass_to_pass=["test_sanity.py::test_sanity"],
        base_commit=buggy_repo["base_commit"],
    )
    assert result.patch_applied
    assert result.fail_to_pass_ok
    assert result.pass_to_pass_ok
    assert result.resolved
    assert result.reward == pytest.approx(1.0)


def test_patch_that_regresses_pass_to_pass_is_not_resolved(buggy_repo):
    sandbox = ExecutionSandbox(timeout=30.0)
    result = sandbox.evaluate(
        repo_path=buggy_repo["repo_path"],
        patch_text=buggy_repo["regressing_patch"],
        fail_to_pass=["test_bug.py::test_add"],
        pass_to_pass=["test_sanity.py::test_sanity"],
        base_commit=buggy_repo["base_commit"],
    )
    assert result.patch_applied
    assert result.fail_to_pass_ok
    assert not result.pass_to_pass_ok
    assert not result.resolved
    assert result.reward == pytest.approx(0.7)


def test_garbage_patch_does_not_apply(buggy_repo):
    sandbox = ExecutionSandbox(timeout=30.0)
    result = sandbox.evaluate(
        repo_path=buggy_repo["repo_path"],
        patch_text="not a real diff\nat all\n",
        fail_to_pass=["test_bug.py::test_add"],
        pass_to_pass=["test_sanity.py::test_sanity"],
        base_commit=buggy_repo["base_commit"],
    )
    assert not result.patch_applied
    assert not result.resolved
    assert result.reward == 0.0


def test_unpatched_repo_fails_fail_to_pass(buggy_repo):
    """Sanity check on the fixture itself: without any patch, the bug test
    genuinely fails and the sanity test genuinely passes.
    """
    sandbox = ExecutionSandbox(timeout=30.0)
    repo_dir = sandbox.checkout_repo(buggy_repo["repo_path"], base_commit=buggy_repo["base_commit"])
    try:
        f2p = sandbox.run_tests(repo_dir, ["test_bug.py::test_add"])
        p2p = sandbox.run_tests(repo_dir, ["test_sanity.py::test_sanity"])
        assert f2p.passed("test_bug.py::test_add") is False
        assert p2p.passed("test_sanity.py::test_sanity") is True
    finally:
        import shutil
        shutil.rmtree(repo_dir.parent, ignore_errors=True)
