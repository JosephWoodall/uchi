"""Tests for oracle.py's Item 7 code-verification veto layer
(SandboxVerificationChecker + FactCheckOracle.is_patch_verified).
"""
import subprocess

import pytest

from uchi.execution_sandbox import ExecutionSandbox, SandboxVerificationChecker
from uchi.oracle import FactCheckOracle


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
    _run(["git", "add", "-A"], cwd=repo)
    _run(["git", "commit", "-m", "buggy base"], cwd=repo)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()

    (repo / "mathutils.py").write_text("def add(a, b):\n    return a + b\n")
    fix_patch = subprocess.run(["git", "diff"], cwd=repo, capture_output=True, text=True).stdout
    _run(["git", "checkout", "--", "."], cwd=repo)

    return {"repo_path": str(repo), "base_commit": base_commit, "fix_patch": fix_patch}


class _RaisingSandbox:
    """Stands in for a sandbox whose evaluate() blows up (e.g. a real timeout
    or environment failure) -- exercises the fail-CLOSED path.
    """
    def evaluate(self, **kwargs):
        raise RuntimeError("simulated sandbox failure")


def test_correct_patch_passes_oracle_veto(buggy_repo):
    checker = SandboxVerificationChecker(ExecutionSandbox(timeout=30.0))
    oracle = FactCheckOracle(sandbox_checker=checker)
    assert oracle.is_patch_verified(
        repo_path=buggy_repo["repo_path"],
        patch_text=buggy_repo["fix_patch"],
        fail_to_pass=["test_bug.py::test_add"],
        pass_to_pass=[],
        base_commit=buggy_repo["base_commit"],
    )
    assert oracle.sandbox_veto_log == []


def test_unpatched_repo_is_vetoed(buggy_repo):
    checker = SandboxVerificationChecker(ExecutionSandbox(timeout=30.0))
    oracle = FactCheckOracle(sandbox_checker=checker)
    assert not oracle.is_patch_verified(
        repo_path=buggy_repo["repo_path"],
        patch_text="",  # no patch applied -- bug still present
        fail_to_pass=["test_bug.py::test_add"],
        pass_to_pass=[],
        base_commit=buggy_repo["base_commit"],
    )
    assert len(oracle.sandbox_veto_log) == 1


def test_sandbox_exception_fails_closed_not_open():
    """The defining behavior this layer exists for: unlike entailment/
    relational, an internal exception must veto, not pass through.
    """
    checker = SandboxVerificationChecker(_RaisingSandbox())
    oracle = FactCheckOracle(sandbox_checker=checker)
    assert not oracle.is_patch_verified(
        repo_path="/irrelevant", patch_text="x", fail_to_pass=["t"], pass_to_pass=[],
    )
    assert len(oracle.sandbox_veto_log) == 1


def test_no_sandbox_checker_raises_rather_than_silently_passing():
    oracle = FactCheckOracle()  # sandbox_checker defaults to None
    with pytest.raises(RuntimeError):
        oracle.is_patch_verified(
            repo_path="/irrelevant", patch_text="x", fail_to_pass=["t"], pass_to_pass=[],
        )


def test_word_overlap_cascade_untouched_when_sandbox_checker_absent():
    """Constructor symmetry check: adding sandbox_checker=None must not
    change is_grounded()'s existing, unrelated behavior at all.
    """
    oracle = FactCheckOracle()
    assert oracle.is_grounded("The sky is blue.", ["The sky is blue today."])
