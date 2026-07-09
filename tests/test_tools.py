"""Tests for uchi/tools.py's RepoToolRegistry (0.5.0 Item 7).

Real, throwaway git repo + real ExecutionSandbox — read_file/grep_repo hit
actual files on disk, apply_patch/run_tests hit actual git/pytest, no mocks.
"""
import subprocess

import pytest

from uchi.execution_sandbox import ExecutionSandbox
from uchi.tools import RepoToolRegistry
from uchi.workspace import WorkspaceViolation


def _run(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, f"{cmd} failed: {result.stderr}"
    return result


@pytest.fixture
def checked_out_repo(tmp_path):
    """A real git repo, checked out into a sandbox dir via ExecutionSandbox,
    with a bug + a failing test + a passing test — same fixture shape as
    test_execution_sandbox.py's buggy_repo, minus the pre-built patches
    (the tools apply their own).
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

    sandbox = ExecutionSandbox(timeout=30.0)
    repo_dir = sandbox.checkout_repo(str(repo), base_commit=base_commit)
    yield sandbox, repo_dir

    import shutil
    shutil.rmtree(repo_dir.parent, ignore_errors=True)


def test_read_file_returns_numbered_lines(checked_out_repo):
    sandbox, repo_dir = checked_out_repo
    tools = RepoToolRegistry(sandbox, repo_dir)

    result = tools.execute("read_file", "mathutils.py")
    assert result.ok
    assert "1: def add(a, b):" in result.output
    assert "2:     return a - b" in result.output


def test_read_file_respects_line_range(checked_out_repo):
    sandbox, repo_dir = checked_out_repo
    tools = RepoToolRegistry(sandbox, repo_dir)

    result = tools.execute("read_file", "mathutils.py:2-2")
    assert result.ok
    assert result.output == "2:     return a - b"


def test_read_file_missing_file(checked_out_repo):
    sandbox, repo_dir = checked_out_repo
    tools = RepoToolRegistry(sandbox, repo_dir)

    result = tools.execute("read_file", "nope.py")
    assert not result.ok
    assert "no such file" in result.output


def test_read_file_rejects_path_escape(checked_out_repo):
    sandbox, repo_dir = checked_out_repo
    tools = RepoToolRegistry(sandbox, repo_dir)

    result = tools.execute("read_file", "../../../etc/passwd")
    assert not result.ok


def test_grep_repo_finds_matches(checked_out_repo):
    sandbox, repo_dir = checked_out_repo
    tools = RepoToolRegistry(sandbox, repo_dir)

    result = tools.execute("grep_repo", r"def add\(")
    assert result.ok
    assert "mathutils.py:1:" in result.output


def test_grep_repo_no_matches(checked_out_repo):
    sandbox, repo_dir = checked_out_repo
    tools = RepoToolRegistry(sandbox, repo_dir)

    result = tools.execute("grep_repo", r"totally_absent_symbol_zzz")
    assert result.ok
    assert result.output == "no matches"


def test_grep_repo_invalid_regex(checked_out_repo):
    sandbox, repo_dir = checked_out_repo
    tools = RepoToolRegistry(sandbox, repo_dir)

    result = tools.execute("grep_repo", "(unclosed")
    assert not result.ok
    assert "invalid regex" in result.output


def test_apply_patch_and_run_tests_round_trip(checked_out_repo):
    sandbox, repo_dir = checked_out_repo
    tools = RepoToolRegistry(sandbox, repo_dir)

    before = tools.execute("run_tests", "test_bug.py::test_add")
    assert before.ok
    assert "FAIL test_bug.py::test_add" in before.output

    (repo_dir / "mathutils.py").write_text("def add(a, b):\n    return a + b\n")
    diff = subprocess.run(["git", "diff"], cwd=repo_dir, capture_output=True, text=True).stdout
    subprocess.run(["git", "checkout", "--", "."], cwd=repo_dir, capture_output=True, text=True)

    applied = tools.execute("apply_patch", diff)
    assert applied.ok
    assert "applied cleanly" in applied.output

    after = tools.execute("run_tests", "test_bug.py::test_add,test_sanity.py::test_sanity")
    assert after.ok
    assert "PASS test_bug.py::test_add" in after.output
    assert "PASS test_sanity.py::test_sanity" in after.output


def test_apply_patch_failure_reports_stderr(checked_out_repo):
    sandbox, repo_dir = checked_out_repo
    tools = RepoToolRegistry(sandbox, repo_dir)

    result = tools.execute("apply_patch", "not a real diff\nat all\n")
    assert not result.ok
    assert "patch failed to apply" in result.output


def test_run_tests_no_ids_given(checked_out_repo):
    sandbox, repo_dir = checked_out_repo
    tools = RepoToolRegistry(sandbox, repo_dir)

    result = tools.execute("run_tests", "")
    assert not result.ok
    assert "no test ids" in result.output


def test_unknown_tool(checked_out_repo):
    sandbox, repo_dir = checked_out_repo
    tools = RepoToolRegistry(sandbox, repo_dir)

    result = tools.execute("fly_to_the_moon", "x")
    assert not result.ok
    assert "unknown tool" in result.output


def test_descriptions_mentions_all_four_tools(checked_out_repo):
    sandbox, repo_dir = checked_out_repo
    tools = RepoToolRegistry(sandbox, repo_dir)

    desc = tools.descriptions()
    for name in ("read_file", "grep_repo", "apply_patch", "run_tests"):
        assert name in desc
