"""Tests for uchi/sandbox_isolation.py (0.5.0 Item 5 Stage 2).

Real bwrap runs (skipped gracefully, not hard-failed, if bwrap isn't
installed) -- confirming isolation is real: a passing command reports
success, a failing command's exit code and stderr survive the wrapper
unchanged, and network access is genuinely blocked, not just assumed.
"""
import socket

import pytest

from uchi.sandbox_isolation import bwrap_available, run_isolated

requires_bwrap = pytest.mark.skipif(not bwrap_available(), reason="bwrap not installed in this environment")


@requires_bwrap
def test_run_isolated_passing_command(tmp_path):
    result = run_isolated(["echo", "hello"], cwd=tmp_path, timeout=10.0)
    assert result.returncode == 0
    assert "hello" in result.stdout


@requires_bwrap
def test_run_isolated_failing_command_reports_real_exit_code(tmp_path):
    result = run_isolated(["python3", "-c", "import sys; sys.exit(7)"], cwd=tmp_path, timeout=10.0)
    assert result.returncode == 7


@requires_bwrap
def test_run_isolated_preserves_real_stderr(tmp_path):
    result = run_isolated(
        ["python3", "-c", "import sys; print('boom', file=sys.stderr); sys.exit(1)"],
        cwd=tmp_path, timeout=10.0,
    )
    assert result.returncode == 1
    assert "boom" in result.stderr


@requires_bwrap
def test_run_isolated_blocks_network(tmp_path):
    # Real socket connect attempt inside the sandbox -- confirms
    # --unshare-net actually blocks it (OSError), not a no-op wrapper.
    probe = (
        "import socket, sys\n"
        "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "s.settimeout(2)\n"
        "try:\n"
        "    s.connect(('8.8.8.8', 53))\n"
        "    sys.exit(1)\n"  # connection should NOT succeed
        "except OSError:\n"
        "    sys.exit(0)\n"  # blocked, as expected
    )
    result = run_isolated(["python3", "-c", probe], cwd=tmp_path, timeout=10.0)
    assert result.returncode == 0, f"network connect should have been blocked: {result.stdout} {result.stderr}"


@requires_bwrap
def test_run_isolated_env_setenv(tmp_path):
    result = run_isolated(
        ["python3", "-c", "import os, sys; sys.exit(0 if os.environ.get('MY_FLAG') == 'x' else 1)"],
        cwd=tmp_path, timeout=10.0, env={"MY_FLAG": "x"},
    )
    assert result.returncode == 0


def test_run_isolated_falls_back_without_bwrap(tmp_path, monkeypatch):
    """When bwrap isn't available, run_isolated must still work -- plain
    subprocess.run, not a hard failure."""
    monkeypatch.setattr("uchi.sandbox_isolation._BWRAP_PATH", None)
    result = run_isolated(["echo", "fallback"], cwd=tmp_path, timeout=10.0)
    assert result.returncode == 0
    assert "fallback" in result.stdout


def test_bwrap_available_matches_real_environment():
    # This environment has bwrap installed (validated by hand before
    # building this module) -- confirm the detection itself is correct,
    # not just that callers degrade gracefully when it's absent.
    import shutil
    assert bwrap_available() == (shutil.which("bwrap") is not None)
