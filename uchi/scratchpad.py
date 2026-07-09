"""scratchpad.py — sandboxed Python execution for tool calls (0.4.0 Item 3).

Uchi predicts discrete tokens; it cannot natively calculate continuous
regressions, calibrate probabilistic uncertainties, or dynamically test
code. This module lets it hand arbitrary multi-line Python to a real
interpreter and get ``stdout``/``stderr``/return code back as text.

Distinct from ``code_engine.REPLOracle``:
  - ``REPLOracle.verify()``  — fast compile-only check used to score MCTS
    code-generation candidates.
  - ``REPLOracle.execute()`` — skill-shaped convention (code must define
    ``def run():``), used by the empirical-synthesis fallback and
    ``ProceduralMemory``.
  - ``run_python()`` (here) — unconstrained arbitrary code, no ``run()``
    requirement. This is what a tool call executes.

Every run is written into the Item 2 workspace sandbox
(``uchi/workspace.py``) rather than the system temp directory, so
generated files stay inside ``.uchi/workspace`` by default.
"""
from __future__ import annotations

import subprocess
import sys
import uuid
from dataclasses import dataclass

from .workspace import DEFAULT_ROOT, delete_file, write_file


@dataclass
class ScratchpadResult:
    stdout: str
    stderr: str
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def as_text(self) -> str:
        """Flatten to text tokens for ingestion back into Uchi's context."""
        parts = [f"[exit code: {self.returncode}]"]
        if self.stdout:
            parts.append(f"stdout:\n{self.stdout}")
        if self.stderr:
            parts.append(f"stderr:\n{self.stderr}")
        return "\n".join(parts)


def run_python(code: str, timeout: float = 5.0, root: str = DEFAULT_ROOT) -> ScratchpadResult:
    """Execute *code* in a subprocess sandboxed under *root*.

    Accepts any multi-line Python string — no ``def run():`` convention
    required. Returns stdout, stderr, and the process return code.
    """
    filename = f"_scratch_{uuid.uuid4().hex}.py"
    path = write_file(filename, code, root=root)
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, timeout=timeout, text=True,
        )
        return ScratchpadResult(result.stdout, result.stderr, result.returncode)
    except subprocess.TimeoutExpired:
        return ScratchpadResult("", "TimeoutExpired", -1)
    except Exception as e:
        return ScratchpadResult("", str(e), -1)
    finally:
        try:
            delete_file(filename, root=root)
        except OSError:
            pass


def lint_python(code: str, timeout: float = 5.0, root: str = DEFAULT_ROOT) -> str:
    """Static-analysis pass over *code* via ``ruff`` (0.4.0 Item 16.7),
    sandboxed the same way as ``run_python``.

    Gives instant "red squiggly line" feedback — syntax and obvious
    correctness issues — before spending a subprocess execution on code
    that was never going to run, speeding up the autonomous debugging
    loop. Returns ``"No issues found."`` when clean, or ruff's own
    findings as text otherwise. Falls back to a plain message if ``ruff``
    isn't installed, rather than raising.
    """
    filename = f"_lint_{uuid.uuid4().hex}.py"
    path = write_file(filename, code, root=root)
    try:
        result = subprocess.run(
            ["ruff", "check", path, "--no-cache"],
            capture_output=True, timeout=timeout, text=True,
        )
        if result.returncode == 0:
            return "No issues found."
        return (result.stdout or result.stderr).strip()
    except FileNotFoundError:
        return "ruff is not installed; skipping lint."
    except subprocess.TimeoutExpired:
        return "lint timed out."
    finally:
        try:
            delete_file(filename, root=root)
        except OSError:
            pass
