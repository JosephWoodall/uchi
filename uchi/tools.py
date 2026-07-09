"""tools.py — 0.5.0 Item 7's agent tool registry: read_file, grep_repo,
apply_patch, run_tests.

Ports the *pattern* of `efficient_llm_training/src/tools.py`'s registry
(name -> callable, `execute(name, arg)` dispatch, tool descriptions folded
into the agent's system prompt) — not its content. That source registers
calculator/physics_lookup/browser for a general chatbot; this repo's actual
action space for `agentic_repair.py`'s SWE-bench loop is the four
code-repair primitives the itemized deliverables doc names. `read_file`/
`grep_repo` are plain filesystem operations (no semantic embedding) —
distinct from `code_retrieval.py`'s `CodeIndex`, which builds a *searchable
index* up front for retrieval-shaped queries; these two are the direct,
un-indexed tool-call primitives a ReAct loop dispatches turn by turn, same
relationship as `workspace.py`'s `read_file` vs. `retrieval.py`'s indexed
`retrieve()`. `apply_patch`/`run_tests` wrap `execution_sandbox.py`'s
`ExecutionSandbox` methods directly — no new execution logic here, only
argument parsing and result formatting for a model that emits
`tool_name[argument]`-shaped actions as plain text (`agentic_repair.py`'s
ReAct parser, ported from `react_agent.py`'s regex approach).

`read_file` reuses `uchi/workspace.py`'s `resolve_path` for the escape
guard (one root-boundary check for the whole codebase, not a second
hand-rolled one here) — rooted at the tool registry's own bound
``repo_dir``, not `workspace.py`'s own default `.uchi/workspace` root.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .execution_sandbox import ExecutionSandbox
from .workspace import WorkspaceViolation, resolve_path

_GREP_MAX_MATCHES = 50
_GREP_SKIP_DIRS = frozenset({
    "__pycache__", ".git", "node_modules", ".tox", ".pytest_cache", ".mypy_cache",
})
_OBSERVATION_CHAR_CAP = 1500


@dataclass
class ToolResult:
    ok: bool
    output: str


class RepoToolRegistry:
    """Binds read_file/grep_repo/apply_patch/run_tests to one sandboxed repo
    checkout (``repo_dir``) for the duration of one repair episode. Not a
    global singleton registry like the ported source's `ToolRegistry` — each
    SWE-bench instance gets its own checkout, so the registry is
    constructed fresh per episode instead.
    """

    def __init__(self, sandbox: ExecutionSandbox, repo_dir: str | Path):
        self.sandbox = sandbox
        self.repo_dir = Path(repo_dir)
        self._tools: dict[str, Callable[[str], ToolResult]] = {
            "read_file": self._read_file,
            "grep_repo": self._grep_repo,
            "apply_patch": self._apply_patch,
            "run_tests": self._run_tests,
        }

    def descriptions(self) -> str:
        return (
            "- read_file[path] or read_file[path:start-end]: read a file "
            "(optionally a 1-indexed line range) from the repo\n"
            "- grep_repo[regex]: search all .py files in the repo for a regex "
            "pattern, returns file:line: matches\n"
            "- apply_patch[<unified diff>]: apply a unified diff to the repo\n"
            "- run_tests[test_id1,test_id2,...]: run specific pytest node ids "
            "against the repo's current state, returns pass/fail + failure output"
        )

    def execute(self, tool_name: str, arg: str) -> ToolResult:
        fn = self._tools.get(tool_name)
        if fn is None:
            return ToolResult(False, f"unknown tool {tool_name!r}. Available: {sorted(self._tools)}")
        try:
            return fn(arg)
        except Exception as e:
            return ToolResult(False, f"{tool_name} error: {e}")

    # ── read_file ─────────────────────────────────────────────────────────

    def _read_file(self, arg: str) -> ToolResult:
        path_spec, _, line_range = arg.strip().partition(":")
        try:
            resolved = resolve_path(path_spec.strip(), root=str(self.repo_dir))
        except WorkspaceViolation as e:
            return ToolResult(False, str(e))
        target = Path(resolved)
        if not target.is_file():
            return ToolResult(False, f"no such file: {path_spec}")

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        offset = 1
        if line_range:
            start_s, _, end_s = line_range.partition("-")
            start = max(int(start_s), 1) if start_s else 1
            end = int(end_s) if end_s else len(lines)
            lines = lines[start - 1:end]
            offset = start
        numbered = "\n".join(f"{offset + i}: {line}" for i, line in enumerate(lines))
        return ToolResult(True, numbered[:_OBSERVATION_CHAR_CAP])

    # ── grep_repo ─────────────────────────────────────────────────────────

    def _grep_repo(self, arg: str) -> ToolResult:
        pattern = arg.strip()
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return ToolResult(False, f"invalid regex {pattern!r}: {e}")

        matches: list[str] = []
        for path in sorted(self.repo_dir.rglob("*.py")):
            if any(part in _GREP_SKIP_DIRS for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            rel = path.relative_to(self.repo_dir)
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{rel}:{lineno}: {line.strip()}")
                    if len(matches) >= _GREP_MAX_MATCHES:
                        matches.append(f"... (capped at {_GREP_MAX_MATCHES} matches)")
                        return ToolResult(True, "\n".join(matches))
        if not matches:
            return ToolResult(True, "no matches")
        return ToolResult(True, "\n".join(matches))

    # ── apply_patch / run_tests ──────────────────────────────────────────

    def _apply_patch(self, arg: str) -> ToolResult:
        result = self.sandbox.apply_patch(self.repo_dir, arg)
        if result.applied:
            return ToolResult(True, "patch applied cleanly")
        return ToolResult(False, f"patch failed to apply:\n{result.stderr}"[:_OBSERVATION_CHAR_CAP])

    def _run_tests(self, arg: str) -> ToolResult:
        test_ids = [t.strip() for t in arg.split(",") if t.strip()]
        if not test_ids:
            return ToolResult(False, "no test ids given (comma-separated pytest node ids)")

        result = self.sandbox.run_tests(self.repo_dir, test_ids)
        if result.timed_out:
            return ToolResult(False, "test run timed out")
        if result.error:
            return ToolResult(False, f"test run error: {result.error}")

        lines = []
        for test_id in test_ids:
            status = "PASS" if result.results.get(test_id) else "FAIL"
            lines.append(f"{status} {test_id}")
            if status == "FAIL" and test_id in result.output:
                lines.append(result.output[test_id][:_OBSERVATION_CHAR_CAP])
        return ToolResult(True, "\n".join(lines))
