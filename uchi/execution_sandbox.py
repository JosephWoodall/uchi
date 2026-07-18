"""execution_sandbox.py — patch-and-test execution for SWE-bench-shaped tasks
(0.5.0 Item 5).

Ports and extends ``efficient_llm_training/src/rl_env.py``'s
``ExecutionSandbox`` (single-file: strip markdown fencing, run one test
assertion in a subprocess) into something that operates at repo scale: copy
a real repo at a specific commit, apply a unified diff, run its actual test
suite, and grade the result against SWE-bench's own convention —
``FAIL_TO_PASS`` (tests the patch must make pass) and ``PASS_TO_PASS``
(tests the patch must not break).

Distinct from the other execution paths already in this codebase:
  - ``code_engine.REPLOracle`` — single-file compile/execute, scores MCTS
    code-generation candidates. No patch, no repo, no test suite.
  - ``scratchpad.run_python`` — unconstrained single-script execution for
    tool calls. Same idea, still no repo/patch/test-suite concept.
  - ``ExecutionSandbox`` (here) — the only one that operates on an actual
    repo checkout plus a patch plus a real test suite. Shared by Item 6
    (training-time GRPO self-play, reward signal) and Item 7 (inference-
    time verification cascade layer) per the 0.5.0 plan.

Stage 1 only (subprocess + timeout + an explicit, small test-id list —
the FAIL_TO_PASS/PASS_TO_PASS lists SWE-bench itself provides per
instance, so "which tests matter" is already answered at eval time and
doesn't need separate test-impact-analysis subsetting there). Container-
level isolation for self-play volume against the broader training corpus,
where no such list exists, is Stage 2 — tracked in ``tasks/todo.md``, not
built here. Running arbitrary model-generated patches with bare subprocess
isolation is only safe at the volume and trust level this stage is scoped
for; do not point this at untrusted patches at self-play scale before
Stage 2 lands.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import sandbox_isolation
from .workspace import DEFAULT_ROOT

SANDBOX_SUBDIR = "sandbox"


@dataclass
class PatchResult:
    applied: bool
    stdout: str
    stderr: str


@dataclass
class TestRunResult:
    """Per-test-id pass/fail, plus whether the run itself completed.

    ``output`` (0.5.0 Item 7 addition): each test id's captured
    stdout+stderr from its own pytest invocation, verbatim -- the actual
    assertion diff / stack trace, not just the pass/fail boolean.
    ``results``/``timed_out``/``error`` are unchanged from Stage 1;
    nothing that already reads only ``results``/``passed()`` needs to
    change. Needed by the agentic repair loop (``agentic_repair.py``),
    which has to splice a failing attempt's real failure text back in as
    an Observation -- a bare boolean has nothing for a proposer to learn
    from.
    """
    results: dict[str, bool] = field(default_factory=dict)
    timed_out: bool = False
    error: str = ""
    output: dict[str, str] = field(default_factory=dict)

    def passed(self, test_id: str) -> bool:
        return self.results.get(test_id, False)


@dataclass
class SandboxEvalResult:
    """Full grading of one candidate patch against one SWE-bench-shaped instance."""
    patch_applied: bool
    fail_to_pass: dict[str, bool]
    pass_to_pass: dict[str, bool]
    timed_out: bool
    error: str
    stdout: str = ""
    stderr: str = ""

    @property
    def fail_to_pass_ok(self) -> bool:
        return bool(self.fail_to_pass) and all(self.fail_to_pass.values())

    @property
    def pass_to_pass_ok(self) -> bool:
        return all(self.pass_to_pass.values())

    @property
    def resolved(self) -> bool:
        """SWE-bench's own bar: every FAIL_TO_PASS now passes, no PASS_TO_PASS regressed."""
        return self.patch_applied and self.fail_to_pass_ok and self.pass_to_pass_ok

    @property
    def reward(self) -> float:
        """Partial credit: weighted fraction passing, zero if the patch didn't even apply.

        FAIL_TO_PASS is weighted higher — it's the actual fix being scored;
        PASS_TO_PASS is a no-regression check, not the goal itself.
        """
        if not self.patch_applied:
            return 0.0
        f2p = list(self.fail_to_pass.values())
        p2p = list(self.pass_to_pass.values())
        f2p_frac = sum(f2p) / len(f2p) if f2p else 1.0
        p2p_frac = sum(p2p) / len(p2p) if p2p else 1.0
        return 0.7 * f2p_frac + 0.3 * p2p_frac


class ExecutionSandbox:
    """Copies a repo at a commit, applies a patch, runs real tests, grades the result."""

    def __init__(self, timeout: float = 120.0, root: str = DEFAULT_ROOT, isolate: bool = False):
        self.timeout = timeout
        self.root = Path(root) / SANDBOX_SUBDIR
        # Stage 2 (0.5.0 Item 5): container/VM-level isolation via
        # sandbox_isolation.run_isolated, on top of Stage 1's bare
        # subprocess execution. Defaults False -- additive, not a
        # behavior change for any existing caller; opt in once self-play
        # actually runs at volume (see sandbox_isolation.py's docstring).
        self.isolate = isolate

    def _new_sandbox_dir(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        d = self.root / uuid.uuid4().hex
        d.mkdir(parents=True, exist_ok=False)
        return d

    def checkout_repo(self, repo_path: str, base_commit: str | None = None) -> Path:
        """Copy a local repo checkout into an isolated sandbox directory.

        *repo_path* must already be a local clone (fetching from a remote
        is a corpus-ingestion concern, not the sandbox's job). If
        *base_commit* is given, the sandbox copy is reset to it — the
        source repo is never mutated.
        """
        sandbox_dir = self._new_sandbox_dir()
        repo_dir = sandbox_dir / "repo"
        shutil.copytree(repo_path, repo_dir, symlinks=False)
        if base_commit:
            result = subprocess.run(
                ["git", "checkout", "--force", base_commit],
                cwd=repo_dir, capture_output=True, text=True, timeout=30.0,
            )
            if result.returncode != 0:
                shutil.rmtree(sandbox_dir, ignore_errors=True)
                raise RuntimeError(
                    f"failed to checkout {base_commit!r} in sandbox copy: {result.stderr}"
                )
        return repo_dir

    def apply_patch(self, repo_dir: Path, patch_text: str) -> PatchResult:
        """Apply a unified diff inside *repo_dir* via ``git apply``.

        No fallback to increasingly permissive apply modes on failure —
        a patch that doesn't apply cleanly is a failed candidate, not
        something to coerce into applying.

        When ``self.isolate``, the patch tempfile is written inside
        *repo_dir* itself rather than the system temp dir -- a bwrap
        sandbox only sees *repo_dir* (read-write) and `/usr` (read-only),
        not the host's real `/tmp` (replaced by an empty `--tmpfs`), so a
        patch path outside *repo_dir* would be invisible to the isolated
        `git apply` call. Cleaned up in `finally` either way, before any
        other diff-taking touches *repo_dir*.
        """
        patch_dir = str(repo_dir) if self.isolate else None
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False, dir=patch_dir) as f:
            f.write(patch_text)
            patch_path = f.name
        try:
            cmd = ["git", "apply", "--whitespace=fix", patch_path]
            if self.isolate:
                result = sandbox_isolation.run_isolated(cmd, cwd=repo_dir, timeout=30.0)
            else:
                result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, timeout=30.0)
            return PatchResult(
                applied=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired:
            return PatchResult(applied=False, stdout="", stderr="patch apply timed out")
        finally:
            Path(patch_path).unlink(missing_ok=True)

    def run_tests(self, repo_dir: Path, test_ids: list[str]) -> TestRunResult:
        """Run each test id in its own pytest invocation, graded by that
        invocation's own exit code (0 = passed).

        One subprocess per test rather than a single batched run parsed
        after the fact — a batched run's output still has to be mapped
        back to individual node ids (pytest's JUnit XML uses a
        dotted-module ``classname`` + bare ``name``, not the ``::``
        nodeid it was invoked with, so that mapping is itself lossy).
        Per-test invocation sidesteps the mapping problem entirely at the
        cost of more subprocess calls — acceptable at Stage 1's scale
        (SWE-bench instances typically list single-digit-to-low-tens of
        FAIL_TO_PASS/PASS_TO_PASS tests); revisit if Stage 2's volume
        makes this the actual bottleneck.

        Clears every ``__pycache__`` under *repo_dir* before running and
        disables bytecode writes for the run (``PYTHONDONTWRITEBYTECODE``).
        Found live (0.5.0 Item 7, via ``agentic_repair.py``'s interactive
        apply_patch -> run_tests -> apply_patch loop on one mutable
        checkout): Python's default timestamp-based ``.pyc`` invalidation
        can treat a just-edited source file's stale cached bytecode as
        still valid when two writes land within the same mtime-comparison
        window, silently serving pre-patch behavior to a post-patch test
        run. Stage 1's original design (one `evaluate()` call, checkout ->
        apply once -> test once) never hit this — nothing re-tested a repo
        already known to have been tested before. A wrong pass/fail here
        would corrupt the exact verdict this whole module exists to get
        right, so it is not an acceptable place to save a few subprocess
        calls' worth of recompilation.
        """
        results: dict[str, bool] = {}
        output: dict[str, str] = {}
        timed_out = False
        error = ""
        pytest_env = {"PYTHONDONTWRITEBYTECODE": "1"}
        run_env = dict(os.environ, **pytest_env)
        cmd = [sys.executable, "-m", "pytest", "", "-q", "--no-header"]
        for test_id in test_ids:
            for cache_dir in repo_dir.rglob("__pycache__"):
                shutil.rmtree(cache_dir, ignore_errors=True)
            cmd[3] = test_id
            try:
                if self.isolate:
                    result = sandbox_isolation.run_isolated(
                        cmd, cwd=repo_dir, timeout=self.timeout, env=pytest_env,
                    )
                else:
                    result = subprocess.run(
                        cmd, cwd=repo_dir, capture_output=True, text=True, timeout=self.timeout,
                        env=run_env,
                    )
                results[test_id] = result.returncode == 0
                output[test_id] = result.stdout + result.stderr
            except subprocess.TimeoutExpired as e:
                results[test_id] = False
                # stdout/stderr are populated on TimeoutExpired if anything was
                # captured before the kill (Python's own subprocess contract) --
                # worth keeping, but there's no reliable stack trace to learn
                # from a hung process, which is exactly why callers must treat
                # timed_out as abstain, not ordinary failure feedback.
                out = (e.stdout or "") if isinstance(e.stdout, str) else ""
                err = (e.stderr or "") if isinstance(e.stderr, str) else ""
                output[test_id] = out + err
                timed_out = True
            except Exception as e:
                results[test_id] = False
                output[test_id] = str(e)
                error = str(e)
        return TestRunResult(results=results, timed_out=timed_out, error=error, output=output)

    def evaluate(
        self,
        repo_path: str,
        patch_text: str,
        fail_to_pass: list[str],
        pass_to_pass: list[str],
        base_commit: str | None = None,
        cleanup: bool = True,
    ) -> SandboxEvalResult:
        """Full pipeline: checkout → apply patch → run FAIL_TO_PASS + PASS_TO_PASS."""
        repo_dir = self.checkout_repo(repo_path, base_commit=base_commit)
        sandbox_dir = repo_dir.parent
        try:
            patch_result = self.apply_patch(repo_dir, patch_text)
            if not patch_result.applied:
                return SandboxEvalResult(
                    patch_applied=False,
                    fail_to_pass={t: False for t in fail_to_pass},
                    pass_to_pass={t: False for t in pass_to_pass},
                    timed_out=False,
                    error=patch_result.stderr,
                )

            f2p_run = self.run_tests(repo_dir, fail_to_pass)
            p2p_run = self.run_tests(repo_dir, pass_to_pass)
            return SandboxEvalResult(
                patch_applied=True,
                fail_to_pass=f2p_run.results,
                pass_to_pass=p2p_run.results,
                timed_out=f2p_run.timed_out or p2p_run.timed_out,
                error=f2p_run.error or p2p_run.error,
            )
        finally:
            if cleanup:
                shutil.rmtree(sandbox_dir, ignore_errors=True)


class SandboxVerificationChecker:
    """Additive veto layer for ``FactCheckOracle`` (0.5.0 Item 7) — wraps an
    ``ExecutionSandbox`` so a candidate patch only reaches the user if it
    actually resolves the issue under real test execution.

    Fails CLOSED, unlike ``oracle.py``'s other optional veto layers
    (entailment/relational), which fail OPEN on their own internal
    exceptions — reasonable there, since a classifier bug isn't evidence
    the claim is false. Here, an exception or timeout means "we don't know
    if this patch works," and for code that uncertainty *is* the finding:
    silence is not innocence for a test suite the way it might be for an
    unrelated classifier hiccup on a prose claim. No fallback to any other
    cascade layer either — nothing else in the cascade has an opinion about
    test execution.
    """

    def __init__(self, sandbox: ExecutionSandbox):
        self.sandbox = sandbox

    def is_unverified(
        self,
        repo_path: str,
        patch_text: str,
        fail_to_pass: list[str],
        pass_to_pass: list[str],
        base_commit: str | None = None,
    ) -> bool:
        """True = veto (patch not confirmed to resolve the issue)."""
        try:
            result = self.sandbox.evaluate(
                repo_path=repo_path,
                patch_text=patch_text,
                fail_to_pass=fail_to_pass,
                pass_to_pass=pass_to_pass,
                base_commit=base_commit,
            )
        except Exception:
            return True
        return not result.resolved
