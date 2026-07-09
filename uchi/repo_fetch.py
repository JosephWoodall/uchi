"""repo_fetch.py — on-demand, cached local clones for SWE-bench-shaped
evaluation (0.5.0 Item 9's real-execution harness).

Deliberately a separate module from ``execution_sandbox.py``: that module's
own docstring scopes "fetching from a remote" out of its job explicitly
(``checkout_repo``'s docstring: "*repo_path* must already be a local clone
... fetching from a remote is a corpus-ingestion concern, not the sandbox's
job"). This is that concern's actual home.

Uses a **targeted shallow fetch by commit SHA**
(``git fetch --depth 1 origin <sha>``), not a full clone. GitHub allows
fetching an arbitrary reachable commit directly
(``uploadpack.allowReachableSHA1InWant``, enabled on github.com) — verified
against a local path remote in this module's tests, same fetch-by-sha
mechanics apply to a real GitHub remote. This matters at this benchmark's
actual scale: several SWE-bench repos (django, sympy, astropy, ...) carry
gigabyte-scale full histories that are entirely irrelevant weight for
grading one commit's FAIL_TO_PASS/PASS_TO_PASS state.

Cached per ``(repo, commit)`` pair under ``<root>/repo_cache/`` — SWE-bench
instances from the same repo often cluster around nearby commits but not
identical ones, so caching is keyed by the exact pair, never just the repo
name (that would risk silently grading one instance's issue against
another instance's checkout).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .workspace import DEFAULT_ROOT

REPO_CACHE_SUBDIR = "repo_cache"


class RepoFetchError(RuntimeError):
    """Raised when a repo can't be fetched at the requested commit."""


def _run(cmd: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise RepoFetchError(f"{' '.join(cmd)} timed out after {timeout}s") from e
    if result.returncode != 0:
        raise RepoFetchError(f"{' '.join(cmd)} failed in {cwd}: {result.stderr.strip()}")
    return result


def ensure_local_clone(
    repo: str,
    commit: str,
    root: str = DEFAULT_ROOT,
    remote_url: str | None = None,
    fetch_timeout: float = 300.0,
) -> str:
    """Return a local path containing *repo* checked out at *commit*,
    fetching and caching it if not already present.

    *repo* is ``"owner/name"``; *remote_url* defaults to the matching
    ``https://github.com/{repo}.git`` but can be overridden (used by this
    module's own offline tests, which point it at a local path "remote"
    instead of hitting real GitHub). A pre-existing cache entry is trusted
    as-is and returned immediately without re-fetching — the cache key
    already pins the exact commit, so there is nothing to become stale.
    """
    cache_dir = Path(root) / REPO_CACHE_SUBDIR / repo.replace("/", "__") / commit
    if (cache_dir / ".git").exists():
        return str(cache_dir)

    url = remote_url or f"https://github.com/{repo}.git"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        _run(["git", "init"], cwd=cache_dir, timeout=30.0)
        _run(["git", "remote", "add", "origin", url], cwd=cache_dir, timeout=30.0)
        _run(["git", "fetch", "--depth", "1", "origin", commit], cwd=cache_dir, timeout=fetch_timeout)
        _run(["git", "checkout", "FETCH_HEAD"], cwd=cache_dir, timeout=30.0)
    except RepoFetchError:
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)
        raise
    return str(cache_dir)
