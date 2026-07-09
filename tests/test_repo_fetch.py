"""Tests for uchi/repo_fetch.py (0.5.0 Item 9's real-execution harness).

Offline tests point `ensure_local_clone` at a real local git repo used as
the "remote" (`remote_url` override) — real git fetch-by-sha mechanics,
just against a local path instead of github.com, so no network is needed
for the default run. One `eval`-marked test confirms the same fetch-by-sha
approach against real github.com.
"""
import subprocess

import pytest

from uchi.repo_fetch import RepoFetchError, ensure_local_clone


def _run(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, f"{cmd} failed: {result.stderr}"
    return result


@pytest.fixture
def local_remote(tmp_path):
    """A real git repo with two commits, used as a stand-in 'remote' —
    returns (path, commit1_sha, commit2_sha) so tests can fetch an
    arbitrary historical commit, not just the tip.
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    _run(["git", "init"], cwd=origin)
    _run(["git", "config", "user.email", "test@example.com"], cwd=origin)
    _run(["git", "config", "user.name", "Test"], cwd=origin)

    (origin / "f.txt").write_text("v1\n")
    _run(["git", "add", "-A"], cwd=origin)
    _run(["git", "commit", "-m", "v1"], cwd=origin)
    c1 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=origin, capture_output=True, text=True).stdout.strip()

    (origin / "f.txt").write_text("v2\n")
    _run(["git", "add", "-A"], cwd=origin)
    _run(["git", "commit", "-m", "v2"], cwd=origin)
    c2 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=origin, capture_output=True, text=True).stdout.strip()

    return str(origin), c1, c2


def test_ensure_local_clone_fetches_the_requested_historical_commit(local_remote, tmp_path):
    origin, c1, c2 = local_remote
    cache_root = str(tmp_path / "cache")

    repo_path = ensure_local_clone("acme/widget", c1, root=cache_root, remote_url=origin)

    from pathlib import Path
    assert Path(repo_path, "f.txt").read_text() == "v1\n"  # c1, not c2's tip content


def test_ensure_local_clone_caches_by_repo_and_commit(local_remote, tmp_path, monkeypatch):
    origin, c1, _ = local_remote
    cache_root = str(tmp_path / "cache")

    first = ensure_local_clone("acme/widget", c1, root=cache_root, remote_url=origin)

    def boom(*a, **kw):
        raise AssertionError("should not re-fetch a cached (repo, commit) pair")
    monkeypatch.setattr("uchi.repo_fetch._run", boom)

    second = ensure_local_clone("acme/widget", c1, root=cache_root, remote_url=origin)
    assert first == second


def test_ensure_local_clone_different_commits_get_different_cache_entries(local_remote, tmp_path):
    origin, c1, c2 = local_remote
    cache_root = str(tmp_path / "cache")

    path1 = ensure_local_clone("acme/widget", c1, root=cache_root, remote_url=origin)
    path2 = ensure_local_clone("acme/widget", c2, root=cache_root, remote_url=origin)

    from pathlib import Path
    assert path1 != path2
    assert Path(path1, "f.txt").read_text() == "v1\n"
    assert Path(path2, "f.txt").read_text() == "v2\n"


def test_ensure_local_clone_bad_remote_raises_and_cleans_up(tmp_path):
    cache_root = str(tmp_path / "cache")
    commit = "deadbeef" * 5
    with pytest.raises(RepoFetchError):
        ensure_local_clone(
            "acme/nope", commit, root=cache_root,
            remote_url=str(tmp_path / "does_not_exist"),
        )
    from pathlib import Path
    # The exact (repo, commit) cache entry must not be left behind
    # masquerading as a valid clone -- a future ensure_local_clone call for
    # this same pair must retry the fetch, not trust a half-written dir.
    assert not (Path(cache_root) / "repo_cache" / "acme__nope" / commit).exists()


def test_ensure_local_clone_bad_commit_raises(local_remote, tmp_path):
    origin, _, _ = local_remote
    cache_root = str(tmp_path / "cache")
    with pytest.raises(RepoFetchError):
        ensure_local_clone("acme/widget", "f" * 40, root=cache_root, remote_url=origin)


@pytest.mark.eval
def test_ensure_local_clone_live_github(tmp_path):
    """Confirms fetch-by-sha works the same way against real github.com,
    fetching the repo's *first* commit specifically (not just the tip) --
    the exact scenario a historical SWE-bench base_commit needs.
    """
    cache_root = str(tmp_path / "cache")
    repo_path = ensure_local_clone(
        "octocat/Hello-World",
        "553c2077f0edc3d5dc5d17262f6aa498e69d6f8e",  # the repo's first commit
        root=cache_root,
    )
    from pathlib import Path
    assert (Path(repo_path) / "README").exists()
