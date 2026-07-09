"""Tests for uchi/corpus_decontamination.py (0.5.0 Item 1's decontamination gate).

Offline tests exercise the filter logic, cache round-trip, and CLI against an
in-memory/fake repo set — no network required, runs in the normal fast path.
The `eval`-marked tests hit the live HF Hub to resolve the actual SWE-bench
repo list; excluded from the default run (`pytest -m eval` to include them).
"""
from __future__ import annotations

import json

import pytest

from uchi.corpus_decontamination import (
    DEFAULT_SWE_BENCH_SPECS,
    DatasetSpec,
    FilterSummary,
    filter_corpus,
    is_excluded,
    load_repo_cache,
    main,
    resolve_excluded_repos,
    save_repo_cache,
)

FAKE_EXCLUDED = frozenset({"django/django", "sympy/sympy"})


def test_is_excluded_matches_repo_level():
    assert is_excluded({"repo": "django/django", "instance_id": "x-1"}, FAKE_EXCLUDED)
    # Same repo, a wholly different issue than any SWE-bench instance —
    # still excluded, because the gate is repo-level, not instance-level.
    assert is_excluded({"repo": "django/django", "instance_id": "never-seen-by-swebench"}, FAKE_EXCLUDED)


def test_is_excluded_allows_clean_repo():
    assert not is_excluded({"repo": "psf/black", "instance_id": "y-1"}, FAKE_EXCLUDED)


def test_filter_corpus_splits_and_counts():
    records = [
        {"repo": "django/django", "id": 1},
        {"repo": "psf/black", "id": 2},
        {"repo": "sympy/sympy", "id": 3},
        {"repo": "psf/black", "id": 4},
        {"repo": "django/django", "id": 5},
    ]
    kept, summary = filter_corpus(records, FAKE_EXCLUDED)

    assert [r["id"] for r in kept] == [2, 4]
    assert summary.total == 5
    assert summary.kept == 2
    assert summary.excluded == 3
    assert summary.excluded_by_repo == {"django/django": 2, "sympy/sympy": 1}


def test_filter_corpus_empty_input():
    kept, summary = filter_corpus([], FAKE_EXCLUDED)
    assert kept == []
    assert summary.total == 0
    assert summary.kept == 0
    assert summary.excluded == 0


def test_filter_summary_excluded_is_derived():
    summary = FilterSummary(total=10, kept=7, excluded_by_repo={"a/a": 3})
    assert summary.excluded == 3


def test_default_specs_cover_full_lite_verified():
    ids = {spec.dataset_id for spec in DEFAULT_SWE_BENCH_SPECS}
    assert ids == {
        "SWE-bench/SWE-bench",
        "SWE-bench/SWE-bench_Lite",
        "SWE-bench/SWE-bench_Verified",
    }
    assert all(spec.split == "test" for spec in DEFAULT_SWE_BENCH_SPECS)


def test_cache_round_trip(tmp_path):
    cache_path = tmp_path / "repos.json"
    saved_path = save_repo_cache(FAKE_EXCLUDED, path=cache_path)
    assert saved_path == cache_path

    loaded = load_repo_cache(cache_path)
    assert loaded == FAKE_EXCLUDED

    payload = json.loads(cache_path.read_text())
    assert payload["repos"] == sorted(FAKE_EXCLUDED)
    assert "resolved_at" in payload
    assert payload["specs"]


def test_cache_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_repo_cache(tmp_path / "does_not_exist.json")


def test_cli_end_to_end_uses_existing_cache_without_network(tmp_path, capsys):
    cache_path = tmp_path / "repos.json"
    save_repo_cache(FAKE_EXCLUDED, path=cache_path)

    in_path = tmp_path / "corpus.jsonl"
    records = [
        {"repo": "django/django", "id": 1},
        {"repo": "psf/black", "id": 2},
        {"repo": "sympy/sympy", "id": 3},
    ]
    in_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    out_path = tmp_path / "filtered.jsonl"
    rc = main([
        "--input", str(in_path),
        "--output", str(out_path),
        "--cache-file", str(cache_path),
    ])
    assert rc == 0

    out_records = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert out_records == [{"repo": "psf/black", "id": 2}]

    err = capsys.readouterr().err
    assert "total=3 kept=1 excluded=2" in err
    assert "django/django: 1" in err
    assert "sympy/sympy: 1" in err


def test_cli_rejects_missing_input_gracefully(tmp_path):
    cache_path = tmp_path / "repos.json"
    save_repo_cache(FAKE_EXCLUDED, path=cache_path)
    with pytest.raises(FileNotFoundError):
        main([
            "--input", str(tmp_path / "nope.jsonl"),
            "--cache-file", str(cache_path),
        ])


@pytest.mark.eval
def test_resolve_excluded_repos_live_matches_known_swebench_repos():
    """Live network test: resolve the actual SWE-bench repo set from the HF
    Hub and check it contains the well-known repos, at minimum.
    """
    repos = resolve_excluded_repos()
    expected_subset = {
        "django/django",
        "sympy/sympy",
        "astropy/astropy",
        "matplotlib/matplotlib",
        "pytest-dev/pytest",
        "psf/requests",
        "pallets/flask",
        "scikit-learn/scikit-learn",
    }
    assert expected_subset.issubset(repos)


@pytest.mark.eval
def test_lite_and_verified_are_subsets_of_full_repo_population():
    """Confirms Lite/Verified don't introduce repos outside the full test
    split's population — documented as fact in the module docstring, this
    test guards against a future SWE-bench revision silently changing that.
    """
    full_repos = resolve_excluded_repos([DatasetSpec("SWE-bench/SWE-bench", "test")])
    lite_repos = resolve_excluded_repos([DatasetSpec("SWE-bench/SWE-bench_Lite", "test")])
    verified_repos = resolve_excluded_repos([DatasetSpec("SWE-bench/SWE-bench_Verified", "test")])

    assert lite_repos.issubset(full_repos)
    assert verified_repos.issubset(full_repos)
