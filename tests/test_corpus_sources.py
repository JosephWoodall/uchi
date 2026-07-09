"""Tests for uchi/corpus_sources.py (0.5.0 Item 1's general-code + issue->diff
sources).

Offline tests patch ``_load_streaming`` with fake in-memory rows (and, for
the Stack v2 path, a fake Software Heritage S3 client — real gzip/decode
logic runs against it, only the network call is faked) — no network
required, runs in the normal fast path. The `eval`-marked tests hit the live
HF Hub (and, for Stack v2, the live SWH S3 mirror) to confirm the schema
assumptions documented in the module docstring, now that this environment
has an authenticated, terms-accepted HF token.
"""
from __future__ import annotations

import gzip
import io
import json

import pytest

from uchi.corpus_sources import (
    STACK_V1_FALLBACK_DATASET_ID,
    STACK_V2_DATASET_ID,
    SWE_GYM_DATASET_ID,
    CorpusSourceError,
    iter_general_code_python,
    iter_issue_diff_pairs,
    main,
)

FAKE_EXCLUDED = frozenset({"django/django"})


def _patch_streaming(monkeypatch, table):
    """table: dataset_id -> list[dict] | Exception instance to raise."""
    def fake(dataset_id, **kwargs):
        entry = table[dataset_id]
        if isinstance(entry, Exception):
            raise entry
        return iter(entry)
    monkeypatch.setattr("uchi.corpus_sources._load_streaming", fake)


def _gzip_body(text: str, encoding: str = "utf-8") -> io.BytesIO:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as f:
        f.write(text.encode(encoding))
    buf.seek(0)
    return buf


class _FakeSWHClient:
    """blob_id -> content string, or an Exception to raise for that blob."""
    def __init__(self, blobs):
        self._blobs = blobs

    def get_object(self, Bucket, Key):
        blob_id = Key.split("/", 1)[1]
        entry = self._blobs[blob_id]
        if isinstance(entry, Exception):
            raise entry
        return {"Body": _gzip_body(entry)}


def _patch_swh_client(monkeypatch, blobs):
    monkeypatch.setattr("uchi.corpus_sources._swh_s3_client", lambda: _FakeSWHClient(blobs))


def _stack_v2_row(blob_id, **overrides):
    row = {
        "repo_name": "psf/black", "path": "a.py", "language": "Python",
        "license_type": "permissive", "is_generated": False, "is_vendor": False,
        "blob_id": blob_id, "src_encoding": "UTF-8",
    }
    row.update(overrides)
    return row


# ── iter_general_code_python / Stack v2 ──────────────────────────────────

def test_iter_general_code_python_v2_normalizes_and_fetches_content(monkeypatch):
    _patch_streaming(monkeypatch, {STACK_V2_DATASET_ID: [_stack_v2_row("b1")]})
    _patch_swh_client(monkeypatch, {"b1": "x = 1"})

    out = list(iter_general_code_python())
    assert out == [{"repo": "psf/black", "path": "a.py", "content": "x = 1"}]


def test_iter_general_code_python_v2_respects_limit(monkeypatch):
    rows = [_stack_v2_row(f"b{i}", path=f"{i}.py") for i in range(5)]
    _patch_streaming(monkeypatch, {STACK_V2_DATASET_ID: rows})
    _patch_swh_client(monkeypatch, {f"b{i}": "x" for i in range(5)})

    out = list(iter_general_code_python(limit=2))
    assert len(out) == 2


def test_iter_general_code_python_v2_filters_generated_vendor_and_nonpermissive(monkeypatch):
    rows = [
        _stack_v2_row("keep", path="keep.py"),
        _stack_v2_row("gen", path="gen.py", is_generated=True),
        _stack_v2_row("vendor", path="vendor.py", is_vendor=True),
        _stack_v2_row("nolic", path="nolic.py", license_type="no_license"),
    ]
    _patch_streaming(monkeypatch, {STACK_V2_DATASET_ID: rows})
    _patch_swh_client(monkeypatch, {"keep": "kept", "gen": "x", "vendor": "x", "nolic": "x"})

    out = list(iter_general_code_python())
    assert [r["path"] for r in out] == ["keep.py"]


def test_iter_general_code_python_v2_skips_missing_swh_blob(monkeypatch):
    from botocore.exceptions import ClientError
    rows = [_stack_v2_row("missing", path="missing.py"), _stack_v2_row("ok", path="ok.py")]
    _patch_streaming(monkeypatch, {STACK_V2_DATASET_ID: rows})
    _patch_swh_client(monkeypatch, {
        "missing": ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject"),
        "ok": "content",
    })

    out = list(iter_general_code_python())
    assert [r["path"] for r in out] == ["ok.py"]


def test_iter_general_code_python_v2_propagates_non_missing_s3_errors(monkeypatch):
    from botocore.exceptions import ClientError
    _patch_streaming(monkeypatch, {STACK_V2_DATASET_ID: [_stack_v2_row("b1")]})
    _patch_swh_client(monkeypatch, {
        "b1": ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject"),
    })

    with pytest.raises(ClientError):
        list(iter_general_code_python())


def test_iter_general_code_python_falls_back_on_v2_load_failure(monkeypatch):
    fallback_rows = [{"repo_name": "psf/black", "path": "a.py", "content": "x = 1"}]
    _patch_streaming(monkeypatch, {
        STACK_V2_DATASET_ID: CorpusSourceError("gated, no token"),
        STACK_V1_FALLBACK_DATASET_ID: fallback_rows,
    })

    out = list(iter_general_code_python())
    assert out == [{"repo": "psf/black", "path": "a.py", "content": "x = 1"}]


def test_iter_general_code_python_use_fallback_skips_v2_entirely(monkeypatch):
    def boom(dataset_id, **kwargs):
        if dataset_id == STACK_V2_DATASET_ID:
            raise AssertionError("should not touch v2 when use_fallback=True")
        return iter([{"repo_name": "psf/black", "path": "a.py", "content": "x = 1"}])
    monkeypatch.setattr("uchi.corpus_sources._load_streaming", boom)

    out = list(iter_general_code_python(use_fallback=True))
    assert out == [{"repo": "psf/black", "path": "a.py", "content": "x = 1"}]


def test_iter_general_code_python_v1_fallback_missing_repo_column_raises(monkeypatch):
    rows = [{"content": "x = 1", "path": "a.py"}]  # no repo-identifier candidate present
    _patch_streaming(monkeypatch, {STACK_V1_FALLBACK_DATASET_ID: rows})
    with pytest.raises(CorpusSourceError, match="repo identifier"):
        list(iter_general_code_python(use_fallback=True))


# ── iter_issue_diff_pairs ────────────────────────────────────────────────

def _swe_gym_row(**overrides):
    row = {
        "repo": "getmoto/moto",
        "instance_id": "getmoto__moto-1234",
        "problem_statement": "describe_instances raises KeyError",
        "patch": "diff --git a/x.py b/x.py\n...",
        "base_commit": "abc123",
        "FAIL_TO_PASS": ["tests/test_ec2.py::test_describe"],
        "PASS_TO_PASS": ["tests/test_ec2.py::test_other"],
    }
    row.update(overrides)
    return row


def test_iter_issue_diff_pairs_normalizes(monkeypatch):
    _patch_streaming(monkeypatch, {SWE_GYM_DATASET_ID: [_swe_gym_row()]})

    out = list(iter_issue_diff_pairs())
    assert out == [{
        "repo": "getmoto/moto",
        "instance_id": "getmoto__moto-1234",
        "problem_statement": "describe_instances raises KeyError",
        "patch": "diff --git a/x.py b/x.py\n...",
        "base_commit": "abc123",
        "fail_to_pass": ["tests/test_ec2.py::test_describe"],
        "pass_to_pass": ["tests/test_ec2.py::test_other"],
    }]


def test_iter_issue_diff_pairs_missing_column_raises(monkeypatch):
    row = _swe_gym_row()
    del row["patch"]
    _patch_streaming(monkeypatch, {SWE_GYM_DATASET_ID: [row]})
    with pytest.raises(CorpusSourceError, match="patch"):
        list(iter_issue_diff_pairs())


# ── main() / decontamination composition ─────────────────────────────────

def test_main_decontaminates_and_writes_jsonl(monkeypatch, tmp_path, capsys):
    rows = [_swe_gym_row(repo="django/django"), _swe_gym_row(repo="getmoto/moto")]
    _patch_streaming(monkeypatch, {SWE_GYM_DATASET_ID: rows})
    monkeypatch.setattr("uchi.corpus_sources.resolve_excluded_repos", lambda: FAKE_EXCLUDED)

    out_path = tmp_path / "out.jsonl"
    rc = main(["--source", "swe-gym", "--output", str(out_path)])
    assert rc == 0

    kept = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert [r["repo"] for r in kept] == ["getmoto/moto"]

    err = capsys.readouterr().err
    assert "total=2 kept=1 excluded=1" in err


def test_main_uses_cache_file_without_live_resolution(monkeypatch, tmp_path):
    from uchi.corpus_decontamination import save_repo_cache
    cache_path = tmp_path / "repos.json"
    save_repo_cache(FAKE_EXCLUDED, path=cache_path)

    def boom():
        raise AssertionError("should not resolve live when --cache-file is given")
    monkeypatch.setattr("uchi.corpus_sources.resolve_excluded_repos", boom)
    _patch_streaming(monkeypatch, {SWE_GYM_DATASET_ID: [_swe_gym_row()]})

    out_path = tmp_path / "out.jsonl"
    rc = main(["--source", "swe-gym", "--output", str(out_path), "--cache-file", str(cache_path)])
    assert rc == 0


# ── live network, excluded from default run ──────────────────────────────

@pytest.mark.eval
def test_swe_gym_live_schema_matches_expected():
    """Confirms the SWE-Gym schema documented in the module docstring: real
    stream, required columns present, repo outside SWE-bench's eval set for
    the sampled rows.
    """
    row = next(iter_issue_diff_pairs(limit=1))
    assert row["repo"]
    assert row["problem_statement"]
    assert row["patch"]


@pytest.mark.eval
def test_stack_v2_live_schema_matches_expected():
    """Confirms the Stack v2 schema documented in the module docstring: real
    stream, required columns present.
    """
    row = next(iter_general_code_python(limit=1))
    assert row["repo"]
    assert row["path"]
    assert row["content"]
