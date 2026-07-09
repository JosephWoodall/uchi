"""Tests for uchi/code_retrieval.py (0.5.0 Item 3.5).

Builds a real, throwaway small package on disk (not mocks) — a database
module, an http module that imports it (deliberate cross-file dependency),
an unrelated math_utils module (distractor), and one file with a syntax
error (skip-and-record behavior) — so chunking, the import graph, the
from-scratch skip-gram embeddings, and the rerank all run against actual
``ast``/filesystem behavior.
"""
from __future__ import annotations

import pytest

from uchi.code_retrieval import CodeIndex, _tokenize_code


DATABASE_PY = '''"""Database connection pooling helpers."""
import sqlite3


class ConnectionPool:
    """Manages pooled sqlite connections to a single database path."""

    def __init__(self, path):
        self.path = path

    def acquire(self):
        """Return a new sqlite3 connection to the pool's database path."""
        return sqlite3.connect(self.path)


def close_all(pool):
    """Close every open connection acquired from the pool."""
    pool.acquire().close()
'''

HTTP_CLIENT_PY = '''"""HTTP client helpers that reuse the shared database connection pool."""
from pkg.database import ConnectionPool


def fetch_and_log(url, pool: ConnectionPool):
    """Fetch a url and log the request using the shared connection pool."""
    conn = pool.acquire()
    conn.execute("insert into log(url) values (?)", (url,))
    return url


def parse_response(payload):
    """Parse an http response payload into a dict of headers."""
    headers = {}
    for line in payload.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()
    return headers
'''

MATH_UTILS_PY = '''"""Arithmetic helper functions unrelated to networking or storage."""


def add(a, b):
    """Return the sum of a and b."""
    return a + b


def multiply(a, b):
    """Return the product of a and b."""
    return a * b
'''

BROKEN_PY = "def bad(:\n    pass\n"


@pytest.fixture
def small_repo(tmp_path):
    repo = tmp_path / "repo"
    pkg = repo / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "database.py").write_text(DATABASE_PY)
    (pkg / "http_client.py").write_text(HTTP_CLIENT_PY)
    (pkg / "math_utils.py").write_text(MATH_UTILS_PY)
    (pkg / "broken.py").write_text(BROKEN_PY)
    return repo


@pytest.fixture
def index(small_repo):
    return CodeIndex.build(small_repo, dim=32, epochs=10, seed=0)


def _by_qualname(index, qualname, file=None):
    for c in index.chunks:
        if c.qualname == qualname and (file is None or c.file == file):
            return c
    raise AssertionError(f"no chunk with qualname={qualname!r} file={file!r}")


# ── tokenizer ────────────────────────────────────────────────────────────

def test_tokenize_code_splits_identifiers_and_drops_keywords():
    tokens = _tokenize_code("def acquire_connection(self): return ConnectionPool")
    assert "def" not in tokens
    assert "self" not in tokens
    assert "return" not in tokens
    assert "acquire" in tokens
    assert "connection" in tokens
    assert "pool" in tokens


# ── chunking ─────────────────────────────────────────────────────────────

def test_chunking_extracts_functions_methods_classes_and_module_chunk(index):
    qualnames = {(c.file, c.qualname, c.kind) for c in index.chunks}
    assert ("pkg/database.py", "ConnectionPool", "class") in qualnames
    assert ("pkg/database.py", "ConnectionPool.__init__", "method") in qualnames
    assert ("pkg/database.py", "ConnectionPool.acquire", "method") in qualnames
    assert ("pkg/database.py", "close_all", "function") in qualnames
    assert ("pkg/http_client.py", "fetch_and_log", "function") in qualnames
    assert ("pkg/http_client.py", "parse_response", "function") in qualnames
    assert ("pkg/math_utils.py", "add", "function") in qualnames
    assert ("pkg/math_utils.py", "multiply", "function") in qualnames
    # module-level chunk: the docstring + import line, not covered by any def/class
    module_chunks = [c for c in index.chunks if c.file == "pkg/database.py" and c.kind == "module"]
    assert len(module_chunks) == 1
    assert "sqlite3" in module_chunks[0].source


def test_broken_file_is_skipped_not_crashed(index):
    assert "pkg/broken.py" in index.skipped_files
    assert all(c.file != "pkg/broken.py" for c in index.chunks)


def test_chunk_ids_are_unique_and_sequential(index):
    ids = [c.chunk_id for c in index.chunks]
    assert ids == list(range(len(index.chunks)))


# ── import graph ─────────────────────────────────────────────────────────

def test_import_graph_cross_file_edges(index):
    assert "pkg/database.py" in index.imports_of("pkg/http_client.py")
    assert "pkg/http_client.py" in index.imported_by("pkg/database.py")
    # unrelated file has no edges either direction
    assert index.imports_of("pkg/math_utils.py") == set()
    assert index.imported_by("pkg/math_utils.py") == set()


def test_neighbor_chunks_cross_file_navigation(index):
    fetch = _by_qualname(index, "fetch_and_log", file="pkg/http_client.py")
    neighbors = index.neighbor_chunks(fetch.chunk_id, direction="imports")
    neighbor_files = {c.file for c in neighbors}
    assert neighbor_files == {"pkg/database.py"}
    neighbor_qualnames = {c.qualname for c in neighbors}
    assert "ConnectionPool" in neighbor_qualnames
    assert "ConnectionPool.acquire" in neighbor_qualnames


def test_neighbor_chunks_invalid_id_raises(index):
    with pytest.raises(IndexError):
        index.neighbor_chunks(999999)


# ── embeddings / semantic retrieval ───────────────────────────────────────

def test_topically_related_chunk_has_higher_raw_cosine_than_unrelated(index):
    acquire = _by_qualname(index, "ConnectionPool.acquire", file="pkg/database.py")
    parse_resp = _by_qualname(index, "parse_response", file="pkg/http_client.py")
    qv = index._vec("database connection pool")
    assert qv is not None
    sim_acquire = float(index._P[acquire.chunk_id] @ qv)
    sim_unrelated = float(index._P[parse_resp.chunk_id] @ qv)
    assert sim_acquire > sim_unrelated


def test_retrieve_returns_hybrid_ranked_results(index):
    results = index.retrieve("database connection pool acquire", k=3)
    assert results
    top_chunk, top_score = results[0]
    assert isinstance(top_score, float)
    # lexical overlap ("connection", "pool", "acquire", "database" all appear
    # verbatim in ConnectionPool/acquire) should put a database.py chunk on top.
    assert top_chunk.file == "pkg/database.py"


def test_retrieve_empty_query_or_empty_index_returns_nothing(index):
    empty_index = CodeIndex(dim=8)
    assert empty_index.retrieve("anything") == []
    # a query with no in-vocabulary tokens at all
    assert index.retrieve("!!! ??? ...") == []


# ── graph-weighted rerank ─────────────────────────────────────────────────

def test_graph_context_boost_prioritizes_imported_file(index):
    fetch = _by_qualname(index, "fetch_and_log", file="pkg/http_client.py")
    n = len(index)

    without_graph = index.retrieve(
        "helper functions", k=n, context_chunk_ids=[fetch.chunk_id],
        lex_weight=0.5, graph_weight=0.0,
    )
    with_graph = index.retrieve(
        "helper functions", k=n, context_chunk_ids=[fetch.chunk_id],
        lex_weight=0.5, graph_weight=10.0,
    )

    def best_rank(results, file):
        return min(i for i, (c, _) in enumerate(results) if c.file == file)

    rank_before = best_rank(without_graph, "pkg/database.py")
    rank_after = best_rank(with_graph, "pkg/database.py")
    assert rank_after < rank_before
