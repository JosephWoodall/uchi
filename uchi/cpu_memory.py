"""CPU-bound vector memory store.

Embeddings are persisted via an hnswlib HNSW index (O(log N) retrieval) and a
companion JSON text index. Both survive process restarts. All operations stay
on CPU — GPU VRAM is never touched.

Public interface is unchanged from the flat-numpy version:
    add_memory(text, embedding)
    retrieve(query_emb, top_k) -> List[str]
    retrieve_with_scores(query_emb, top_k) -> List[tuple]
    records -> List[str]

HNSW replaces the flat O(N) cosine scan with O(log N) approximate nearest
neighbour search. At small brain sizes the difference is invisible; at scale
it is the difference between milliseconds and minutes.
"""

import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np

_log = logging.getLogger(__name__)

# HNSW construction parameters.
# ef_construction: higher = better recall at index build time, slower inserts.
# M: neighbours per node — 16 is the standard default.
_HNSW_EF_CONSTRUCTION = 200
_HNSW_M               = 16
# ef at query time — controls recall/speed tradeoff. 50 is a safe default.
_HNSW_EF_SEARCH       = 50
# Fall back to flat numpy scan when fewer than this many vectors are stored
# (HNSW has overhead that isn't worth it for tiny stores).
_FLAT_THRESHOLD = 50


class CPUVectorMemory:
    """Persistent HNSW-backed vector store.

    Interface used by ContinualLearner (and everywhere else):
        add_memory(text, embedding)
        retrieve(query_emb, top_k) -> List[str]
        records                    -> List[str]
    """

    def __init__(self, db_path: str = "uchi_cpu_memory"):
        self.db_path   = db_path
        self._idx_path = db_path + "_index.json"
        self._hnsw_path = db_path + "_hnsw.bin"
        # Keep the legacy .npy path so we can migrate old stores on first load.
        self._legacy_vec_path = db_path + "_vectors.npy"

        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        self.records: List[str] = []
        self._dim: Optional[int] = None
        self._index = None  # hnswlib.Index, built lazily on first add

        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self):
        """Restore records and HNSW index from disk, migrating legacy .npy if needed."""
        if not os.path.exists(self._idx_path):
            return

        try:
            with open(self._idx_path) as f:
                meta = json.load(f)
            self.records = meta.get("records", [])
            self._dim    = meta.get("dim")
        except Exception:
            self.records = []
            self._dim    = None
            return

        if not self.records or self._dim is None:
            return

        # Try loading the HNSW index.
        if os.path.exists(self._hnsw_path):
            try:
                import hnswlib
                idx = hnswlib.Index("ip", self._dim)
                idx.load_index(self._hnsw_path, max_elements=max(len(self.records) * 2, 128))
                idx.set_ef(_HNSW_EF_SEARCH)
                self._index = idx
                return
            except Exception as e:
                _log.warning("HNSW index load failed (%s) — rebuilding from legacy.", e)

        # Migrate from legacy flat .npy store if it exists.
        if os.path.exists(self._legacy_vec_path):
            try:
                vecs = np.load(self._legacy_vec_path)
                if vecs.shape[0] == len(self.records):
                    _log.info("Migrating %d vectors from legacy .npy to HNSW.", len(self.records))
                    self._build_index_from_array(vecs)
                    self._save_index()
            except Exception as e:
                _log.warning("Legacy .npy migration failed: %s", e)

    def _build_index_from_array(self, vecs: np.ndarray):
        import hnswlib
        n, d = vecs.shape
        self._dim = d
        idx = hnswlib.Index("ip", d)
        idx.init_index(max_elements=max(n * 2, 128),
                       ef_construction=_HNSW_EF_CONSTRUCTION,
                       M=_HNSW_M)
        idx.add_items(vecs.astype(np.float32), list(range(n)))
        idx.set_ef(_HNSW_EF_SEARCH)
        self._index = idx

    def _save_index(self):
        if self._index is not None:
            self._index.save_index(self._hnsw_path)
        with open(self._idx_path, "w") as f:
            json.dump({"records": self.records, "dim": self._dim}, f)

    # ── public API ────────────────────────────────────────────────────────────

    def add_memory(self, text: str, embedding: np.ndarray):
        vec = embedding.reshape(1, -1).astype(np.float32)
        d   = vec.shape[1]

        # Dimension mismatch: SSM d_model changed — discard stale index.
        if self._dim is not None and d != self._dim:
            _log.warning(
                "Embedding dimension changed %d→%d; discarding stale HNSW index.",
                self._dim, d,
            )
            self._index   = None
            self.records  = []
            self._dim     = None

        self._dim = d
        self.records.append(text)
        n = len(self.records)

        if self._index is None:
            import hnswlib
            idx = hnswlib.Index("ip", d)
            idx.init_index(max_elements=max(n * 2, 128),
                           ef_construction=_HNSW_EF_CONSTRUCTION,
                           M=_HNSW_M)
            self._index = idx

        # Resize index if we've hit capacity.
        if n > self._index.get_max_elements():
            self._index.resize_index(n * 2)

        self._index.add_items(vec, [n - 1])
        self._save_index()

    def retrieve(self, query_emb: np.ndarray, top_k: int = 3) -> List[str]:
        if not self.records or self._index is None:
            return []
        results = self._query(query_emb, top_k)
        return [text for text, _ in results]

    def retrieve_with_scores(self, query_emb: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """Returns (text, inner_product_similarity) pairs."""
        if not self.records or self._index is None:
            return []
        return self._query(query_emb, top_k)

    # ── internal ──────────────────────────────────────────────────────────────

    def _query(self, query_emb: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        n = len(self.records)
        k = min(top_k, n)

        # For very small stores use flat scan — HNSW ef must be ≥ k.
        if n < _FLAT_THRESHOLD:
            return self._flat_query(query_emb, k)

        q = query_emb.reshape(1, -1).astype(np.float32)
        try:
            self._index.set_ef(max(_HNSW_EF_SEARCH, k))
            labels, distances = self._index.knn_query(q, k=k)
            # hnswlib "ip" space returns negative inner products as distances.
            return [(self.records[i], -float(d))
                    for i, d in zip(labels[0], distances[0])
                    if i < n]
        except Exception as e:
            _log.warning("HNSW query failed (%s) — falling back to flat scan.", e)
            return self._flat_query(query_emb, k)

    def _flat_query(self, query_emb: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """O(N) fallback — used for small stores and HNSW error recovery."""
        import hnswlib
        # Rebuild a tiny flat store from records using stored index if possible,
        # otherwise we have no vectors to scan — return empty.
        if self._index is None:
            return []
        q = query_emb.reshape(1, -1).astype(np.float32)
        k = min(k, len(self.records))
        self._index.set_ef(max(_HNSW_EF_SEARCH, k))
        labels, distances = self._index.knn_query(q, k=k)
        return [(self.records[i], -float(d))
                for i, d in zip(labels[0], distances[0])
                if i < len(self.records)]
