"""
semantic_index.py
=================
FAISS-backed semantic k-NN index for trie smoothing.

During dreaming, every successful rollout registers its per-step SSM state
vectors and continuation tokens.  At inference, when the lexical trie has zero
children for a query context (cold-start or OOV), this index provides a
synthetic probability distribution over plausible next tokens by retrieving
the k-nearest historical states and applying temperature-scaled softmax over
their cosine similarities.

FAISS (faiss-cpu) is used when available; falls back to a numpy dot-product
scan when FAISS is absent.  Both paths share the same public API.
"""
from __future__ import annotations

import json
import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

_log = logging.getLogger(__name__)

_DEFAULT_PATH = ".uchi/semantic_index"
_D_MODEL      = 256
_K            = 5       # nearest neighbours to retrieve
_TAU          = 0.1     # temperature for softmax over retrieved similarities


class SemanticTrieIndex:
    """
    Maps SSM hidden states → continuation tokens via approximate nearest-neighbour.

    .add(state_vec, token)  — register one (state, next_token) pair
    .query(state_vec)       — return synthetic P(next_token|state) distribution
    .save(path) / .load(path)
    """

    def __init__(self, d_model: int = _D_MODEL, k: int = _K, tau: float = _TAU):
        self.d_model = d_model
        self.k       = k
        self.tau     = tau

        self._tokens: list[str]         = []
        self._vecs:   list[np.ndarray]  = []   # for numpy fallback
        self._n:      int               = 0

        self._faiss_index = None
        self._use_faiss   = False
        self._init_index()

    # ── index construction ────────────────────────────────────────────────────

    def _init_index(self) -> None:
        try:
            import faiss  # type: ignore
            self._faiss_index = faiss.IndexFlatIP(self.d_model)
            self._use_faiss   = True
            _log.debug("SemanticTrieIndex: using FAISS IndexFlatIP(%d)", self.d_model)
        except ImportError:
            _log.debug("SemanticTrieIndex: FAISS not available — numpy fallback active")

    def add(self, state_vec: torch.Tensor, token: str) -> None:
        """Register one (state, next_token) pair."""
        try:
            v = self._to_numpy(state_vec)
            if self._use_faiss:
                self._faiss_index.add(v)
            else:
                self._vecs.append(v)
            self._tokens.append(token)
            self._n += 1
        except Exception:
            pass

    # ── query ────────────────────────────────────────────────────────────────

    def query(self, state_vec: torch.Tensor) -> dict[str, float]:
        """
        Return a synthetic next-token distribution using the k nearest stored states.
        Returns {} when the index is empty.
        """
        if self._n == 0:
            return {}

        k = min(self.k, self._n)
        try:
            q = self._to_numpy(state_vec)

            if self._use_faiss:
                sims, ids = self._faiss_index.search(q, k)
                sims = sims[0].tolist()
                ids  = ids[0].tolist()
            else:
                mat  = np.concatenate(self._vecs[:self._n], axis=0)  # (n, d)
                dots = (mat @ q.T).squeeze(-1)                        # (n,)
                top  = np.argsort(dots)[::-1][:k]
                ids  = top.tolist()
                sims = dots[top].tolist()

            # Temperature-scaled softmax (numerically stable)
            max_s    = max(sims)
            exp_sims = [math.exp((s - max_s) / self.tau) for s in sims]
            total    = sum(exp_sims) or 1.0

            dist: dict[str, float] = {}
            for exp_s, idx in zip(exp_sims, ids):
                if 0 <= idx < len(self._tokens):
                    tok = self._tokens[idx]
                    dist[tok] = dist.get(tok, 0.0) + exp_s / total
            return dist

        except Exception as exc:
            _log.debug("SemanticTrieIndex.query error: %s", exc)
            return {}

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str = _DEFAULT_PATH) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path + ".tokens.json", "w") as f:
                json.dump(self._tokens, f)
            if self._use_faiss:
                import faiss
                faiss.write_index(self._faiss_index, path + ".faiss")
            else:
                if self._vecs:
                    mat = np.concatenate(self._vecs, axis=0)
                    np.save(path + ".npy", mat)
        except Exception as exc:
            _log.debug("SemanticTrieIndex.save error: %s", exc)

    def load(self, path: str = _DEFAULT_PATH) -> bool:
        try:
            tpath = path + ".tokens.json"
            if not os.path.exists(tpath):
                return False
            with open(tpath) as f:
                self._tokens = json.load(f)
            self._n = len(self._tokens)

            if self._use_faiss:
                import faiss
                fpath = path + ".faiss"
                if os.path.exists(fpath):
                    self._faiss_index = faiss.read_index(fpath)
                    return True
            else:
                npath = path + ".npy"
                if os.path.exists(npath):
                    mat = np.load(npath)
                    self._vecs = [mat[i : i + 1] for i in range(mat.shape[0])]
                    return True
            return self._n > 0
        except Exception as exc:
            _log.debug("SemanticTrieIndex.load error: %s", exc)
            return False

    # ── helpers ───────────────────────────────────────────────────────────────

    def _to_numpy(self, vec: torch.Tensor) -> np.ndarray:
        """L2-normalize and convert to (1, d_model) float32 numpy array."""
        v = vec.detach().float().cpu()
        v = F.normalize(v.view(1, -1), p=2, dim=-1)
        return v.numpy().astype(np.float32)

    def __len__(self) -> int:
        return self._n


# ── module-level singleton ─────────────────────────────────────────────────────

_INDEX: SemanticTrieIndex | None = None


def get_semantic_index() -> SemanticTrieIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = SemanticTrieIndex()
        _INDEX.load()
    return _INDEX


def reset_semantic_index() -> None:
    global _INDEX
    _INDEX = None
