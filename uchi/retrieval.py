"""
retrieval.py — the semantic index over Uchi's brain corpus.

Generate-and-Ground grounds every answer in the brain. This module is the
retrieval layer: it stores the brain's knowledge as passages, each embedded with
skip-gram word vectors (validated in experiments/skipgram_probe.py — the char
encoder could not carry word meaning; skip-gram cleanly can), and returns the
passages most relevant to a question.

Design notes
------------
- Word vectors are the semantic substrate. A passage vector is the mean of its
  content-word vectors, L2-normalised. Retrieval is cosine similarity.
- Pure-NumPy at inference (no torch needed to *retrieve*), so it is cheap and
  serialises inside ``brain.uchi`` with the rest of the router state.
- ``build_from_corpus`` is called during the brain build; ``retrieve`` is called
  per question inside the Generate-and-Ground loop.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

_WORD = re.compile(r"[a-z0-9']+")
_STOP = frozenset(
    "the a an of to in and or is are was were be been for on at by with as that "
    "this these those it its there what which who how why when where do does did "
    "from about your you we they he she his her their our".split()
)


class SemanticIndex:
    """Cosine retrieval over passage embeddings built from skip-gram word vectors.

    Parameters
    ----------
    w2i : dict[str, int]
        Word → row index into the embedding matrix.
    E : np.ndarray  (V, dim), float32, L2-normalised rows
        Skip-gram word embeddings.
    """

    def __init__(self, w2i: dict[str, int], E: np.ndarray) -> None:
        self.w2i = w2i
        self.E = E.astype(np.float32)
        self.dim = E.shape[1]
        self.passages: list[str] = []
        self._P: Optional[np.ndarray] = None   # (N, dim) passage embeddings

    # ── construction ──────────────────────────────────────────────────────────
    @classmethod
    def from_embeddings_file(cls, path: str) -> "SemanticIndex":
        """Load skip-gram embeddings saved as a torch dict {w2i, E, dim}."""
        import torch
        d = torch.load(path, map_location="cpu")
        E = torch.nn.functional.normalize(d["E"], p=2, dim=-1).cpu().numpy()
        return cls(d["w2i"], E)

    def _vec(self, text: str) -> Optional[np.ndarray]:
        ids = [self.w2i[w] for w in _WORD.findall(text.lower())
               if w in self.w2i and w not in _STOP and len(w) > 2]
        if not ids:
            return None
        v = self.E[ids].mean(0)
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 0 else None

    def add(self, sentences: list[str]) -> None:
        """Embed and append passages to the index (skips un-embeddable ones)."""
        vecs = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            v = self._vec(s)
            if v is not None:
                self.passages.append(s)
                vecs.append(v)
        if vecs:
            new = np.stack(vecs)
            self._P = new if self._P is None else np.vstack([self._P, new])

    def build_from_corpus(self, text: str) -> None:
        """Split a corpus blob into sentences and index them."""
        sents = [s for s in re.split(r"(?<=[.!?])\s+", text)
                 if 5 <= len(_WORD.findall(s)) <= 60]
        self.add(sents)

    # ── retrieval ─────────────────────────────────────────────────────────────
    def retrieve(self, query: str, k: int = 10, lex_weight: float = 0.5) -> list[tuple[str, float]]:
        """Return up to k (passage, cosine) pairs, hybrid-ranked.

        Semantic cosine finds the right topic; a lexical term-overlap re-rank over
        the semantic candidate pool surfaces the answer-BEARING passage (which
        shares the question's content words). The returned score is the semantic
        cosine (so downstream similarity gates stay meaningful); ordering is hybrid.
        """
        if self._P is None or not self.passages:
            return []
        qv = self._vec(query)
        if qv is None:
            return []
        sims = self._P @ qv
        pool = min(max(k * 5, k), len(self.passages))
        cand = np.argpartition(-sims, pool - 1)[:pool]
        qwords = {w for w in _WORD.findall(query.lower())
                  if w not in _STOP and len(w) > 2}
        def hybrid(i: int) -> float:
            if not qwords:
                return float(sims[i])
            pw = set(_WORD.findall(self.passages[i].lower()))
            lex = len(qwords & pw) / len(qwords)
            return float(sims[i]) + lex_weight * lex
        top = sorted(cand.tolist(), key=hybrid, reverse=True)[:k]
        return [(self.passages[i], float(sims[i])) for i in top]

    def __len__(self) -> int:
        return len(self.passages)

    # keep pickling lean: embeddings matrix can be large but is the point of the
    # index; passages + _P travel with the brain.
    def __getstate__(self):
        return {"w2i": self.w2i, "E": self.E, "dim": self.dim,
                "passages": self.passages, "_P": self._P}

    def __setstate__(self, s):
        self.__dict__.update(s)
