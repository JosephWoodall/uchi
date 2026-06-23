"""CPU-bound vector memory store.

Embeddings are persisted as a binary .npy file; text records are stored in a
companion JSON index. Both live on disk so memory survives process restarts.
All operations stay on CPU — GPU VRAM is never touched.
"""

import os
import json
import numpy as np
from typing import List


class CPUVectorMemory:
    """Persistent flat vector store backed by numpy + JSON.

    Interface used by ContinualLearner:
        add_memory(text, embedding)
        retrieve(query_emb, top_k) -> List[str]
        records                    -> List[str]
    """

    def __init__(self, db_path: str = "uchi_cpu_memory"):
        self.db_path = db_path
        self._vec_path = db_path + "_vectors.npy"
        self._idx_path = db_path + "_index.json"

        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        if os.path.exists(self._idx_path) and os.path.exists(self._vec_path):
            with open(self._idx_path) as f:
                self.records: List[str] = json.load(f)
            self._vectors: np.ndarray | None = np.load(self._vec_path)
        else:
            self.records = []
            self._vectors = None

    def add_memory(self, text: str, embedding: np.ndarray):
        vec = embedding.reshape(1, -1).astype(np.float32)
        self._vectors = vec if self._vectors is None else np.vstack([self._vectors, vec])
        self.records.append(text)
        self._save()

    def retrieve(self, query_emb: np.ndarray, top_k: int = 3) -> List[str]:
        if self._vectors is None or not self.records:
            return []
        q = query_emb.reshape(-1).astype(np.float32)
        sims = self._vectors @ q
        k = min(top_k, len(self.records))
        top_idx = np.argsort(sims)[-k:][::-1]
        return [self.records[i] for i in top_idx]

    def retrieve_with_scores(self, query_emb: np.ndarray, top_k: int = 3) -> List[tuple]:
        """Returns (text, cosine_similarity) pairs. Score is in [-1, 1]."""
        if self._vectors is None or not self.records:
            return []
        q = query_emb.reshape(-1).astype(np.float32)
        sims = self._vectors @ q
        k = min(top_k, len(self.records))
        top_idx = np.argsort(sims)[-k:][::-1]
        return [(self.records[i], float(sims[i])) for i in top_idx]

    def _save(self):
        np.save(self._vec_path, self._vectors)
        with open(self._idx_path, "w") as f:
            json.dump(self.records, f)
