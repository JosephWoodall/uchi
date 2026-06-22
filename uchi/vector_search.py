import math
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

class NumpyVectorIndex:
    """
    Optimal Vector Search Index using NumPy.
    Stores and retrieves distributions (vectors) to enable semantic generation.
    """
    def __init__(self, vocab_size: int, use_faiss: bool = False):
        if not _NUMPY_AVAILABLE:
            raise ImportError("numpy is required for Optimal Vector Search")
        self.vocab_size = vocab_size
        self.vectors = []
        self.payloads = [] # Store reference to TrieNode or distributions
        self._matrix = None
        self._needs_rebuild = True

    def add(self, vector: list[float] | dict, payload: any):
        vec = self._to_dense(vector)
        # Normalize for cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self.vectors.append(vec)
        self.payloads.append(payload)
        self._needs_rebuild = True

    def _to_dense(self, vector) -> np.ndarray:
        if isinstance(vector, dict):
            dense = np.zeros(self.vocab_size, dtype=np.float32)
            for k, v in vector.items():
                if isinstance(k, int) and k < self.vocab_size:
                    dense[k] = v
            return dense
        return np.array(vector, dtype=np.float32)

    def _rebuild(self):
        if not self.vectors:
            self._matrix = np.empty((0, self.vocab_size), dtype=np.float32)
        else:
            self._matrix = np.vstack(self.vectors)
        self._needs_rebuild = False

    def search(self, query: list[float] | dict, top_k: int = 5) -> list[tuple[any, float]]:
        """
        Search for top_k most similar vectors using cosine similarity.
        Returns list of (payload, similarity_score).
        """
        if not self.vectors:
            return []
        if self._needs_rebuild:
            self._rebuild()
            
        q_vec = self._to_dense(query)
        norm = np.linalg.norm(q_vec)
        if norm == 0:
            return []
        q_vec = q_vec / norm
        
        # Cosine similarity (dot product of normalized vectors)
        scores = np.dot(self._matrix, q_vec)
        
        if len(scores) == 0:
            return []
            
        # Get top-k indices
        k = min(top_k, len(scores))
        # argpartition is faster than sort
        idx = np.argpartition(scores, -k)[-k:]
        # Sort the top k
        idx = idx[np.argsort(-scores[idx])]
        
        results = []
        for i in idx:
            if scores[i] > 0.0: # Only return somewhat similar items
                results.append((self.payloads[i], float(scores[i])))
                
        return results

    def __len__(self):
        return len(self.vectors)
