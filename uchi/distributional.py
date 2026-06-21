import math
from typing import Any

def js_divergence(p: dict, q: dict) -> float:
    """Calculate Jensen-Shannon Divergence between two probability dicts."""
    vocab = set(p.keys()) | set(q.keys())
    if not vocab:
        return 0.0
        
    def kld(dist1, dist2):
        res = 0.0
        for k in vocab:
            v1 = dist1.get(k, 0.0)
            if v1 > 0:
                v2 = dist2.get(k, 0.0)
                if v2 == 0:
                    return float('inf')
                res += v1 * math.log2(v1 / v2)
        return res

    m = {}
    for k in vocab:
        m[k] = 0.5 * (p.get(k, 0.0) + q.get(k, 0.0))
        
    return 0.5 * kld(p, m) + 0.5 * kld(q, m)


class DistributionalTokenizer:
    """
    Online semantic clustering via Jensen-Shannon divergence.
    Replaces NLTK WordNet with pure mathematical distributions.
    If two tokens are followed by similar distributions, they are merged.
    """
    def __init__(self, merge_threshold_jsd: float = 0.1, min_obs: int = 50):
        self.merge_threshold = merge_threshold_jsd
        self.min_obs = min_obs
        
        self.successor_counts = {}  # token -> dict of next_token -> count
        self.token_totals = {}      # token -> int
        
        self.clusters = {}          # token -> cluster_id
        self.next_cluster_id = 0
        
        self._last_token = None

    def _normalize(self, token: Any) -> dict:
        counts = self.successor_counts.get(token, {})
        total = self.token_totals.get(token, 0)
        if total == 0:
            return {}
        return {k: v / total for k, v in counts.items()}

    def observe(self, token: Any) -> None:
        if self._last_token is not None:
            prev = self._last_token
            if prev not in self.successor_counts:
                self.successor_counts[prev] = {}
                self.token_totals[prev] = 0
                
            self.successor_counts[prev][token] = self.successor_counts[prev].get(token, 0) + 1
            self.token_totals[prev] += 1
            
            # Periodically attempt clustering
            if self.token_totals[prev] % 100 == 0 and self.token_totals[prev] >= self.min_obs:
                self._attempt_cluster(prev)
                
        self._last_token = token

    def _attempt_cluster(self, target_token: Any):
        if target_token in self.clusters:
            return
            
        target_dist = self._normalize(target_token)
        
        best_candidate = None
        lowest_jsd = float('inf')
        
        for candidate, total in self.token_totals.items():
            if candidate == target_token or total < self.min_obs:
                continue
                
            candidate_dist = self._normalize(candidate)
            jsd = js_divergence(target_dist, candidate_dist)
            
            if jsd < lowest_jsd:
                lowest_jsd = jsd
                best_candidate = candidate
                
        if lowest_jsd < self.merge_threshold and best_candidate is not None:
            # Merge!
            if best_candidate in self.clusters:
                c_id = self.clusters[best_candidate]
            else:
                c_id = f"__CLUSTER_{self.next_cluster_id}__"
                self.clusters[best_candidate] = c_id
                self.next_cluster_id += 1
                
            self.clusters[target_token] = c_id

    def tokenize(self, token: Any) -> Any:
        """Tokenize a single token and update distributions online."""
        self.observe(token)
        if token in self.clusters:
            return self.clusters[token]
        return token
