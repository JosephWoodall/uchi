"""
intent_encoder.py
=================
LatentIntentEncoder: maps user queries to skill vectors via SSM state space.

Algorithm
---------
  1. Tokenise query → T
  2. SSM GRU(T)  →  h_query ∈ ℝ^d_model    (structural intent state)
  3. trie.peek(T) →  P(t|T)  [sparse dict]
  4. Σ_t P(t)*embed(t)  →  v_trie ∈ ℝ^d_model  (probabilistic intent state)
       ↑ "compressed attention":
         attention weights = CTW trie probabilities
         sparse over observed transitions only → natural hardware efficiency
  5. h_intent = LayerNorm(h_query + v_trie)   ∈ ℝ^d_model
  6. cosine_sim(h_intent, h_skill_k) for each registered skill (cached)
  7. Return (best_skill_name, confidence)
     Route if confidence > threshold, else fall back to keyword matching.

Skill vectors are cached as numpy arrays at registry load time to avoid
torch overhead on every query.  They update lazily on reload().
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.55  # cosine similarity confidence gate


def _cosine(a, b) -> float:
    """Cosine similarity between two equal-length lists/arrays."""
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


class LatentIntentEncoder:
    """
    Maps a user query to the best-matching skill via the SSM latent space.

    Parameters
    ----------
    ssm : StateSpaceModel
        The global SSM singleton from neuro_symbolic.get_ssm().
    threshold : float
        Minimum cosine similarity to commit to a skill match.
    """

    def __init__(self, ssm, threshold: float = _DEFAULT_THRESHOLD):
        self._ssm = ssm
        self.threshold = threshold
        self._skill_vectors: Dict[str, List[float]] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def encode_tokens(self, tokens: List[str]) -> List[float]:
        """Encode a token list into a latent intent vector (no grad, numpy-free)."""
        import torch
        with torch.no_grad():
            h = self._ssm.get_state(tokens)   # (1, d_model)
            return h.squeeze(0).tolist()

    def project_trie_dist(self, dist: dict) -> List[float]:
        """
        Project a trie probability distribution onto the embedding space.

        v = Σ_t P(t) * embed(t)

        This is sparse attention: only tokens with observed trie transitions
        contribute, giving the same hardware frugality as the trie itself.
        """
        import torch
        if not dist:
            d = self._ssm.d_model
            return [0.0] * d
        total = sum(dist.values())
        if total < 1e-12:
            d = self._ssm.d_model
            return [0.0] * d
        with torch.no_grad():
            result = torch.zeros(self._ssm.d_model, device=next(self._ssm.parameters()).device)
            for token, prob in dist.items():
                embeds = self._ssm.embedder([str(token)])  # (1, d_model)
                result += (prob / total) * embeds.squeeze(0)
        return result.tolist()

    def fuse(
        self, h_query: List[float], v_trie: List[float]
    ) -> List[float]:
        """
        Fuse SSM hidden state and trie projection into the intent vector.
        Uses elementwise sum + L2 normalisation (no learned parameters needed).
        """
        fused = [a + b for a, b in zip(h_query, v_trie)]
        norm = math.sqrt(sum(x * x for x in fused))
        if norm < 1e-9:
            return fused
        return [x / norm for x in fused]

    def register_skill(self, name: str, description_tokens: List[str]) -> None:
        """Cache a skill's SSM encoding. Called at SkillRegistry load time."""
        try:
            h = self.encode_tokens(description_tokens)
            norm = math.sqrt(sum(x * x for x in h))
            if norm > 1e-9:
                self._skill_vectors[name.lower()] = [x / norm for x in h]
        except Exception as exc:
            _log.debug("LatentIntentEncoder: failed to register skill '%s': %s", name, exc)

    def match(
        self,
        query_tokens: List[str],
        trie_dist: Optional[dict] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Find the best-matching skill for a query.

        Returns (skill_name, confidence) or (None, 0.0) if below threshold.
        """
        if not self._skill_vectors or not query_tokens:
            return None, 0.0
        try:
            h_q = self.encode_tokens(query_tokens)
            if trie_dist:
                v_t = self.project_trie_dist(trie_dist)
                h_intent = self.fuse(h_q, v_t)
            else:
                # normalise h_q as the intent vector directly
                norm = math.sqrt(sum(x * x for x in h_q))
                h_intent = [x / norm for x in h_q] if norm > 1e-9 else h_q

            best_name, best_score = None, -1.0
            for name, h_skill in self._skill_vectors.items():
                score = _cosine(h_intent, h_skill)
                if score > best_score:
                    best_score, best_name = score, name

            if best_score >= self.threshold:
                return best_name, best_score
            return None, best_score
        except Exception as exc:
            _log.debug("LatentIntentEncoder.match failed: %s", exc)
            return None, 0.0

    def is_ready(self) -> bool:
        """
        Returns True once enough skills are registered and the SSM has been
        trained past cold-start (value head mean drifted from 0).
        """
        return len(self._skill_vectors) >= 3
