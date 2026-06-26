"""
vector_oracle.py
================
Geometric grounding for Uchi's Convergent MCTS Engine.

Functions
---------
  encode()              — maps token sequences into ℝ^N via SSM + trie projection
  similarity()          — cosine similarity between two vectors
  token_diversity()     — mean pairwise Jaccard distance (MCTS stopping criterion)
  contrastive_update()  — post-hoc online SSM alignment (called by router after delivery)

The encode() function uses the identical projection pipeline for both queries and
response candidates, ensuring their cosine similarities are semantically meaningful:

  h   = SSM.get_state(tokens)               — GRU hidden state (structural intent)
  v   = Σ_t P(t|tokens) * embed(t)          — sparse trie projection (probabilistic intent)
  out = normalize(h + v)                    — fused intent vector ∈ ℝ^N
"""
from __future__ import annotations

import math
import logging
from typing import Dict, List, Optional

_log = logging.getLogger(__name__)

# ── contrastive loss metrics ──────────────────────────────────────────────────
# Ring buffer: last N loss values recorded by contrastive_update().
# Readable by tests and the evaluator; never blocks the hot path.
_contrastive_loss_history: list[float] = []
_CONTRASTIVE_HISTORY_MAX = 1000


# ── cosine similarity ─────────────────────────────────────────────────────────

def similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length float lists, clamped to [-1, 1]."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    # Clamp to [-1, 1]: floating-point arithmetic on nearly-identical vectors
    # can produce dot / (na * nb) = 1.0000000000000002, which breaks range checks.
    return max(-1.0, min(1.0, dot / (na * nb)))


# ── token diversity (MCTS stopping criterion) ─────────────────────────────────

def token_diversity(candidates: List[List[str]]) -> float:
    """
    Mean pairwise Jaccard distance over candidate token sets.

    0.0 = all candidates identical
    1.0 = maximally distinct

    Used as the MCTS stopping criterion: when diversity drops below ε,
    the trie is producing similar outputs and more rollouts add no value.
    Computed on token sets (not SSM vectors), so it is reliable from cold start.
    """
    n = len(candidates)
    if n < 2:
        return 1.0
    sets = [set(tokens) for tokens in candidates]
    total, count = 0.0, 0
    cap = min(n, 10)
    for i in range(cap):
        for j in range(i + 1, cap):
            union = len(sets[i] | sets[j])
            inter = len(sets[i] & sets[j])
            total += 1.0 - (inter / max(union, 1))
            count += 1
    return total / max(count, 1)


# ── encoding ─────────────────────────────────────────────────────────────────

def encode(
    concepts: List[str],
    trie_dist: Optional[Dict] = None,
    ssm=None,
) -> List[float]:
    """
    Map a token sequence to ℝ^N using SSM hidden state + sparse trie projection.

    If trie_dist is None or empty, falls back to SSM-only encoding.
    If ssm is None, imports the global singleton from neuro_symbolic.
    Run inside torch.no_grad() — inference only, no gradient tracking.
    """
    import torch
    if ssm is None:
        from uchi.neuro_symbolic import get_ssm
        ssm = get_ssm()
    try:
        with torch.no_grad():
            h = ssm.get_state(concepts).squeeze(0)   # (d_model,)

            if trie_dist:
                total = sum(trie_dist.values()) or 1e-12
                v = torch.zeros_like(h)
                for tok, prob in trie_dist.items():
                    emb = ssm.embedder([str(tok)]).squeeze(0)
                    v += (prob / total) * emb
                fused = h + v
            else:
                fused = h

            norm = fused.norm().item()
            if norm < 1e-9:
                return fused.tolist()
            return (fused / norm).tolist()
    except Exception as exc:
        _log.debug("vector_oracle.encode: %s", exc)
        d = getattr(ssm, 'd_model', 256) if ssm is not None else 256
        return [0.0] * d


# ── online contrastive alignment ─────────────────────────────────────────────

def contrastive_update(
    query_tokens: List[str],
    response_tokens: List[str],
    reward: float,
    optimizer,
    ssm=None,
) -> None:
    """
    Online contrastive alignment: train the SSM so query and successful response
    embeddings are geometrically close (reward > 0) or far apart (reward < 0).

    Raw tokens are re-encoded internally with gradients attached so the backward
    pass updates the SSM's embedder and encoder parameters.

    Called by the router AFTER delivering the response to the user — never
    during candidate generation so the embedding space stays stable per turn.
    """
    import torch
    if ssm is None:
        from uchi.neuro_symbolic import get_ssm
        ssm = get_ssm()
    if not query_tokens or not response_tokens:
        return
    try:
        ssm.train()
        optimizer.zero_grad()

        h_q = ssm.get_state(query_tokens).squeeze(0)      # (d_model,)
        h_r = ssm.get_state(response_tokens).squeeze(0)   # (d_model,)

        h_q_n = h_q / (h_q.norm() + 1e-9)
        h_r_n = h_r / (h_r.norm() + 1e-9)
        cos_sim = (h_q_n * h_r_n).sum()

        if reward > 0:
            loss = (1.0 - cos_sim) * min(abs(reward), 1.0)
        else:
            margin = 0.1
            loss = torch.clamp(cos_sim - margin, min=0.0) * min(abs(reward), 1.0)

        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        _log.debug("contrastive_update loss=%.4f reward=%.2f", loss_val, reward)
        _contrastive_loss_history.append(loss_val)
        if len(_contrastive_loss_history) > _CONTRASTIVE_HISTORY_MAX:
            del _contrastive_loss_history[:-_CONTRASTIVE_HISTORY_MAX]
    except Exception as exc:
        _log.debug("vector_oracle.contrastive_update: %s", exc)


def hard_negative_contrastive_update(
    query_tokens: List[str],
    positive_tokens: List[str],
    negative_tokens: List[str],
    reward: float,
    optimizer,
    ssm=None,
    margin: float = 0.3,
) -> float:
    """
    Triplet-style hard negative training for high-dimensional SSM alignment.

    Critical for 256D space: random negatives are already far apart (the space
    is vast), so standard contrastive loss saturates quickly.  Hard negatives
    are candidates the SSM *incorrectly* scores close to the query — training
    against these is what drives real representational improvement.

    Loss = max(0, sim(q, neg) - sim(q, pos) + margin) + attraction_to_pos

    Args:
        query_tokens:    input question tokens
        positive_tokens: oracle-best response (attract toward query)
        negative_tokens: SSM-highest non-best response (repel from query)
        reward:          reward signal for the positive (scales attraction)
        margin:          minimum sim gap between positive and negative (default 0.3)
    Returns:
        loss value (float), or 0.0 on error
    """
    import torch
    if ssm is None:
        from uchi.neuro_symbolic import get_ssm
        ssm = get_ssm()
    if not query_tokens or not positive_tokens or not negative_tokens:
        return 0.0
    try:
        ssm.train()
        optimizer.zero_grad()

        h_q = ssm.get_state(query_tokens).squeeze(0)     # (d_model,)
        h_p = ssm.get_state(positive_tokens).squeeze(0)  # (d_model,)
        h_n = ssm.get_state(negative_tokens).squeeze(0)  # (d_model,)

        h_q_n = h_q / (h_q.norm() + 1e-9)
        h_p_n = h_p / (h_p.norm() + 1e-9)
        h_n_n = h_n / (h_n.norm() + 1e-9)

        sim_pos = (h_q_n * h_p_n).sum()
        sim_neg = (h_q_n * h_n_n).sum()

        # Triplet loss: positive must be at least `margin` closer than negative
        triplet = torch.clamp(sim_neg - sim_pos + margin, min=0.0)

        # Standard attraction toward positive (reward-gated)
        attract = (1.0 - sim_pos) * min(abs(reward), 1.0) if reward > 0 else torch.tensor(0.0, device=h_q.device)

        loss = triplet + 0.5 * attract
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        _log.debug("hard_negative_update loss=%.4f sim_pos=%.3f sim_neg=%.3f",
                   loss_val, sim_pos.item(), sim_neg.item())
        _contrastive_loss_history.append(loss_val)
        if len(_contrastive_loss_history) > _CONTRASTIVE_HISTORY_MAX:
            del _contrastive_loss_history[:-_CONTRASTIVE_HISTORY_MAX]
        return loss_val
    except Exception as exc:
        _log.debug("vector_oracle.hard_negative_contrastive_update: %s", exc)
        return 0.0
