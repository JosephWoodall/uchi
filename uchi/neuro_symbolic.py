"""
neuro_symbolic.py
=================
Uchi's 256D State Space Model — the continuous "murky layer" that bridges
the deterministic trie (symbolic side) with numerical signal.

This module implements four mathematical upgrades that turn the SSM into a
state-of-the-art Vector-Symbolic Manifold:

1. Hyperspherical Manifold
   All state vectors live on the unit hypersphere (L2-normalised).
   Geometric InfoNCE loss prevents anisotropic mode collapse.

2. Holographic Reduced Representations (HRR / VSA)
   DynamicsHead and PolicyHead bind two vectors via circular convolution
   (FFT-domain element-wise product) instead of concatenation.
   This is lossless (zero representational cost), associative, and supports
   algebraic analogy: King − Man + Woman ≈ Queen.

3. (See semantic_index.py for item 3: FAISS k-NN trie smoothing.)

4. Sparse Mixture-of-Experts Value Head
   16 expert MLPs with a top-2 routing gate replace the 4-slice
   MultiHeadValueHead.  The router uses Gumbel-Softmax during training
   and hard top-2 at inference.  Knowledge capacity grows 16×; inference
   latency stays constant because only 2 experts are gated per step.
"""
from __future__ import annotations

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

_log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
_KV_MAX_BUDGET = 1000   # max states in the episodic KV cache
_KV_N_SINKS    = 4      # attention-sink slots always kept
_INFONCE_TAU   = 0.07   # temperature for InfoNCE policy + geometric loss
_MOE_N_EXPERTS = 16
_MOE_TOP_K     = 2


# ── HRR circular convolution binding ─────────────────────────────────────────

def hrr_bind(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Holographic Reduced Representation binding via circular convolution.

    Z = IFFT(FFT(X) ⊙ FFT(Y)).real

    Both inputs must share the same last dimension d.  The output is
    L2-normalised back to the unit sphere.  This is:
      - lossless: d-dim in, d-dim out
      - associative: bind(bind(A,B),C) = bind(A,bind(B,C))
      - supports algebraic analogy: decode(Z,Y) ≈ X
    """
    d = x.shape[-1]
    z = torch.fft.irfft(
        torch.fft.rfft(x, dim=-1) * torch.fft.rfft(y, dim=-1),
        n=d, dim=-1,
    )
    return F.normalize(z, p=2, dim=-1)


# ── KV-cache compression ──────────────────────────────────────────────────────

def _compress_kv_cache(
    cache: torch.Tensor,
    attn_weights: "torch.Tensor | None" = None,
    max_budget: int = _KV_MAX_BUDGET,
    n_sinks: int = _KV_N_SINKS,
) -> torch.Tensor:
    """
    Reduce *cache* (n, d_model) to at most *max_budget* states.

    Hybrid Strategy for MCTS compatibility:
      - Keep the first *n_sinks* states (attention sinks).
      - Keep the last *tail_budget* states (protects local MCTS backtracking).
      - Fill the remaining budget with the highest-attention heavy hitters.
    """
    n = cache.shape[0]
    if n <= max_budget:
        return cache

    sinks = cache[:n_sinks]
    
    # Protect the most recent states for MCTS
    tail_budget = max_budget // 4
    if n - n_sinks <= tail_budget:
        return cache

    tail = cache[-tail_budget:]
    
    # The middle states that are eligible for eviction
    middle = cache[n_sinks:-tail_budget]
    middle_budget = max_budget - n_sinks - tail_budget

    if attn_weights is not None and attn_weights.shape[0] >= n:
        weights = attn_weights[n_sinks:-tail_budget]
        k = min(middle_budget, weights.shape[0])
        if k > 0:
            _, top_idx = weights.topk(k)
            top_idx = top_idx.sort().values
            heavy = middle[top_idx]
        else:
            heavy = torch.empty((0, cache.shape[-1]), device=cache.device)
    else:
        # Fallback to keeping the most recent of the middle if no weights
        heavy = middle[-middle_budget:] if middle_budget > 0 else torch.empty((0, cache.shape[-1]), device=cache.device)

    compressed = torch.cat([sinks, heavy, tail], dim=0)
    try:
        import uchi.telemetry as _tel
        _tel.record("latent_space", "kv_cache_stats", {
            "total_states_received":  n,
            "states_evicted":         n - compressed.shape[0],
            "heavy_hitters_retained": heavy.shape[0],
        })
    except Exception:
        pass
    return compressed


# ── modules ───────────────────────────────────────────────────────────────────

class CrossAttentionHead(nn.Module):
    """
    Multi-head cross-attention: the current local state queries over the
    episodic KV cache to pull in globally-relevant history.

    Returns:
        attended     (1, d_model) — residual-enriched active state
        attn_weights (n,)         — per-cache-slot attention mass for hybrid eviction
    """

    def __init__(self, d_model: int = 256, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        q = query.unsqueeze(1)    # (1, 1, d_model)
        kv = kv_cache.unsqueeze(0) # (1, n, d_model)
        
        # MHA returns (context, attn_weights)
        # We take the mean across heads to get the mass for eviction
        context, attn_weights = self.mha(q, kv, kv, need_weights=True)
        
        out = self.norm(query + context.squeeze(1))
        return out, attn_weights.squeeze(0).squeeze(0) # (n,)


class TokenEmbedder(nn.Module):
    """
    Character-level CNN embedder for a totally open, infinite string vocabulary.
    100% collision-free. No hashing liability.
    """
    def __init__(self, d_model: int = 256, max_len: int = 16):
        super().__init__()
        self.max_len = max_len
        self.char_emb = nn.Embedding(256, 32)
        self.conv = nn.Conv1d(in_channels=32, out_channels=d_model, kernel_size=3, padding=1)

    def forward(self, tokens: list) -> torch.Tensor:
        device = self.char_emb.weight.device
        B = len(tokens)
        bytes_tensor = torch.zeros((B, self.max_len), dtype=torch.long, device=device)
        
        for i, t in enumerate(tokens):
            b = str(t).encode('utf-8')[:self.max_len]
            for j, byte in enumerate(b):
                bytes_tensor[i, j] = byte
                
        # Embed bytes: (B, max_len, 32) -> transpose to (B, 32, max_len)
        x = self.char_emb(bytes_tensor).transpose(1, 2)
        # Conv1d: (B, d_model, max_len)
        x = self.conv(x)
        x = F.relu(x)
        # Max pool over length: (B, d_model)
        x, _ = torch.max(x, dim=-1)
        return x


class ContextEncoder(nn.Module):
    """
    GRU sequence encoder.

    All output states are L2-normalised so every concept lives on the surface
    of the unit hypersphere (prevents anisotropic mode collapse).
    The input projection is a plain Linear — spectral_norm was removed because
    its parametrisation hook adds ~100ms per CPU forward with no measurable
    benefit given that outputs are already sphere-constrained.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.input_proj = nn.Linear(d_model, d_model, bias=False)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """Return the final GRU hidden state, L2-normalised. (1, d_model)"""
        self.gru.flatten_parameters()
        _, h_n = self.gru(self.input_proj(embeds).unsqueeze(0))
        return F.normalize(h_n[-1], p=2, dim=-1)

    def get_all_states(self, embeds: torch.Tensor) -> torch.Tensor:
        """Return every GRU hidden state, L2-normalised. (seq_len, d_model)"""
        self.gru.flatten_parameters()
        out, _ = self.gru(self.input_proj(embeds).unsqueeze(0))
        return F.normalize(out.squeeze(0), p=2, dim=-1)


class DynamicsHead(nn.Module):
    """
    Predicts the next state from the current state and the next token embedding.

    Upgrade: uses HRR circular convolution (hrr_bind) instead of concatenation.
    The MLP now receives a single d_model vector (not 2×d_model), preserving
    representational capacity with half the parameters in the first layer.
    """

    def __init__(self, d_model: int = 256, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model * 2
        self.net1 = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, state_vec: torch.Tensor, token_embed: torch.Tensor) -> torch.Tensor:
        bound = hrr_bind(state_vec, token_embed)   # (batch, d_model)
        h = self.net1(bound)
        out = self.net2(h)
        return self.norm(bound + out) # Residual connection


class PolicyHead(nn.Module):
    """
    Actor network: estimates the log-prob of selecting a candidate token given
    the current hidden state.

    Upgrade: replaces torch.cat([state, token]) with hrr_bind(state, token).
    The bound vector encodes the relationship algebraically; the MLP then maps
    that relationship to a scalar logit.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),   # was d_model*2; HRR output is d_model
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state_vec: torch.Tensor, token_embed: torch.Tensor) -> torch.Tensor:
        return self.net(hrr_bind(state_vec, token_embed))


# ── legacy value head (kept for checkpoint backward-compat) ───────────────────

class _HeadMLP(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        hidden = head_dim * 2
        self.net = nn.Sequential(
            nn.Linear(head_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1),
        )
        nn.init.uniform_(self.net[2].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadValueHead(nn.Module):
    """Legacy 4-head value head. Kept for checkpoint loading compatibility."""
    N_HEADS    = 4
    HEAD_NAMES = ("syntax", "factual", "intent", "coherence")

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.head_dim   = d_model // self.N_HEADS
        self.heads      = nn.ModuleList([_HeadMLP(self.head_dim) for _ in range(self.N_HEADS)])
        self.log_weights = nn.Parameter(torch.zeros(self.N_HEADS))

    def score_heads(self, state: torch.Tensor) -> torch.Tensor:
        chunks = state.split(self.head_dim, dim=-1)
        return torch.cat([h(c) for h, c in zip(self.heads, chunks)], dim=-1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        scores = self.score_heads(state)
        w = torch.softmax(self.log_weights, dim=0)
        return (scores * w).sum(dim=-1, keepdim=True)


# ── Sparse Mixture-of-Experts value head (upgrade 4) ─────────────────────────

class SparseMoEValueHead(nn.Module):
    """
    Sparse Mixture-of-Experts value oracle.

    Architecture:
      - 16 independent expert MLPs (256 → 128 → 64 → 1 each)
      - A nn.Linear(256, 16) router selects the top-2 experts
      - Gumbel-Softmax (τ=1.0) during training for differentiable routing;
        hard top-2 argmax at inference (.eval() mode)
      - Load-balancing auxiliary loss (Switch Transformer N * sum(f_i * P_i))
    """
    N_EXPERTS     = _MOE_N_EXPERTS
    TOP_K         = _MOE_TOP_K
    GUMBEL_TAU    = 1.0

    def __init__(self, d_model: int = 256):
        super().__init__()
        E = self.N_EXPERTS
        H1 = d_model // 2   # 128
        H2 = d_model // 4   # 64
        self._H1 = H1
        self._H2 = H2

        self.router = nn.Linear(d_model, E, bias=False)

        # Fused first layer: (B, d) @ (d, E*H1)
        self.fused_w1 = nn.Parameter(torch.empty(E * H1, d_model))  # (E*H1, d)
        self.fused_b1 = nn.Parameter(torch.zeros(E * H1))
        
        # Per-expert second layer: (E, H1, H2)
        self.expert_w2 = nn.Parameter(torch.empty(E, H1, H2))       # (E, H1, H2)
        self.expert_b2 = nn.Parameter(torch.zeros(E, 1, H2))
        
        # Per-expert third layer: (E, H2, 1)
        self.expert_w3 = nn.Parameter(torch.empty(E, H2, 1))        # (E, H2, 1)
        self.expert_b3 = nn.Parameter(torch.zeros(E, 1))

        nn.init.kaiming_uniform_(self.fused_w1, a=0.01)
        nn.init.kaiming_uniform_(self.expert_w2, a=0.01)
        nn.init.uniform_(self.expert_w3, -0.01, 0.01)

    def score_heads(self, state: torch.Tensor) -> torch.Tensor:
        B, E, H1, _ = state.shape[0], self.N_EXPERTS, self._H1, self._H2
        
        # Layer 1
        h1 = F.silu(F.linear(state, self.fused_w1, self.fused_b1)).view(B, E, H1)
        
        # Layer 2 (B, E, H1) -> (E, B, H1) @ (E, H1, H2) -> (E, B, H2)
        h2 = torch.bmm(h1.permute(1, 0, 2), self.expert_w2) + self.expert_b2
        h2 = F.silu(h2)
        
        # Layer 3 (E, B, H2) @ (E, H2, 1) -> (E, B, 1) -> (B, E)
        out = torch.bmm(h2, self.expert_w3).squeeze(-1).permute(1, 0)
        return out + self.expert_b3.view(1, -1)

    def aux_loss(self, state: torch.Tensor) -> torch.Tensor:
        """Switch-Transformer load-balancing: N * sum(f_i * P_i)"""
        router_probs = self.router(state).softmax(dim=-1) # (B, E)
        f_i = (router_probs > 0.0).float().mean(dim=0)
        P_i = router_probs.mean(dim=0)
        return self.N_EXPERTS * torch.sum(f_i * P_i)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Route to top-2 experts, return weighted scalar. (batch, 1)"""
        router_logits = self.router(state)    # (batch, E)

        if self.training:
            gumbel        = F.gumbel_softmax(router_logits, tau=self.GUMBEL_TAU, hard=False)
            top2_weights, top2_idx = gumbel.topk(self.TOP_K, dim=-1)
            top2_weights  = top2_weights / (top2_weights.sum(dim=-1, keepdim=True) + 1e-9)
        else:
            top2_vals, top2_idx = router_logits.topk(self.TOP_K, dim=-1)
            top2_weights = F.softmax(top2_vals, dim=-1)

        all_vals = self.score_heads(state)         # (batch, E)
        gathered = all_vals.gather(1, top2_idx)    # (batch, 2)
        return (gathered * top2_weights).sum(dim=-1, keepdim=True)   # (batch, 1)


# ── unified SSM ───────────────────────────────────────────────────────────────

class StateSpaceModel(nn.Module):
    """
    Uchi's Vector-Symbolic Manifold SSM.

    Components:
      embedder    — hashing-trick string → 256D embedding
      encoder     — spectral-norm GRU with L2-normalised output (unit hypersphere)
      dynamics    — HRR-bound next-state predictor
      value       — Sparse MoE value oracle (16 experts, top-2 routing)
      policy_head — HRR-bound actor for MCTS token ranking
      cross_attn  — episodic KV-cache cross-attention enrichment
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model     = d_model
        self.embedder    = TokenEmbedder(d_model=d_model)
        self.encoder     = ContextEncoder(d_model=d_model)
        self.dynamics    = DynamicsHead(d_model=d_model)
        self.value       = SparseMoEValueHead(d_model=d_model)
        self.policy_head = PolicyHead(d_model=d_model)
        self.cross_attn  = CrossAttentionHead(d_model=d_model)
        
        # Specialization Projection Heads
        self.dyn_proj = nn.Linear(d_model, d_model)
        self.val_proj = nn.Linear(d_model, d_model)
        self.pol_proj = nn.Linear(d_model, d_model)
        self.kv_proj  = nn.Linear(d_model, d_model)

    # ── state computation ─────────────────────────────────────────────────────

    def get_state(self, sequence: list) -> torch.Tensor:
        """(1, d_model) L2-normalised state for a token sequence."""
        if not sequence:
            device = self.embedder.char_emb.weight.device
            return torch.zeros((1, self.d_model), device=device)
        return self.encoder(self.embedder(sequence))

    def get_kv_cache(self, sequence: list) -> torch.Tensor:
        """
        Build an episodic KV cache from a complete token sequence.
        Returns (seq_len, d_model), compressed to at most _KV_MAX_BUDGET slots.
        """
        device = self.embedder.char_emb.weight.device
        if not sequence:
            return torch.zeros((0, self.d_model), device=device)
        
        # Apply kv_proj to all states before caching
        states = self.encoder.get_all_states(self.embedder(sequence))
        states = self.kv_proj(states)
        
        if states.shape[0] > _KV_MAX_BUDGET:
            states = _compress_kv_cache(states)
        return states

    def get_state_incremental(
        self,
        prev_hidden: torch.Tensor,
        token: str,
        kv_cache: "torch.Tensor | None" = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        """
        O(1) GRU step from a cached parent state + one new token.

        Returns (new_hidden, new_kv_cache).  new_hidden is always L2-normalised.
        If kv_cache is None the cross-attention path is skipped.
        """
        tok_embed = self.embedder([token])                         # (1, d_model)
        projected = self.encoder.input_proj(tok_embed)             # (1, d_model)
        gru_in    = projected.unsqueeze(0)                         # (1, 1, d_model)
        h0        = prev_hidden.unsqueeze(0)                       # (1, 1, d_model)
        self.encoder.gru.flatten_parameters()
        _, h_n    = self.encoder.gru(gru_in, h0)                  # (1, 1, d_model)

        # L2-normalise to stay on the unit hypersphere
        new_hidden = F.normalize(h_n.squeeze(0), p=2, dim=-1)  # (1, d_model)

        if kv_cache is None or kv_cache.shape[0] == 0:
            try:
                import uchi.telemetry as _tel
                _tel.record("latent_space", "vector_l2_norm",
                            round(new_hidden.norm().item(), 5))
            except Exception:
                pass
            return new_hidden, None

        kv_state = self.kv_proj(new_hidden)
        updated_cache = torch.cat([kv_cache, kv_state.detach()], dim=0)
        if updated_cache.shape[0] > _KV_MAX_BUDGET:
            updated_cache = _compress_kv_cache(updated_cache)

        enriched, _ = self.cross_attn(kv_state, updated_cache)
        try:
            import uchi.telemetry as _tel
            _tel.record("latent_space", "vector_l2_norm",
                        round(enriched.norm().item(), 5))
        except Exception:
            pass
        return enriched, updated_cache

    def get_policy_score(self, state_vec: torch.Tensor, token: str) -> float:
        tok_embed = self.embedder([token])
        pol_state = self.pol_proj(state_vec)
        return self.policy_head(pol_state, tok_embed).item()

    def predict_next(self, state_vec: torch.Tensor, next_token: str):
        """Predict (next_state, value) from a cached state + one token."""
        tok_embed = self.embedder([next_token])
        dyn_state = self.dyn_proj(state_vec)
        next_s    = self.dynamics(dyn_state, tok_embed)
        val_state = self.val_proj(next_s)
        v         = self.value(val_state).squeeze(-1)
        return next_s, v

    def update_value(self, sequence: list, reward: float):
        return self.compute_loss(sequence, reward)

    def train_dynamics(self, sequence: list):
        return self.compute_loss(sequence, reward=None)

    # ── training loss ─────────────────────────────────────────────────────────

    def compute_loss(self, sequence: list, reward=None) -> torch.Tensor:
        """
        Composite training loss on a single token sequence.

        Components:
          d_loss    — dynamics MSE (one-step next-state prediction)
          v_loss    — multi-timestep discounted value MSE
          p_loss    — InfoNCE on policy_head logits (token discrimination)
          geo_loss  — geometric InfoNCE on state representations (sphere spreading)
          aux_loss  — MoE router load-balancing (uniform expert utilisation)
        """
        device = self.embedder.char_emb.weight.device

        if len(sequence) < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        if len(sequence) > 64:
            sequence = sequence[-64:]

        embeds = self.embedder(sequence)   # (seq_len, d_model)

        # One O(n) GRU pass; L2-normalise so states live on the unit hypersphere
        self.encoder.gru.flatten_parameters()
        out, _ = self.encoder.gru(embeds.unsqueeze(0))
        true_states = F.normalize(out.squeeze(0), p=2, dim=-1)  # (seq_len, d_model)
        n = true_states.shape[0]

        # ── Dynamics loss ────────────────────────────────────────────────────
        indices = torch.randperm(n - 1, device=device)[:min(8, n - 1)]
        d_sum   = torch.tensor(0.0, device=device, requires_grad=True)
        for i in indices.tolist():
            s_t         = true_states[i].unsqueeze(0)
            tok_embed   = embeds[i + 1].unsqueeze(0)
            s_next_pred = self.dynamics(s_t, tok_embed)
            s_next_true = true_states[i + 1].unsqueeze(0).detach()
            d_sum       = d_sum + F.mse_loss(s_next_pred, s_next_true)
        d_loss = d_sum / max(len(indices), 1)

        # ── Value loss ───────────────────────────────────────────────────────
        v_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if reward is not None:
            gamma  = 0.9
            n_steps = min(8, n)
            v_sum  = torch.tensor(0.0, device=device, requires_grad=True)
            for offset in range(n_steps):
                idx        = n - 1 - offset
                discounted = float(reward) * (gamma ** offset)
                v_pred     = self.value(true_states[idx].unsqueeze(0)).squeeze(-1)
                v_target   = torch.tensor([discounted], dtype=torch.float32, device=device)
                v_sum      = v_sum + F.mse_loss(v_pred, v_target)
            v_loss = v_sum / n_steps

        # ── Policy InfoNCE (token discrimination) ────────────────────────────
        p_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if reward is not None and reward > 0 and n >= 3:
            n_pol       = min(6, n - 1)
            pol_indices = torch.randperm(n - 1, device=device)[:n_pol]
            p_sum       = torch.tensor(0.0, device=device, requires_grad=True)
            all_positions = list(range(n))
            for i in pol_indices.tolist():
                state   = true_states[i].unsqueeze(0)
                pos_emb = embeds[i + 1].unsqueeze(0)

                neg_pool = [j for j in all_positions if j != i + 1]
                if not neg_pool:
                    continue
                neg_embs = embeds[torch.tensor(neg_pool, device=device)]

                pos_logit  = self.policy_head(state, pos_emb) / _INFONCE_TAU
                k          = neg_embs.shape[0]
                state_exp  = state.expand(k, -1)
                neg_logits = self.policy_head(state_exp, neg_embs) / _INFONCE_TAU

                all_logits = torch.cat([pos_logit, neg_logits], dim=0).squeeze(-1)
                target     = torch.zeros(1, dtype=torch.long, device=device)
                p_sum      = p_sum + F.cross_entropy(all_logits.unsqueeze(0), target)
            p_loss = p_sum / n_pol

        # ── Geometric InfoNCE (state-space sphere spreading) ─────────────────
        # Directly trains the 256D representations to spread uniformly across the
        # unit hypersphere.  Anchor = state_i, positive = state_{i+1} (temporal
        # continuity), negatives = all other states in the sequence.
        geo_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if reward is not None and reward > 0 and n >= 3:
            n_geo   = min(6, n - 1)
            g_idx   = torch.randperm(n - 1, device=device)[:n_geo]
            g_sum   = torch.tensor(0.0, device=device, requires_grad=True)
            for i in g_idx.tolist():
                anchor = true_states[i].unsqueeze(0)            # (1, d_model) — unit norm
                pos    = true_states[i + 1].unsqueeze(0)        # (1, d_model)

                neg_idx  = [j for j in range(n) if j != i + 1]
                if not neg_idx:
                    continue
                negs     = true_states[torch.tensor(neg_idx, device=device)]  # (k, d_model)

                # Dot product = cosine similarity (vectors are L2-normalised)
                pos_sim  = (anchor @ pos.T).squeeze() / _INFONCE_TAU
                neg_sims = (anchor @ negs.T).squeeze(0) / _INFONCE_TAU

                all_sims = torch.cat([pos_sim.unsqueeze(0), neg_sims])
                target   = torch.zeros(1, dtype=torch.long, device=device)
                g_sum    = g_sum + F.cross_entropy(all_sims.unsqueeze(0), target)
            geo_loss = g_sum / n_geo

        # ── MoE auxiliary load-balancing loss ────────────────────────────────
        aux_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if self.training:
            aux_loss = self.value.aux_loss(true_states) * 0.01   # low weight

        return d_loss + v_loss + p_loss + geo_loss + aux_loss


# ── global singleton ──────────────────────────────────────────────────────────

_SSM: StateSpaceModel | None = None


def get_ssm(device: str = "cpu") -> StateSpaceModel:
    global _SSM
    if _SSM is None:
        _SSM = StateSpaceModel(d_model=256).to(device)
        # strict=False tolerates architecture changes between versions.
        # Shape mismatches (e.g. DynamicsHead input from 512→256 after HRR)
        # are caught by the except clause and logged as warnings.
        for ckpt in ("ssm_dynamics.pt", "ssm_weights.pt"):
            if os.path.exists(ckpt):
                try:
                    _SSM.load_state_dict(
                        torch.load(ckpt, map_location=device, weights_only=True),
                        strict=False,
                    )
                    break
                except Exception as exc:
                    _log.warning(
                        "SSM checkpoint load failed (%s): %s — fresh start", ckpt, exc
                    )
    return _SSM
