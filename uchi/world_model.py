"""world_model.py — 0.5.0 Item 8: learns FLUX's SSM hidden-state dynamics
plus a value estimate, the substrate Item 8's MCTS (`uchi/mcts.py`) plans
over.

Ports `efficient_llm_training/src/world_model.py`'s `DynamicsHead`,
`ValueHead`, and `WorldModel` near-verbatim — the architecture and
`predict_next`/`rollout`/`update_value`/`rerank_sequences` methods are
model-agnostic, and `HybridTSSM` (`uchi/flux/model.py`) already exposes
`.embedding`/`.layers`/`.norm_f` under the exact names the source assumes,
so nothing there needed renaming.

DynamicsHead: (state_t, action_embed_t) -> state_{t+1}_predicted
ValueHead:    state_t -> V(state_t)

"State" is the normalized final-layer hidden representation at a token
position — `model.norm_f(hidden)[:, t, :]` — a `(batch, d_model)` vector.

One real adaptation from the source: `train_dynamics` scanned a
project-specific `src/hybrid_shards/*.safetensors` directory that has no
equivalent here. Replaced with `train_dynamics` taking token sequences
directly (e.g. real trajectories collected by 0.5.0 Item 6's `GRPOTrainer`)
— same underlying math (supervise `state_{t+1}` prediction from a real
forward pass' ground-truth hidden states), just sourced from in-memory
sequences instead of a shard-file scan. `update_value`/`update_value_from_state`
were already trajectory-data-driven in the source (no shard dependency),
so those port unchanged.

Training either head for real needs states from a proposer coherent enough
to produce non-degenerate trajectories and rewards — the same Phase 1->4
decision gate `grpo.py` and `mcts.py` are gated on. This module is
buildable and unit-testable now against synthetic tensors (see
`tests/test_world_model.py`); real training waits for that gate.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsHead(nn.Module):
    """Predicts next hidden state given current state + action token embedding."""

    def __init__(self, d_model: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, state_vec: torch.Tensor, token_embed: torch.Tensor) -> torch.Tensor:
        """
        state_vec:   (batch, d_model) — current world state
        token_embed: (batch, d_model) — embedding of action token
        Returns:     (batch, d_model) — predicted next state
        """
        return self.norm(self.net(torch.cat([state_vec, token_embed], dim=-1)))


class ValueHead(nn.Module):
    """Scalar value estimate V(state) from world state vector."""

    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        """state_vec: (batch, d_model) → (batch, 1)"""
        return self.net(state_vec)


class WorldModel(nn.Module):
    """
    Combines DynamicsHead and ValueHead into a unified planning module.

    Usage pattern:
      state_t  = extract_state(model, token_sequence)     # real model hidden state
      state_t1 = world_model.dynamics(state_t, tok_embed)  # imagined next state
      value    = world_model.value(state_t1)               # how good is that state?
    """

    CKPT_PATH = "uchi/flux/checkpoints/world_model.pt"

    def __init__(self, d_model: int = 768, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dynamics = DynamicsHead(d_model)
        self.value = ValueHead(d_model)

    def predict_next(self, state_vec: torch.Tensor, token_embed: torch.Tensor):
        """Returns (next_state_vec: (batch,d_model), value: (batch,))."""
        next_s = self.dynamics(state_vec, token_embed)
        v = self.value(next_s).squeeze(-1)
        return next_s, v

    def rollout(self, state_vec: torch.Tensor, token_embeds: list):
        """Unroll dynamics for multiple steps. Returns list of (state, value)."""
        trajectory, s = [], state_vec
        for emb in token_embeds:
            s, v = self.predict_next(s, emb)
            trajectory.append((s, v))
        return trajectory

    # ── Training ──────────────────────────────────────────────────────────

    def train_dynamics(
        self, model, token_sequences: list[list[int]], device,
        steps: int = 300, lr: float = 1e-3, seq_len: int = 32,
    ):
        """Train DynamicsHead to predict the next hidden state given
        current state + token, from real token sequences (e.g. prompt +
        raw generated response ids collected from 0.5.0 Item 6's
        `GRPOTrainer` branches) rather than the source's shard-file scan.

        State at position t = model.norm_f(hidden)[:, t, :] after all
        layers. Action = raw token embedding of token_{t+1}. Target =
        state at position t+1 (ground truth from a real model forward
        pass over the same sequence).
        """
        if not token_sequences:
            print("  [WorldModel] No token sequences given -- skipping dynamics training.")
            return

        opt = torch.optim.AdamW(self.dynamics.parameters(), lr=lr)
        self.dynamics.train()
        model.eval()
        step, total_loss = 0, 0.0

        print(f"  [WorldModel] Dynamics training: {steps} steps | lr={lr}")

        while step < steps:
            for seq in token_sequences:
                if step >= steps:
                    break
                for start in range(0, max(1, len(seq) - seq_len), seq_len):
                    if step >= steps:
                        break
                    chunk = seq[start: start + seq_len]
                    if len(chunk) < 2:
                        continue
                    ids = torch.tensor([chunk], dtype=torch.long, device=device)

                    with torch.no_grad():
                        hidden = model.embedding(ids)
                        for layer in model.layers:
                            hidden, _ = layer(hidden)
                        states = model.norm_f(hidden)          # (1, n, d_model)
                        tok_embeds = model.embedding(ids)       # (1, n, d_model)

                    n = min(seq_len - 1, states.size(1) - 1)
                    if n <= 0:
                        continue
                    state_t = states[0, :n]
                    action_t = tok_embeds[0, 1:n + 1]
                    target = states[0, 1:n + 1].detach()

                    pred = self.dynamics(state_t, action_t)
                    loss = F.mse_loss(pred, target)

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), 1.0)
                    opt.step()

                    total_loss += loss.item()
                    step += 1
                    if step % 100 == 0:
                        print(f"  [WorldModel] Step {step}/{steps} | loss: {total_loss/step:.5f}")

        print(f"  [WorldModel] Done. Avg dynamics loss: {total_loss/max(step,1):.5f}")

    def update_value(self, model, sequence: torch.Tensor,
                     reward: float, device, lr: float = 1e-3):
        """Single-step value head update: V(final_state) → reward.

        Runs a no-grad forward pass to get the final hidden state, then
        backpropagates MSE(V(state), reward) through the value head only.
        """
        self.value.train()
        with torch.no_grad():
            hidden = model.embedding(sequence)
            for layer in model.layers:
                hidden, _ = layer(hidden)
            final_state = model.norm_f(hidden)[:, -1, :].detach()  # (1, d_model)

        return self.update_value_from_state(final_state, reward, device, lr)

    def update_value_from_state(self, final_state: torch.Tensor,
                                reward: float, device, lr: float = 1e-3):
        """Value head update using a pre-computed final hidden state (avoids redundant forward)."""
        self.value.train()
        target = torch.tensor([[reward]], dtype=torch.float32, device=device)
        pred = self.value(final_state.detach())
        loss = F.mse_loss(pred, target)

        for param in self.value.parameters():
            if param.grad is not None:
                param.grad.zero_()
        loss.backward()
        with torch.no_grad():
            for param in self.value.parameters():
                if param.grad is not None:
                    param.data -= lr * param.grad

        return loss.item()

    def rerank_sequences(self, model, sequences: list, device) -> int:
        """Score a list of token sequences by V(final_state), return index of best.

        Used to pick the best branch from parallel thought without running
        the environment reward — purely from the world model's value estimate.
        """
        self.value.eval()
        best_idx, best_val = 0, float("-inf")
        with torch.no_grad():
            for i, seq in enumerate(sequences):
                if seq.dim() == 1:
                    seq = seq.unsqueeze(0)
                hidden = model.embedding(seq)
                for layer in model.layers:
                    hidden, _ = layer(hidden)
                final_state = model.norm_f(hidden)[:, -1, :]
                v = self.value(final_state).item()
                if v > best_val:
                    best_val, best_idx = v, i
        return best_idx

    def save(self, path: str | None = None):
        path = path or self.CKPT_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, device, path: str | None = None) -> bool:
        path = path or self.CKPT_PATH
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            return True
        return False


def extract_state(model, token_ids: torch.Tensor) -> torch.Tensor:
    """Final-layer normalized hidden state at the last position -- the
    `(batch, d_model)` "state" vector every method above consumes. Shared
    helper so `grpo.py`/`mcts.py` don't each reimplement the
    embedding -> layers -> norm_f walk `HybridTSSM.forward` already does
    internally (kept external here since none of these call sites want
    the LM head applied too).
    """
    with torch.no_grad():
        hidden = model.embedding(token_ids)
        for layer in model.layers:
            hidden, _ = layer(hidden)
        hidden = model.norm_f(hidden)
    return hidden[:, -1, :]
