import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging

_log = logging.getLogger(__name__)

class TokenEmbedder(nn.Module):
    """
    Since Uchi's vocabulary is dynamically growing strings, we use the hashing trick
    to map any string token to an embedding deterministically, without a fixed vocab matrix.
    """
    def __init__(self, vocab_size=100000, d_model=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, tokens):
        # tokens is a list of strings
        device = self.embedding.weight.device
        ids = [hash(str(t)) % self.vocab_size for t in tokens]
        idx_tensor = torch.tensor(ids, dtype=torch.long, device=device)
        return self.embedding(idx_tensor) # (seq_len, d_model)

class ContextEncoder(nn.Module):
    """Encodes a sequence of embeddings into a single state_vec."""
    def __init__(self, d_model=64):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        
    def forward(self, embeds):
        # embeds: (seq_len, d_model)
        # return final state: (1, d_model)
        _, h_n = self.gru(embeds.unsqueeze(0)) 
        return h_n[-1] # (1, d_model)

class DynamicsHead(nn.Module):
    def __init__(self, d_model=64, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, state_vec, token_embed):
        return self.norm(self.net(torch.cat([state_vec, token_embed], dim=-1)))

class ValueHead(nn.Module):
    """
    3-layer value estimator with residual skip connection.
    Small-init output layer keeps cold-start value near 0 (not randomly negative).
    """
    def __init__(self, d_model=64, hidden_dim=128):
        super().__init__()
        self.proj1 = nn.Linear(d_model, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.norm2 = nn.LayerNorm(hidden_dim // 2)
        self.out   = nn.Linear(hidden_dim // 2, 1)
        # Direct skip: state → scalar (lets gradient flow from first step)
        self.skip  = nn.Linear(d_model, 1, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.out.weight, -0.01, 0.01)
        nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.skip.weight)

    def forward(self, state_vec):
        h = F.silu(self.norm1(self.proj1(state_vec)))
        h = F.silu(self.norm2(self.proj2(h)))
        return self.out(h) + self.skip(state_vec)

class StateSpaceModel(nn.Module):
    """
    The unified Neuro-Symbolic State Space Model for Uchi.
    """
    def __init__(self, d_model=64):
        super().__init__()
        self.d_model = d_model
        self.embedder = TokenEmbedder(d_model=d_model)
        self.encoder = ContextEncoder(d_model=d_model)
        self.dynamics = DynamicsHead(d_model=d_model)
        self.value = ValueHead(d_model=d_model)
        
    def get_state(self, sequence):
        """Calculates state_t given a raw sequence of string tokens."""
        if not sequence:
            device = self.embedder.embedding.weight.device
            return torch.zeros((1, self.d_model), device=device)
        embeds = self.embedder(sequence)
        return self.encoder(embeds)
        
    def predict_next(self, state_vec, next_token):
        """Predicts state_{t+1} given state_t and a new token string."""
        tok_embed = self.embedder([next_token]) # (1, d_model)
        next_s = self.dynamics(state_vec, tok_embed)
        v = self.value(next_s).squeeze(-1)
        return next_s, v

    def update_value(self, sequence, reward):
        return self.compute_loss(sequence, reward)

    def train_dynamics(self, sequence):
        return self.compute_loss(sequence, reward=None)

    def compute_loss(self, sequence, reward=None):
        if len(sequence) < 2:
            return torch.tensor(0.0, device=self.embedder.embedding.weight.device, requires_grad=True)

        # Cap sequence length to avoid O(n²) blowup on long code blocks
        if len(sequence) > 64:
            sequence = sequence[-64:]

        embeds = self.embedder(sequence)  # (seq_len, d_model)

        # Single O(n) GRU pass — all hidden states
        out, _ = self.encoder.gru(embeds.unsqueeze(0))
        true_states = out.squeeze(0)  # (seq_len, d_model)

        # Dynamics loss: one-step predictions on a random subset of transitions
        n = true_states.shape[0]
        indices = torch.randperm(n - 1, device=embeds.device)[:min(8, n - 1)]
        d_loss = torch.tensor(0.0, device=embeds.device, requires_grad=True)
        for i in indices.tolist():
            s_t = true_states[i].unsqueeze(0)
            tok_embed = embeds[i + 1].unsqueeze(0)
            s_next_pred = self.dynamics(s_t, tok_embed)
            s_next_true = true_states[i + 1].unsqueeze(0).detach()
            d_loss = d_loss + F.mse_loss(s_next_pred, s_next_true)

        # Multi-timestep value loss: discounted reward propagated backward through sequence.
        # Training on multiple timesteps gives the value head denser signal than
        # a single final-state target — especially important on short sequences.
        v_loss = torch.tensor(0.0, device=embeds.device, requires_grad=True)
        if reward is not None:
            gamma = 0.9
            n_steps = min(8, n)
            v_sum = torch.tensor(0.0, device=embeds.device, requires_grad=True)
            for offset in range(n_steps):
                idx = n - 1 - offset
                discounted = float(reward) * (gamma ** offset)
                v_pred = self.value(true_states[idx].unsqueeze(0)).squeeze(-1)
                v_target = torch.tensor([discounted], dtype=torch.float32, device=v_pred.device)
                v_sum = v_sum + F.mse_loss(v_pred, v_target)
            v_loss = v_sum / n_steps

        return d_loss + v_loss

# Global singleton
_SSM = None

def get_ssm(device="cpu"):
    global _SSM
    if _SSM is None:
        _SSM = StateSpaceModel().to(device)
        # strict=False: tolerate architecture changes between versions
        for ckpt in ("ssm_dynamics.pt", "ssm_weights.pt"):
            if os.path.exists(ckpt):
                try:
                    _SSM.load_state_dict(
                        torch.load(ckpt, map_location=device, weights_only=True),
                        strict=False,
                    )
                    break
                except Exception as exc:
                    _log.warning("SSM checkpoint load failed (%s): %s — fresh start", ckpt, exc)
    return _SSM

