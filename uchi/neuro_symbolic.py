import torch
import torch.nn as nn
import torch.nn.functional as F
import os

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
    def __init__(self, d_model=64, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state_vec):
        return self.net(state_vec)

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

        # Single O(n) GRU pass returns all hidden states — encoder.gru is (seq_len, d_model)
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
            d_loss = d_loss + nn.functional.mse_loss(s_next_pred, s_next_true)

        # Value loss from final state
        v_loss = torch.tensor(0.0, device=embeds.device, requires_grad=True)
        if reward is not None:
            v_pred = self.value(true_states[-1].unsqueeze(0)).squeeze(-1)
            v_target = torch.tensor([reward], dtype=torch.float32, device=v_pred.device)
            v_loss = nn.functional.mse_loss(v_pred, v_target)

        return d_loss + v_loss

# Global singleton
_SSM = None

def get_ssm(device="cpu"):
    global _SSM
    if _SSM is None:
        _SSM = StateSpaceModel().to(device)
        # Attempt to load weights if they exist
        ckpt = "ssm_weights.pt"
        if os.path.exists(ckpt):
            _SSM.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return _SSM

