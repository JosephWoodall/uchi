"""AttentionBlock — causal multi-head self-attention for the hybrid SSM+attention stack.

Placed at every 6th layer in HybridTSSM (positions 5, 11, 17, … for n_layers=20).
Provides the long-range inductive capabilities that pure SSMs lack on complex
pattern-matching and in-context reasoning tasks (Arora et al. 2024, "Zoology").

Interface mirrors TSSMBlock:
  forward(x)            → (output, None)           — training / prefill full sequence
  decode_step(x_t, buf) → (y_t, updated_buffer)    — O(L) single-token decode

References:
  Zoology: Measuring and Improving Recall in Efficient Language Models (2024)
  Jamba: A Hybrid Transformer-Mamba Language Model (2024)
  Zamba2: A Compact 2.7B SSM Hybrid Model (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """Causal multi-head self-attention block with pre-norm and residual.

    Designed to slot directly into HybridTSSM.layers alongside TSSMBlock.
    Uses standard PyTorch MHA — no flash-attention dependency, so it runs
    on any hardware. If flash_attn is available, swap in for 2-4× prefill speedup.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model  = d_model
        self.n_heads  = n_heads

        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads,
            batch_first=True,
            dropout=dropout,
            bias=False,
        )
        # Small feed-forward sublayer (1× expansion, keeps param count modest)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

    # ── Training / prefill (full sequence) ────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """Full-sequence causal self-attention.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (output, None) — None matches TSSMBlock's (output, state) signature.
                             State is managed externally via SSMCache for decode.
        """
        seq_len = x.size(1)

        # Pre-norm attention
        normed = self.norm(x)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_out, _ = self.attn(normed, normed, normed,
                                attn_mask=causal_mask, need_weights=False)
        x = x + attn_out

        # Pre-norm feed-forward
        x = x + self.ff(self.ff_norm(x))

        return x, None   # None = no recurrent state

    # ── Single-token decode (KV-buffer mode) ──────────────────────────────────

    def decode_step(
        self,
        x_t: torch.Tensor,         # (batch, d_model)
        kv_buffer: torch.Tensor,    # (batch, past_len, d_model) or None
    ):
        """Append x_t to the KV buffer and compute causal attention.

        Args:
            x_t:       current token embedding (batch, d_model)
            kv_buffer: accumulated input history (batch, past_len, d_model)
                       Stored in SSMCache for this layer. None on first token.

        Returns:
            y_t:        output for current token (batch, d_model)
            new_buffer: kv_buffer with x_t appended (batch, past_len+1, d_model)
        """
        x_t_3d = x_t.unsqueeze(1)  # (batch, 1, d_model)

        if kv_buffer is not None:
            full = torch.cat([kv_buffer, x_t_3d], dim=1)
        else:
            full = x_t_3d   # (batch, 1, d_model)

        seq_len = full.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x_t.device, dtype=torch.bool),
            diagonal=1,
        )

        normed = self.norm(full)
        attn_out, _ = self.attn(normed, normed, normed,
                                attn_mask=causal_mask, need_weights=False)
        attended = full + attn_out

        # Feed-forward on last position only
        last = attended[:, -1, :]                    # (batch, d_model)
        ff_out = last + self.ff(self.ff_norm(last))  # (batch, d_model)

        # Update the stored buffer with the raw input (not post-attention)
        # so future steps can attend to the actual token embeddings
        return ff_out, full
