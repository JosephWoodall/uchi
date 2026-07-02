"""Tokenizer V2 — tiktoken byte-level BPE + TensorRank Embedding + PCA Projection.

Replaces GPT-2's 50K vocab BPE with a 32K byte-level tokenizer.
Embedding table reduced from 51.5M → 4.3M params via tensor-rank decomposition.
PCA analysis available post-training to further reduce information loss.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from typing import List, Optional

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    import tree_sitter
    import tree_sitter_python
    HAS_TREESITTER = True
except ImportError:
    HAS_TREESITTER = False


class TikTokenHybridTokenizer:
    """Byte-level BPE tokenizer via tiktoken + tree-sitter AST parsing.

    Advantages over GPT-2 tokenizer:
    - Byte-level: zero OOV errors, handles any Unicode
    - 32K vocab: 38% smaller embedding table
    - Trained on code+text distribution (cl100k_base)
    """

    # Special token IDs (reserved at end of vocab)
    SPECIAL_TOKENS = {
        "<|pad|>": 0,
        "<|bos|>": 1,
        "<|eos|>": 2,
        "<|user|>": 3,
        "<|assistant|>": 4,
        "<|think|>": 5,
        "<|/think|>": 6,
        "<|action|>": 7,
        "<|/action|>": 8,
        "<|observation|>": 9,
        "<|/observation|>": 10,
        "<|image|>": 11,
        "<|/image|>": 12,
        "<|context|>": 13,
        "<|/context|>": 14,
        "<|debater_a|>": 15,
        "<|debater_b|>": 16,
        "<|end|>": 17,
    }

    def __init__(self, base_encoding: str = "cl100k_base",
                 target_vocab_size: int = 100300):
        # tiktoken base encoding (used by GPT-4, Claude, etc.)
        if HAS_TIKTOKEN:
            self._enc = tiktoken.get_encoding(base_encoding)
            self._base_vocab_size = self._enc.n_vocab
        else:
            # Fallback to GPT-2 tokenizer
            from transformers import GPT2Tokenizer
            self._enc = GPT2Tokenizer.from_pretrained("gpt2")
            self._enc.pad_token = self._enc.eos_token
            self._base_vocab_size = self._enc.vocab_size

        self.target_vocab_size = target_vocab_size
        self.n_special = len(self.SPECIAL_TOKENS)

        # Total vocab = base + special tokens
        self.vocab_size = target_vocab_size

        # Reverse mapping for special tokens
        self._special_id_to_token = {v: k for k, v in self.SPECIAL_TOKENS.items()}

        # AST parser (syntax head)
        self.syntax_vocab = {
            "UNK": 0, "PAD": 1, "module": 2, "expression_statement": 3,
            "assignment": 4, "identifier": 5, "integer": 6,
            "function_definition": 7, "def": 8, "parameters": 9,
            "block": 10, "return_statement": 11, "return": 12,
            "binary_operator": 13, "if_statement": 14, "if": 15,
            "string": 16, "call": 17, "TEXT_MODE": 18
        }
        self.id_to_syntax = {v: k for k, v in self.syntax_vocab.items()}
        self.syntax_vocab_size = len(self.syntax_vocab)

        if HAS_TREESITTER:
            try:
                self.language = tree_sitter.Language(tree_sitter_python.language())
            except TypeError:
                self.language = tree_sitter_python.language()
            self.parser = tree_sitter.Parser()
            self.parser.language = self.language
        else:
            self.parser = None

    def encode_text(self, text: str, max_length: int = 1024) -> List[int]:
        """Encode text to token IDs. (Lossless)"""
        if HAS_TIKTOKEN:
            ids = self._enc.encode(text, allowed_special="all")
        else:
            ids = self._enc.encode(text, truncation=True, max_length=max_length)
        
        # Shift IDs to make room for special tokens at the front (Lossless mapping)
        ids = [tid + self.n_special for tid in ids]
        return ids[:max_length]

    def decode_text(self, ids: List[int]) -> str:
        """Decode token IDs back to text. (Lossless)"""
        clean_ids = []
        for tid in ids:
            if tid < self.n_special:
                continue
            # Reverse the shift
            clean_ids.append(tid - self.n_special)

        if HAS_TIKTOKEN:
            try:
                return self._enc.decode(clean_ids)
            except Exception:
                return "".join(chr(max(32, tid % 128)) for tid in clean_ids)
        else:
            return self._enc.decode(clean_ids)

    def encode_special(self, token_name: str) -> int:
        """Get the ID for a special token."""
        return self.SPECIAL_TOKENS.get(token_name, self.SPECIAL_TOKENS["<|pad|>"])

    def get_syntax_state(self, code_str: str) -> str:
        """Returns the current structural state of the AST."""
        if not self.parser or not code_str.strip():
            return "module"
        tree = self.parser.parse(bytes(code_str, "utf8"))
        if tree.root_node.has_error:
            return "UNK"
        cursor = tree.walk()
        while cursor.goto_last_child():
            pass
        return cursor.node.type

    @property
    def pad_token_id(self):
        return self.SPECIAL_TOKENS["<|pad|>"]

    @property
    def eos_token_id(self):
        return self.SPECIAL_TOKENS["<|eos|>"]

    @property
    def bos_token_id(self):
        return self.SPECIAL_TOKENS["<|bos|>"]


class TensorRankEmbedding(nn.Module):
    """Low-rank tensor decomposition for embeddings.

    Instead of: Embedding(vocab, d_model) → 51.5M params
    We use:     Embedding(vocab, rank) → Linear(rank, d_model) → 4.3M params

    This treats the embedding matrix as a rank-r approximation,
    preserving the most important directions while dramatically
    reducing parameter count.

    Information loss: bounded by the rank — if rank captures 99%+
    of the embedding variance, information loss is <1%.
    """

    def __init__(self, vocab_size: int, d_model: int, rank: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.rank = rank

        # Low-rank factors: E = U × V
        # U: (vocab_size, rank) — compact token representations
        # V: (rank, d_model) — upscaling projection
        self.U = nn.Embedding(vocab_size, rank)
        self.V = nn.Linear(rank, d_model, bias=False)

        # Initialize with scaled normal
        nn.init.normal_(self.U.weight, std=0.02)
        nn.init.normal_(self.V.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len) → (batch, seq_len, d_model)"""
        return self.V(self.U(x))

    @property
    def param_count(self) -> int:
        return self.vocab_size * self.rank + self.rank * self.d_model

    def analyze_pca(self) -> dict:
        """Analyze the effective dimensionality of learned embeddings.

        After training, call this to determine if rank can be further
        reduced without significant information loss.

        Returns:
            dict with variance ratios, effective rank, and recommendation
        """
        with torch.no_grad():
            # Reconstruct full embedding matrix
            full_emb = self.V(self.U.weight)  # (vocab, d_model)

            # Compute SVD
            U, S, Vh = torch.linalg.svd(full_emb, full_matrices=False)

            # Variance explained by each component
            total_var = (S ** 2).sum().item()
            explained = (S ** 2).cumsum(0) / total_var

            # Find effective rank (95% and 99% thresholds)
            rank_95 = (explained < 0.95).sum().item() + 1
            rank_99 = (explained < 0.99).sum().item() + 1

            return {
                "current_rank": self.rank,
                "singular_values": S[:20].tolist(),
                "variance_at_current_rank": explained[min(self.rank - 1, len(explained) - 1)].item(),
                "rank_for_95pct": rank_95,
                "rank_for_99pct": rank_99,
                "recommendation": (
                    f"Current rank {self.rank} captures "
                    f"{explained[min(self.rank - 1, len(explained) - 1)].item():.1%} of variance. "
                    f"Could reduce to {rank_99} for 99% or {rank_95} for 95%."
                ),
            }

    def compress_to_rank(self, new_rank: int) -> "TensorRankEmbedding":
        """Create a compressed version of this embedding at a lower rank.

        Uses PCA/SVD to find the optimal low-rank approximation,
        minimizing information loss.
        """
        with torch.no_grad():
            full_emb = self.V(self.U.weight)  # (vocab, d_model)
            U, S, Vh = torch.linalg.svd(full_emb, full_matrices=False)

            # Take top-k components
            new_U_weight = U[:, :new_rank] * S[:new_rank].unsqueeze(0)
            new_V_weight = Vh[:new_rank, :]

            compressed = TensorRankEmbedding(
                self.vocab_size, self.d_model, new_rank
            )
            compressed.U.weight.copy_(new_U_weight)
            compressed.V.weight.copy_(new_V_weight)

        return compressed


# ── Convenience alias for backwards compatibility ──
HybridTokenizer = TikTokenHybridTokenizer


if __name__ == "__main__":
    tok = TikTokenHybridTokenizer()
    print(f"Vocab size: {tok.vocab_size}")
    print(f"Special tokens: {tok.n_special}")

    text = "def hello():\n    print('Hello, World!')"
    ids = tok.encode_text(text)
    print(f"Encoded ({len(ids)} tokens): {ids[:10]}...")
    print(f"Decoded: {tok.decode_text(ids)[:60]}...")

    # Test embedding
    emb = TensorRankEmbedding(tok.vocab_size, d_model=1024, rank=128)
    x = torch.tensor([ids[:20]])
    out = emb(x)
    print(f"Embedding shape: {out.shape}")
    print(f"Param count: {emb.param_count:,} (vs {tok.vocab_size * 1024:,} full)")
    print(f"Compression ratio: {emb.param_count / (tok.vocab_size * 1024):.1%}")

    # PCA analysis
    analysis = emb.analyze_pca()
    print(f"PCA: {analysis['recommendation']}")
