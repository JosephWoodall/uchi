"""vocab_prune.py — vocabulary pruning for FLUX scale-up (0.4.0 Item 0).

The embedding table (tied with the output head) is ~66% of FLUX's total
116M parameters at the full cl100k_base vocab (100,300 tokens, d_model=768
— see the benchmark in this module's tests). Most of that vocab is never
touched by Uchi's actual training corpus (code + prose, not "every token
cl100k_base can represent"). This computes which token IDs the real
corpus actually uses and builds a compact remapping to a smaller vocab —
reclaiming embedding-table capacity for reasoning layers, per the 0.4.0
Item 0 plan.

This is a real, tested utility, not a training run: switching a live
model to a pruned vocab changes the embedding table's shape, which is
incompatible with the currently-loaded ``flux_best.pt`` checkpoint (a
new vocab requires re-tokenizing the corpus and training from scratch —
a multi-hour+ GPU job outside what this session can execute). Run this
against the real training corpus to get the actual achievable
``pruned_vocab_size`` before committing to it for the next training run.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List

if TYPE_CHECKING:
    from .tokenizer_v2 import TikTokenHybridTokenizer

_UNK_TOKEN_ID = -1  # sentinel; mapped vocab reserves slot 0 for UNK


@dataclass
class PrunedVocab:
    """A remapping from full cl100k_base token IDs to a compact vocab."""

    old_to_new: Dict[int, int]      # full vocab id -> compact id (0 = UNK)
    new_to_old: Dict[int, int]      # compact id -> full vocab id (excludes UNK)
    size: int                       # compact vocab size, including UNK slot

    def remap(self, token_ids: Iterable[int]) -> List[int]:
        """Map full-vocab token IDs to compact IDs, falling back to the
        UNK slot (0) for anything outside the pruned vocabulary."""
        return [self.old_to_new.get(t, 0) for t in token_ids]


def compute_used_vocab(
    token_id_sequences: Iterable[List[int]],
    max_vocab: int = 32000,
) -> PrunedVocab:
    """Build a compact vocab from the token IDs actually seen in a corpus.

    *token_id_sequences* is an iterable of already-tokenized (full
    cl100k_base ID) sequences — e.g. ``tokenizer.encode_text(doc)`` for
    every document in the training corpus. Keeps the *max_vocab - 1* most
    frequent token IDs (slot 0 reserved for UNK), so ``size <= max_vocab``.
    """
    counts: Counter = Counter()
    for seq in token_id_sequences:
        counts.update(seq)

    most_common = [tok for tok, _ in counts.most_common(max(0, max_vocab - 1))]
    old_to_new = {old_id: new_id + 1 for new_id, old_id in enumerate(most_common)}
    new_to_old = {v: k for k, v in old_to_new.items()}

    return PrunedVocab(
        old_to_new=old_to_new,
        new_to_old=new_to_old,
        size=len(old_to_new) + 1,  # +1 for the UNK slot
    )


def coverage(pruned: PrunedVocab, token_id_sequences: Iterable[List[int]]) -> float:
    """Fraction of tokens in *token_id_sequences* covered by *pruned*
    (i.e. not falling back to UNK) — the real-world check that a given
    ``max_vocab`` doesn't lose too much of the actual corpus."""
    total = 0
    covered = 0
    for seq in token_id_sequences:
        for tok in seq:
            total += 1
            if tok in pruned.old_to_new:
                covered += 1
    return covered / total if total else 1.0


def load_pruned_vocab(path: str) -> PrunedVocab:
    """Load a ``PrunedVocab`` saved by ``scripts/build_pruned_vocab.py``."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    old_to_new = {int(k): v for k, v in data["old_to_new"].items()}
    new_to_old = {int(k): v for k, v in data["new_to_old"].items()}
    return PrunedVocab(old_to_new=old_to_new, new_to_old=new_to_old, size=data["size"])


class PrunedTikTokenTokenizer:
    """Drop-in wrapper around ``TikTokenHybridTokenizer`` with a pruned
    vocabulary (0.4.0 Item 0).

    Special-token IDs (``0 .. n_special-1``) are left untouched. The
    shifted base cl100k_base range is remapped through a ``PrunedVocab``,
    so the embedding table only needs ``pruned.size + n_special`` rows
    instead of the full ~100K — the actual mechanism behind "cut the
    vocab, reclaim the capacity for reasoning layers."

    Anything outside the pruned vocab decodes as nothing (the UNK slot
    has no real-text inverse) — expected: a token that essentially never
    appeared in the real training corpus isn't worth spending an
    embedding row on, by construction.
    """

    def __init__(self, base_tokenizer: "TikTokenHybridTokenizer", pruned: PrunedVocab):
        self._base = base_tokenizer
        self._pruned = pruned
        self.n_special = base_tokenizer.n_special
        self.vocab_size = pruned.size + self.n_special
        self.syntax_vocab_size = base_tokenizer.syntax_vocab_size
        self.syntax_vocab = base_tokenizer.syntax_vocab
        self.id_to_syntax = base_tokenizer.id_to_syntax
        self.SPECIAL_TOKENS = base_tokenizer.SPECIAL_TOKENS

    def encode_text(self, text: str, max_length: int = 1024) -> List[int]:
        raw_ids = self._base.encode_text(text, max_length=max_length)
        out = []
        for tid in raw_ids:
            if tid < self.n_special:
                out.append(tid)
            else:
                new_id = self._pruned.old_to_new.get(tid - self.n_special, 0)
                out.append(new_id + self.n_special)
        return out

    def decode_text(self, ids: List[int]) -> str:
        raw_ids = []
        for tid in ids:
            if tid < self.n_special:
                raw_ids.append(tid)
                continue
            base_id = self._pruned.new_to_old.get(tid - self.n_special)
            if base_id is not None:
                raw_ids.append(base_id + self.n_special)
            # else: UNK slot -- no real-text inverse, drop it.
        return self._base.decode_text(raw_ids)

    def encode_special(self, token_name: str) -> int:
        return self._base.encode_special(token_name)

    def get_syntax_state(self, code_str: str) -> str:
        return self._base.get_syntax_state(code_str)

    @property
    def pad_token_id(self) -> int:
        return self._base.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self._base.eos_token_id

    @property
    def bos_token_id(self) -> int:
        return self._base.bos_token_id


def load_pruned_tokenizer(path: str, base_tokenizer: "TikTokenHybridTokenizer" = None) -> PrunedTikTokenTokenizer:
    """Load a saved pruned vocab and wrap it around a base tokenizer."""
    if base_tokenizer is None:
        from .tokenizer_v2 import TikTokenHybridTokenizer
        base_tokenizer = TikTokenHybridTokenizer()
    pruned = load_pruned_vocab(path)
    return PrunedTikTokenTokenizer(base_tokenizer, pruned)
