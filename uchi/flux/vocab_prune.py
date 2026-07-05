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

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List

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
