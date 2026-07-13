"""build_pruned_vocab_0_5_0.py -- 0.5.0's own pruned vocab, sampled from the
ACTUAL 0.5.0 corpus mix (Stack v2 + SWE-Gym + SWE-Gym-Raw + FineWeb-Edu),
not 0.4.0's old sources (OpenWebText/Wikipedia/the-stack-smol/UltraChat/
OpenOrca/Magicoder -- `scripts/build_pruned_vocab.py`, left untouched).

Why a separate script rather than reusing 0.4.0's: this retrain's corpus is
~75% code/issue-diff by token count (SWE-Gym-Raw alone contributes ~342.6M
of the real, decontaminated ~366M-token local corpus) -- a fundamentally
different token distribution than 0.4.0's general-web-text-dominated mix.
Reusing 0.4.0's `pruned_vocab_32k.json` would prune to the WRONG frequent-
token set for this corpus, undermining the entire point of pruning
(concentrate the embedding table on the tokens THIS corpus actually uses).

Samples proportionally from the real sources this Phase 1 run will
actually consume: Stack v2 + SWE-Gym + SWE-Gym-Raw (from disk, all of
Item 1's real pull, capped at --docs-per-local-source for tokenization
cost) + FineWeb-Edu (streamed, matching the general-text portion).

Usage:
    python -m scripts.build_pruned_vocab_0_5_0 --max-vocab 32000
"""
from __future__ import annotations

import argparse
import json
import os

CORPUS_DIR = os.path.join(".uchi", "corpus")


def _sample_local_jsonl(path: str, n_docs: int, text_fn) -> list[str]:
    texts: list[str] = []
    if not os.path.exists(path):
        print(f"  [!] {path} not found -- skipping")
        return texts
    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(texts) >= n_docs:
                break
            row = json.loads(line)
            text = text_fn(row)
            if text:
                texts.append(text)
    print(f"  sampled {len(texts)} docs from {path}")
    return texts


def _sample_fineweb(n_docs: int) -> list[str]:
    from datasets import load_dataset
    print(f"  sampling {n_docs} docs from FineWeb-Edu ...")
    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    except Exception as e:
        print(f"    [!] FineWeb-Edu unavailable ({type(e).__name__}); skipping.")
        return []
    texts = []
    for i, row in enumerate(ds):
        if i >= n_docs:
            break
        if row.get("text"):
            texts.append(row["text"])
    print(f"    got {len(texts)} docs")
    return texts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-vocab", type=int, default=32000)
    parser.add_argument("--docs-per-local-source", type=int, default=4000)
    parser.add_argument("--fineweb-docs", type=int, default=4000)
    parser.add_argument("--out", type=str, default="uchi/flux/checkpoints/pruned_vocab_0_5_0_32k.json")
    args = parser.parse_args()

    from uchi.flux.tokenizer_v2 import TikTokenHybridTokenizer
    from uchi.flux.vocab_prune import compute_used_vocab, coverage

    tok = TikTokenHybridTokenizer()
    n = args.docs_per_local_source

    all_texts: list[str] = []
    all_texts += _sample_local_jsonl(
        os.path.join(CORPUS_DIR, "stack_v2_sample.jsonl"), n, lambda r: r.get("content", ""),
    )
    all_texts += _sample_local_jsonl(
        os.path.join(CORPUS_DIR, "swe_gym_full.jsonl"), n,
        lambda r: f"# Issue: {r['problem_statement']}\n\n# Fix:\n{r['patch']}",
    )
    all_texts += _sample_local_jsonl(
        os.path.join(CORPUS_DIR, "swe_gym_raw_full.jsonl"), n,
        lambda r: f"# Issue: {r['problem_statement']}\n\n# Fix:\n{r['patch']}",
    )
    all_texts += _sample_fineweb(args.fineweb_docs)

    print(f"\ntotal sampled docs: {len(all_texts)}")
    print("tokenizing...")
    # Raw base-vocab IDs (unshifted) -- strip the n_special shift that
    # encode_text() adds, matching build_pruned_vocab.py's exact convention
    # (PrunedVocab operates on the base cl100k_base ID space; the wrapper
    # re-applies the shift at encode/decode time).
    sequences = []
    for t in all_texts:
        shifted = tok.encode_text(t, max_length=1024)
        sequences.append([i - tok.n_special for i in shifted if i >= tok.n_special])

    total_tokens = sum(len(s) for s in sequences)
    print(f"total tokens: {total_tokens:,}")

    pruned = compute_used_vocab(sequences, max_vocab=args.max_vocab)
    cov = coverage(pruned, sequences)
    print(f"\npruned vocab size: {pruned.size:,} (target max_vocab={args.max_vocab:,})")
    print(f"coverage on sampled corpus: {cov:.4%}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "max_vocab": args.max_vocab,
                "size": pruned.size,
                "coverage": cov,
                "total_tokens_sampled": total_tokens,
                "old_to_new": pruned.old_to_new,
                "new_to_old": {str(k): v for k, v in pruned.new_to_old.items()},
            },
            f,
        )
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
