"""build_pruned_vocab.py — compute the 0.4.0 Item 0 pruned vocab from a real
sample of the actual training corpus mix (OpenWebText, Wikipedia, The Stack,
UltraChat, OpenOrca, Magicoder), not a synthetic guess.

Streams a fixed number of documents from each source, tokenizes with the
full cl100k_base tokenizer, and keeps the most frequent `max_vocab` token
IDs. Saves the resulting PrunedVocab to disk so training scripts can load
it without recomputing (and so it's inspectable/auditable before a
multi-hour run depends on it).

Usage:
    python scripts/build_pruned_vocab.py --max-vocab 32000 --docs-per-source 4000
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _sample_texts(name, config, split, field, n_docs, text_fn=None):
    from datasets import load_dataset
    print(f"  sampling {n_docs} docs from {name} ({split})...")
    try:
        ds = load_dataset(name, config, split=split, streaming=True) if config else \
             load_dataset(name, split=split, streaming=True)
    except Exception as e:
        print(f"    [!] {name} unavailable ({type(e).__name__}); skipping.")
        return []
    texts = []
    try:
        for i, row in enumerate(ds):
            if i >= n_docs:
                break
            text = text_fn(row) if text_fn else row.get(field, "")
            if text:
                texts.append(text)
    except Exception as e:
        print(f"    [!] {name} streaming stopped early ({type(e).__name__}): {e}")
    print(f"    got {len(texts)} docs")
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-vocab", type=int, default=32000)
    parser.add_argument("--docs-per-source", type=int, default=4000)
    parser.add_argument("--out", type=str,
                         default="uchi/flux/checkpoints/pruned_vocab_32k.json")
    args = parser.parse_args()

    from uchi.flux.tokenizer_v2 import TikTokenHybridTokenizer
    from uchi.flux.vocab_prune import compute_used_vocab, coverage

    tok = TikTokenHybridTokenizer()
    n = args.docs_per_source

    all_texts = []
    all_texts += _sample_texts("Skylion007/openwebtext", None, "train", "text", n)
    all_texts += _sample_texts("wikimedia/wikipedia", "20231101.en", "train", "text", n)
    all_texts += _sample_texts("bigcode/the-stack-smol", None, "train", None, n,
                                text_fn=lambda r: r.get("content", ""))
    all_texts += _sample_texts(
        "HuggingFaceH4/ultrachat_200k", None, "train_sft", None, n,
        text_fn=lambda r: " ".join(m["content"] for m in r.get("messages", []) if "content" in m),
    )
    all_texts += _sample_texts(
        "Open-Orca/OpenOrca", None, "train", None, n,
        text_fn=lambda r: f"{r.get('question', '')} {r.get('response', '')}",
    )
    all_texts += _sample_texts(
        "ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train", None, n,
        text_fn=lambda r: f"{r.get('problem', '')} {r.get('solution', '')}",
    )

    print(f"\ntotal sampled docs: {len(all_texts)}")
    print("tokenizing...")
    # Raw base-vocab IDs (unshifted) -- strip the n_special shift that
    # encode_text() adds, since PrunedVocab operates on the base cl100k_base
    # ID space and the wrapper re-applies the shift at encode/decode time.
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


if __name__ == "__main__":
    main()
