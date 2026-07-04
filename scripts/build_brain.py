"""
build_brain.py — build the premade general-knowledge brain shipped as
uchi/data/embeddings.pt.

Fixes a real gap: SemanticIndex._vec() only embeds words already in its
vocabulary (w2i) — it never learns new words from learn()/build_from_corpus().
Without a shipped vocabulary, a fresh Uchi() has an EMPTY w2i, so learn() is a
silent no-op and every question abstains (no words known -> _known_fraction=0).

This script:
  1. Loads the existing skip-gram vocabulary (uchi/data/skipgram_emb.pt, 46K
     words trained by experiments/skipgram_probe.py over Wikipedia+MMLU+ARC).
  2. Streams a broad, diverse sample of Wikipedia article openings (breadth
     over depth — general knowledge, not deep dives) and embeds them as
     passages against that vocabulary via SemanticIndex.build_from_corpus.
  3. Saves {w2i, E, dim, passages, P} to uchi/data/embeddings.pt so a fresh
     Uchi() can retrieve real general knowledge with zero learn() calls.

Usage:
    .venv/bin/python scripts/build_brain.py --articles 20000
"""
import argparse
import os
import time

import torch
import numpy as np

from uchi.retrieval import SemanticIndex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", type=int, default=20_000)
    ap.add_argument("--chars-per-article", type=int, default=500,
                     help="How much of each article's opening text to use (breadth over depth)")
    ap.add_argument("--vocab-src", default="uchi/data/skipgram_emb.pt")
    ap.add_argument("--out", default="uchi/data/embeddings.pt")
    args = ap.parse_args()

    print(f"Loading vocabulary from {args.vocab_src} ...")
    vd = torch.load(args.vocab_src, map_location="cpu")
    E = torch.nn.functional.normalize(vd["E"], p=2, dim=-1).numpy().astype(np.float32)
    w2i = vd["w2i"]
    print(f"  vocab size: {len(w2i):,}  dim: {E.shape[1]}")

    idx = SemanticIndex(w2i, E)

    from datasets import load_dataset
    print("Streaming Wikipedia for general-knowledge passages ...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    # Collect all article snippets FIRST, embed in one batch at the end.
    # SemanticIndex.add() does np.vstack on every call — calling it once per
    # article (as build_from_corpus does internally) re-copies the whole,
    # ever-growing passage matrix each time: O(N^2) over N articles. A single
    # batched call is O(N) — this is what turned a ~2s trial into a 12-minute
    # stall once the corpus grew large enough for the copies to dominate.
    t0 = time.time()
    n_articles = 0
    chunks = []
    for row in ds:
        text = row["text"][: args.chars_per_article]
        if len(text) < 40:
            continue
        chunks.append(text)
        n_articles += 1
        if n_articles % 5000 == 0:
            print(f"  streamed {n_articles:,}/{args.articles:,} articles ({time.time()-t0:.0f}s)")
        if n_articles >= args.articles:
            break
    print(f"Streamed {n_articles:,} articles in {time.time()-t0:.0f}s. Embedding (single batch) ...")

    t1 = time.time()
    # Join with a separator that can't merge sentences across articles.
    idx.build_from_corpus("\n\n".join(chunks))
    print(f"Embedded -> {len(idx):,} passages in {time.time()-t1:.0f}s")
    print(f"Done: {n_articles:,} articles -> {len(idx):,} passages in {time.time()-t0:.0f}s total")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "w2i": w2i,
        "E": torch.from_numpy(E),
        "dim": E.shape[1],
        "passages": idx.passages,
        "P": torch.from_numpy(idx._P) if idx._P is not None else None,
    }, args.out)
    print(f"Saved brain seed -> {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
