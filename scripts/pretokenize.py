"""
pretokenize.py — one-time corpus tokenization to a memmap .bin (nanoGPT-style).

Why: FLUX's Phase-1 loop tokenized text on the fly, single-threaded, while
streaming from HF. That starved the GPU (~185 tok/s vs ~1800 tok/s it can
sustain). Pre-tokenizing once removes tokenization + streaming from the hot
loop entirely, so training becomes GPU-bound.

Output (uint32, because vocab=100300 > uint16 range):
    uchi/flux/data/train.bin
    uchi/flux/data/val.bin

Token stream is a flat concatenation of documents separated by <|eos|>, exactly
matching TikTokenHybridTokenizer.encode_text ids (cl100k shifted by n_special).

Usage:
    .venv/bin/python -m scripts.pretokenize --train-tokens 80_000_000 --val-tokens 1_000_000
"""
import argparse
import os
import time

import numpy as np

from uchi.flux.tokenizer_v2 import TikTokenHybridTokenizer

OUT_DIR = os.path.join("uchi", "flux", "data")


def _text_stream(split_skip=0, split_take=None):
    """Yield raw document texts from FineWeb-Edu (educational-filtered web).

    FineWeb-Edu beats raw OpenWebText per-token at this model scale (the Phi /
    'textbooks' result): quality-per-token dominates for small models. It already
    spans the general/educational knowledge Wikipedia was added for, so we use it
    as the single source (train and val now share one distribution → clean val
    signal). Falls back to OpenWebText if FineWeb-Edu is unreachable.
    """
    from datasets import load_dataset

    def _load():
        try:
            ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                              split="train", streaming=True)
            print("  corpus: FineWeb-Edu (sample-10BT)")
            return ds
        except Exception as e:
            print(f"  [!] FineWeb-Edu unavailable ({type(e).__name__}); "
                  f"falling back to OpenWebText. {e}")
            return load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    ds = _load()
    if split_take is not None:                    # validation carve-out (corpus head)
        for row in ds.take(split_take):
            yield row["text"]
        return
    for row in ds.skip(split_skip):               # train tail (disjoint from val head)
        yield row["text"]


def build_bin(out_path, n_tokens, tok, split_skip=0, split_take=None, batch_docs=512):
    """Tokenize until n_tokens are written to out_path as a uint32 memmap."""
    eos = tok.eos_token_id
    shift = tok.n_special
    enc = tok._enc                                # raw tiktoken (batch, multithreaded)

    arr = np.memmap(out_path, dtype=np.uint32, mode="w+", shape=(n_tokens,))
    written = 0
    t0 = time.time()
    buf_texts = []

    def flush(texts):
        nonlocal written
        if not texts:
            return
        # encode_ordinary_batch releases the GIL and uses all cores (Rust).
        for ids in enc.encode_ordinary_batch(texts):
            if not ids:
                continue
            seq = np.fromiter((t + shift for t in ids), dtype=np.uint32, count=len(ids))
            n = min(len(seq), n_tokens - written)
            if n > 0:
                arr[written:written + n] = seq[:n]
                written += n
            if written < n_tokens:                # doc separator
                arr[written] = eos
                written += 1

    for text in _text_stream(split_skip, split_take):
        if not text or not text.strip():
            continue
        buf_texts.append(text)
        if len(buf_texts) >= batch_docs:
            flush(buf_texts); buf_texts = []
            if written % (2_000_000) < batch_docs * 8:
                rate = written / max(time.time() - t0, 1e-6)
                print(f"    {written:,}/{n_tokens:,} tokens  ({rate:,.0f} tok/s)")
            if written >= n_tokens:
                break
    if written < n_tokens:
        flush(buf_texts)

    arr.flush()
    dt = time.time() - t0
    print(f"  ✔ {out_path}: {written:,} tokens in {dt:.0f}s ({written/max(dt,1e-6):,.0f} tok/s)")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tokens", type=int, default=80_000_000)
    ap.add_argument("--val-tokens", type=int, default=1_000_000)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = TikTokenHybridTokenizer()
    print(f"Tokenizer vocab={tok.vocab_size}, n_special={tok.n_special}, eos={tok.eos_token_id}")

    # Validation = first 20k docs; train skips them so the sets are disjoint.
    print("Building val.bin (held-out corpus head) ...")
    build_bin(os.path.join(args.out_dir, "val.bin"), args.val_tokens, tok,
              split_take=20_000)
    print("Building train.bin (corpus tail) ...")
    build_bin(os.path.join(args.out_dir, "train.bin"), args.train_tokens, tok,
              split_skip=20_000)
    print("Done. Point train_v2 at these with --data-bin / --val-bin.")


if __name__ == "__main__":
    main()
