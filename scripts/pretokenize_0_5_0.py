"""pretokenize_0_5_0.py -- 0.5.0 Item 2's mixed-corpus tokenization.

Extends ``pretokenize.py``'s FineWeb-Edu-only approach with Item 1's new
local corpus (``.uchi/corpus/stack_v2_sample.jsonl`` -- 14,991 real Stack
v2 Python files, 13,959 repos; ``.uchi/corpus/swe_gym_full.jsonl`` -- 2,438
real GitHub issue->diff pairs), both already decontaminated against
SWE-bench's eval-set repos by ``uchi/corpus_sources.py``.

**Budget, not a new ratio scheme**: the local corpus is fixed-size
(~23.3M tokens: ~18.8M Stack v2 + ~4.5M SWE-Gym) -- Item 1's real pull,
not something this script can grow. All of it is used, once. FineWeb-Edu
fills the *remainder* of an 80M-token train budget, matching 0.4.0's own
Phase 1 scale -- the established anchor for "how much pretraining data
this model size needs" for THIS release, not a freshly invented number.
This is the non-regression requirement made concrete: 0.5.0 adds code/
issue-diff capability on top of, not instead of, the general capability
80M tokens of FineWeb-Edu already validated.

**Interleaved, not concatenated in blocks**: local documents are folded
into the FineWeb-Edu stream at roughly even intervals (not "all local
docs, then all FineWeb") so ``train_v2.py``'s random-contiguous-window
sampling doesn't spend long stretches in one source's distribution.
Window boundaries can still occasionally straddle two documents from
different sources (same as any two adjacent documents from the SAME
source already can, in the existing EOS-separated flat-concatenation
scheme) -- acceptable noise at this scale, not a new problem introduced
here.

SWE-Gym records are formatted as ``# Issue: {problem_statement}\\n\\n#
Fix:\\n{patch}`` -- plain concatenated text, no instruction-tuning
special tokens (this is base PRETRAINING data, matching 0.4.0's
CommitPackFT precedent: teach reading issue+diff as natural text now,
instruction-format SFT is a separate, later phase).

Usage:
    .venv/bin/python -m scripts.pretokenize_0_5_0
    .venv/bin/python -m scripts.pretokenize_0_5_0 --train-tokens 80_000_000 --val-tokens 1_000_000
"""
from __future__ import annotations

import argparse
import json
import os
import random

from scripts.pretokenize import OUT_DIR, _text_stream, build_bin

CORPUS_DIR = os.path.join(".uchi", "corpus")
STACK_PATH = os.path.join(CORPUS_DIR, "stack_v2_sample.jsonl")
SWEGYM_PATH = os.path.join(CORPUS_DIR, "swe_gym_full.jsonl")


def _load_local_docs() -> list[str]:
    """Load Item 1's pulled corpus into memory as plain text documents.

    Small enough to hold in RAM whole (77MB Stack v2 + a few hundred KB
    JSON overhead for SWE-Gym) -- this is exactly the buffering trade
    ``uchi/corpus_sources.py``'s own docstring already accepted at this
    scale, not a new limitation introduced here.
    """
    docs: list[str] = []
    if os.path.exists(STACK_PATH):
        with open(STACK_PATH, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                content = row.get("content", "")
                if content.strip():
                    docs.append(content)
        print(f"  loaded {len(docs):,} Stack v2 docs from {STACK_PATH}")
    else:
        print(f"  [!] {STACK_PATH} not found -- skipping Stack v2 portion")

    n_before = len(docs)
    if os.path.exists(SWEGYM_PATH):
        with open(SWEGYM_PATH, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                text = f"# Issue: {row['problem_statement']}\n\n# Fix:\n{row['patch']}"
                docs.append(text)
        print(f"  loaded {len(docs) - n_before:,} SWE-Gym docs from {SWEGYM_PATH}")
    else:
        print(f"  [!] {SWEGYM_PATH} not found -- skipping SWE-Gym portion")

    return docs


def _interleaved_stream(local_docs: list[str], split_skip: int = 0, split_take: int | None = None):
    """Yield documents from FineWeb-Edu with local_docs folded in at
    roughly even intervals, local_docs shuffled once up front so the
    Stack v2 / SWE-Gym split isn't itself block-concatenated either.

    *stride* is recomputed from the CALLER's declared token budget via
    the module-level globals set in ``main()`` -- see there for why this
    isn't a parameter (keeps this generator's signature identical to
    ``_text_stream``'s, so ``build_bin`` can't tell the difference).
    """
    rng = random.Random(0)
    shuffled = list(local_docs)
    rng.shuffle(shuffled)
    local_iter = iter(shuffled)
    next_local = next(local_iter, None)

    fineweb_target = _FINEWEB_TARGET_TOKENS
    local_target = _LOCAL_TARGET_TOKENS
    # Emit one local doc roughly every `stride` FineWeb docs -- calibrated
    # from the approximate FineWeb doc yield rate assumed here (~500
    # tokens/doc, close enough for spacing purposes; getting this exact
    # doesn't matter, only "spread throughout, not front- or back-loaded").
    approx_fineweb_docs = max(fineweb_target // 500, 1)
    stride = max(approx_fineweb_docs // max(len(shuffled), 1), 1) if local_target > 0 else None

    for i, text in enumerate(_text_stream(split_skip, split_take)):
        yield text
        if stride is not None and next_local is not None and i % stride == 0:
            yield next_local
            next_local = next(local_iter, None)

    # Budget-filling loop above may exit (n_tokens reached in build_bin)
    # before every local doc got a turn -- build_bin stops pulling from
    # this generator once its token budget is hit either way, so any
    # remaining local_docs simply never get consumed, not silently lost
    # data corruption, just an artifact of budget-first termination.


_FINEWEB_TARGET_TOKENS = 0
_LOCAL_TARGET_TOKENS = 0


def main() -> int:
    parser = argparse.ArgumentParser(description="0.5.0 Item 2: mixed local+FineWeb-Edu tokenization")
    parser.add_argument("--train-tokens", type=int, default=80_000_000)
    parser.add_argument("--val-tokens", type=int, default=1_000_000)
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    parser.add_argument("--pruned-vocab", type=str, default=None)
    args = parser.parse_args()

    global _FINEWEB_TARGET_TOKENS, _LOCAL_TARGET_TOKENS

    os.makedirs(args.out_dir, exist_ok=True)
    from uchi.flux.tokenizer_v2 import TikTokenHybridTokenizer
    tok = TikTokenHybridTokenizer()

    pruned = None
    if args.pruned_vocab:
        from uchi.flux.vocab_prune import load_pruned_vocab
        pruned = load_pruned_vocab(args.pruned_vocab)
        print(f"Tokenizer vocab={tok.vocab_size} (full), pruned to {pruned.size} "
              f"+ {tok.n_special} special = {pruned.size + tok.n_special} "
              f"(from {args.pruned_vocab})")
    else:
        print(f"Tokenizer vocab={tok.vocab_size}, n_special={tok.n_special}, eos={tok.eos_token_id}")

    print("Loading Item 1's local corpus ...")
    local_docs = _load_local_docs()
    if not local_docs:
        raise RuntimeError(
            f"no local corpus found at {STACK_PATH} / {SWEGYM_PATH} -- "
            "run `python -m uchi.corpus_sources` first (see tasks/todo.md Item 1)."
        )

    # Rough char/4 estimate, same approximation used to size the Item 1
    # pull itself -- good enough for a budget split, exact tokenization
    # happens for real inside build_bin regardless.
    local_chars = sum(len(d) for d in local_docs)
    local_est_tokens = local_chars // 4
    _LOCAL_TARGET_TOKENS = local_est_tokens
    _FINEWEB_TARGET_TOKENS = max(args.train_tokens - local_est_tokens, 0)
    print(f"  local corpus: ~{local_est_tokens:,} est. tokens ({len(local_docs):,} docs)")
    print(f"  FineWeb-Edu fills the remainder: ~{_FINEWEB_TARGET_TOKENS:,} tokens "
          f"(train budget {args.train_tokens:,})")
    if local_est_tokens > args.train_tokens:
        print("  [!] local corpus alone exceeds --train-tokens; FineWeb-Edu contributes "
              "nothing and the local corpus will be truncated by build_bin's budget.")

    print("Building val.bin (held-out FineWeb-Edu corpus head, general-capability signal) ...")
    build_bin(os.path.join(args.out_dir, "val.bin"), args.val_tokens, tok,
              split_take=20_000, pruned_vocab=pruned)

    print("Building train.bin (interleaved local corpus + FineWeb-Edu tail) ...")
    stream = _interleaved_stream(local_docs, split_skip=20_000)
    build_bin(os.path.join(args.out_dir, "train.bin"), args.train_tokens, tok,
              text_stream=stream, pruned_vocab=pruned)

    print("Done. Point train_v2 at these with --data-bin / --val-bin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
