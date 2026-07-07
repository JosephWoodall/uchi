#!/usr/bin/env python3
"""benchmark_mcts_latency.py -- real-hardware latency benchmark for the
sentence-level MCTS verifier cascade, BEFORE the PUCT loop gets written.

Scoring rule pinned down for this benchmark (and for the real
implementation): the Verifier always scores the FULL ACCUMULATED PATH
(s_1..t), never the isolated new sentence alone. Scoring just the
increment would swap the fragment/MNLI-mismatch problem for a
coreference mismatch ("It was built in 1889" has no antecedent without
the prior sentence) -- MNLI/SNLI hypotheses are self-contained
propositions, and the accumulated path is what actually matches that
distribution.

This benchmarks the WORST CASE: full exhaustive expansion of a
depth x branching tree, no PUCT pruning. Real PUCT search only ever
calls the proposer/verifier fewer times than this (it prunes/selects
rather than expanding everything), so this number is a safe upper
bound to size K against -- measure this first, write the smarter
search second.

Forced to CPU (device="cpu" explicitly, not just relying on
CUDA_VISIBLE_DEVICES) so this can run safely while FLUX proposer
training is using the GPU -- these numbers are NOT the authoritative
real-world latency (CPU inference is much slower than GPU), they
validate the benchmark harness and give relative depth/branching
scaling behavior. Re-run with device="cuda" once the GPU is free for
the number that actually matters.
"""
from __future__ import annotations

import argparse
import os
import time

import torch


def build_dummy_verifier(vocab_size: int = 100300, d_model: int = 256, n_layers: int = 8):
    """Random-weight EntailmentClassifier of the real production shape.

    Latency is a function of architecture (forward-pass FLOPs), not of
    whether the weights are trained -- a random-weight model of the
    same shape has identical compute cost to the trained one. Item 17's
    real verifier_best.pt doesn't exist yet (training deferred until
    Phase 3/4 free the GPU), so this is the honest way to benchmark
    real latency now without waiting on that. vocab_size defaults to the
    FULL tokenizer's 100,300 -- the verifier deliberately uses the full,
    not pruned, vocab (its embedding table is separate from FLUX's).
    """
    from uchi.flux.verifier_model import EntailmentClassifier
    model = EntailmentClassifier(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers)
    model.eval()
    return model


def score_path(verifier, tokenizer, path_text: str, evidence_text: str, device: str) -> float:
    # Same premise/hypothesis encoding as EntailmentChecker._encode -- the
    # real verifier's actual tokenization pattern, not an ad hoc one.
    context_open = tokenizer.encode_special("<|context|>")
    context_close = tokenizer.encode_special("<|/context|>")
    user_id = tokenizer.encode_special("<|user|>")
    ids = ([context_open] + tokenizer.encode_text(evidence_text) + [context_close]
           + [user_id] + tokenizer.encode_text(path_text))
    ids = ids[:256]
    x = torch.tensor([ids], device=device).long()
    mask = torch.ones_like(x, dtype=torch.float32)
    with torch.no_grad():
        pooled = verifier.encode(x, mask)
        logits = verifier.classifier_head(pooled)
    return float(torch.softmax(logits, dim=-1)[0, 0].item())  # entailment prob, value proxy


def run_benchmark(depth: int, branching: int, device: str, checkpoint: str | None):
    from uchi.flux.inference_engine import build_generate_fn
    from uchi.flux.tokenizer_v2 import TikTokenHybridTokenizer

    print(f"Loading FLUX proposer on device={device} ...")
    t_load0 = time.perf_counter()
    generate_fn = build_generate_fn(checkpoint=checkpoint, device=device)
    tokenizer = TikTokenHybridTokenizer()
    verifier = build_dummy_verifier().to(device)
    t_load1 = time.perf_counter()
    print(f"  load time: {t_load1 - t_load0:.2f}s")

    evidence_text = "The Eiffel Tower is a wrought-iron tower in Paris, completed in 1889."
    root_prompt = "Context:\n" + evidence_text + "\n\nQuestion: Tell me about the Eiffel Tower.\nAnswer:"

    proposer_calls = 0
    verifier_calls = 0
    proposer_time = 0.0
    verifier_time = 0.0

    def expand(path_text: str, remaining_depth: int):
        nonlocal proposer_calls, verifier_calls, proposer_time, verifier_time
        if remaining_depth == 0:
            return
        for _ in range(branching):
            t0 = time.perf_counter()
            next_sentence = generate_fn(root_prompt + " " + path_text, 24) or "It is a landmark."
            t1 = time.perf_counter()
            proposer_calls += 1
            proposer_time += (t1 - t0)

            new_path = (path_text + " " + next_sentence).strip()

            t2 = time.perf_counter()
            score_path(verifier, tokenizer, new_path, evidence_text, device)
            t3 = time.perf_counter()
            verifier_calls += 1
            verifier_time += (t3 - t2)

            expand(new_path, remaining_depth - 1)

    t_start = time.perf_counter()
    expand("", depth)
    t_end = time.perf_counter()

    total = t_end - t_start
    print(f"\nTree: depth={depth} branching={branching} "
          f"(expected nodes = {sum(branching**i for i in range(1, depth + 1))})")
    print(f"  proposer calls: {proposer_calls}  total {proposer_time:.2f}s  "
          f"avg {1000 * proposer_time / max(1, proposer_calls):.1f}ms/call")
    print(f"  verifier calls: {verifier_calls}  total {verifier_time:.2f}s  "
          f"avg {1000 * verifier_time / max(1, verifier_calls):.1f}ms/call")
    print(f"  TOTAL WALL TIME: {total:.2f}s")
    print(f"\n(device={device} -- {'CPU numbers, NOT representative of real deployment latency; '
          're-run with --device cuda once the GPU is free for the authoritative number.' if device == 'cpu' else 'authoritative GPU measurement.'})")
    return {
        "depth": depth, "branching": branching, "device": device,
        "proposer_calls": proposer_calls, "verifier_calls": verifier_calls,
        "proposer_time": proposer_time, "verifier_time": verifier_time,
        "total_wall_time": total,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--branching", type=int, default=3)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # belt-and-suspenders: refuse the GPU entirely

    run_benchmark(args.depth, args.branching, args.device, args.checkpoint)
