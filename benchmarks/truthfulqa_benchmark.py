"""truthfulqa_benchmark.py — 0.5.0 Item 9: TruthfulQA baseline, the one
benchmark this item's checklist named that was never started
(`benchmarks/trustworthiness.py` is a different suite entirely — SQuAD-based
Generate-and-Ground trust metrics, not TruthfulQA).

Modeled on `humaneval_benchmark.py`/`swebench_real_eval.py`'s structure
(argparse CLI, `--sample`/`--checkpoint`, JSON results file) — the current
FLUX-based real-execution/real-scoring template.

Uses TruthfulQA's **MC1 multiple-choice formulation** (`mc1_targets`), not
the original paper's free-generation + GPT-judge variant. That variant needs
another LLM to grade truthfulness — this project has a hard, explicit
no-LLM constraint (a from-scratch, ~64M-param model, never an LLM anywhere
in the pipeline), so an LLM-judge grader is a non-starter here regardless of
its popularity elsewhere. MC1 is exact-match scorable: exactly one choice
per question is correct, score each choice's log-likelihood under FLUX given
the question, and check whether the argmax is the correct one. No judge.

Scoring reuses `uchi.grpo.sequence_log_prob` (0.5.0 Item 6) directly — it
already computes exactly what's needed here (mean per-token log-prob of a
candidate continuation given a prompt prefix), not reimplemented.

Usage:
    python -m benchmarks.truthfulqa_benchmark --sample 100
    python -m benchmarks.truthfulqa_benchmark --sample 0 --checkpoint path/to/ckpt.pt \\
        --pruned-vocab path/to/vocab.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

DEFAULT_DATASET = "truthfulqa/truthful_qa"
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "truthfulqa_results.json")


def score_question(model, tokenizer, device, question: str, choices: list[str]) -> int:
    """Returns the index of the choice with the highest log-likelihood
    under FLUX given the question as a prompt prefix."""
    from uchi.grpo import sequence_log_prob

    prompt = f"Question: {question}\nAnswer:"
    scores = [
        sequence_log_prob(model, tokenizer, prompt, " " + choice, device=device).item()
        for choice in choices
    ]
    return scores.index(max(scores))


def run(sample: int, dataset_id: str, verbose: bool,
        checkpoint: str | None = None, pruned_vocab: str | None = None,
        device: str | None = None) -> dict:
    from datasets import load_dataset

    from uchi.flux.inference_engine import _load_flux_for_inference

    print(f"  Loading {dataset_id} (multiple_choice) ...")
    ds = load_dataset(dataset_id, "multiple_choice", split="validation")
    if sample and sample < len(ds):
        import random
        ds = ds.select(random.sample(range(len(ds)), sample))
    print(f"  Running {len(ds)} TruthfulQA MC1 question(s)")

    print(f"  Loading FLUX ({checkpoint or 'default flux_best.pt'}, device={device or 'auto'}) ...")
    model, tokenizer, device, *_ = _load_flux_for_inference(checkpoint, device, pruned_vocab)
    model.eval()

    n_correct = 0
    per_question = []
    t0 = time.time()

    for i, row in enumerate(ds):
        question = row["question"]
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        correct_idx = labels.index(1)

        pred_idx = score_question(model, tokenizer, device, question, choices)
        correct = pred_idx == correct_idx
        n_correct += int(correct)
        per_question.append({"question": question, "correct": correct})

        if verbose or (i + 1) % 20 == 0 or (i + 1) == len(ds):
            elapsed = time.time() - t0
            print(f"    [{i+1}/{len(ds)}] {'PASS' if correct else 'fail'}  elapsed={elapsed:.0f}s")

    n = len(per_question)
    elapsed = time.time() - t0
    mc1_accuracy = n_correct / n if n else 0.0

    print(f"\n  {'─'*70}")
    print("  TruthfulQA MC1 Results")
    print(f"  {'─'*70}")
    print(f"  Questions     : {n}")
    print(f"  Correct       : {n_correct}  (MC1 accuracy = {mc1_accuracy*100:.1f}%)")
    if n:
        print(f"  Time          : {elapsed:.1f}s  ({elapsed/n:.1f}s/question)")
    print(f"  {'─'*70}")

    return {
        "dataset": dataset_id, "config": "multiple_choice", "n": n,
        "n_correct": n_correct, "mc1_accuracy": round(mc1_accuracy, 4),
        "elapsed_s": round(elapsed, 1), "per_question": per_question,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Uchi TruthfulQA MC1 Benchmark (0.5.0 Item 9)")
    parser.add_argument("--sample", type=int, default=100, help="Questions to sample (0 = full ~817)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", default=None,
                        help="FLUX checkpoint to evaluate (default: production flux_best.pt)")
    parser.add_argument("--pruned-vocab", default=None,
                        help="Path to the PrunedVocab JSON --checkpoint was trained with. "
                             "build_generate_fn's auto-detection falls back to 0.4.0's "
                             "pruned_vocab_32k.json whenever this is omitted -- silently wrong for "
                             "any 0.5.0 checkpoint using a different pruned vocab of the same size. "
                             "Always pass this explicitly for non-default checkpoints.")
    parser.add_argument("--device", default=None, help="Force cpu/cuda (default: auto-detect)")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    print("\n" + "=" * 70)
    print(" Uchi TruthfulQA MC1 Benchmark (0.5.0 Item 9)")
    print("=" * 70 + "\n")

    results = run(
        args.sample, args.dataset, args.verbose,
        checkpoint=args.checkpoint, pruned_vocab=args.pruned_vocab, device=args.device,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
