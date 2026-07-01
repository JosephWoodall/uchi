"""
arc_benchmark.py
================
Evaluates Uchi on the ARC-Challenge benchmark (AI2 Reasoning Challenge).

ARC-Challenge is a 4-choice MCQ benchmark of elementary science questions
designed to require reasoning beyond pattern matching. Unlike MMLU, questions
are specifically selected because they cannot be answered by co-occurrence or
simple retrieval — they require combining multiple concepts.

This is the 0.3.0 reasoning baseline. The 0.4.0 exit criterion is that
Meta-Uchi must not regress from this number when run through the full
agentic facade.

Scoring:
  Prompt format matches arc_benchmark ingestion exactly so trie paths align.
  Accuracy is reported overall and no-parse rate isolates format from knowledge.
  Random baseline is 25% (4-choice).

Usage:
    python benchmarks/arc_benchmark.py
    python benchmarks/arc_benchmark.py --sample 200
    python benchmarks/arc_benchmark.py --sample 0          # full 1,172 questions
    python benchmarks/arc_benchmark.py --brain path/to/brain.uchi
    python benchmarks/arc_benchmark.py --out results/arc_v030_baseline.json
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import os
import pickle
import re
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_W = 80
_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
_DEFAULT_OUT  = os.path.join(_RESULTS_DIR, "arc_results.json")

_LETTER_LABELS = {"A", "B", "C", "D"}


@contextlib.contextmanager
def _no_web():
    try:
        import uchi.web_search as _m
        orig = _m.perform_web_search
        _m.perform_web_search = lambda *a, **kw: ""
        try:
            yield
        finally:
            _m.perform_web_search = orig
    except ImportError:
        yield


# ── prompt formatting ─────────────────────────────────────────────────────────

def _format_prompt(question: str, labels: list[str], texts: list[str]) -> str:
    lines = ["The following is a multiple choice question.", "", question, ""]
    for label, text in zip(labels, texts):
        lines.append(f"{label}. {text}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


# ── answer parsing ────────────────────────────────────────────────────────────

def _parse_answer(response: str, labels: list[str], texts: list[str]) -> Optional[str]:
    """Return the label selected by Uchi's response, or None."""
    if not response:
        return None

    r = response.strip()

    # 1. Explicit letter at start: "A", "A.", "A)", "(A)"
    letter_match = re.match(r"^\(?([A-Da-d])[.):\s]", r)
    if letter_match:
        candidate = letter_match.group(1).upper()
        if candidate in labels:
            return candidate

    # 2. "the answer is A" / "answer: B" patterns
    ans_match = re.search(r"(?:answer\s*(?:is)?[:\s]*)\(?([A-Da-d])\)?", r, re.IGNORECASE)
    if ans_match:
        candidate = ans_match.group(1).upper()
        if candidate in labels:
            return candidate

    # 3. Any standalone letter A-D in response
    letters = re.findall(r"\b([A-Da-d])\b", r)
    for letter in letters:
        candidate = letter.upper()
        if candidate in labels:
            return candidate

    # 4. Response text matches a choice text
    r_lower = r.lower()
    for label, text in zip(labels, texts):
        if text.lower() in r_lower or r_lower in text.lower():
            return label

    return None


# ── main benchmark ────────────────────────────────────────────────────────────

def run_arc(router, sample: int, verbose: bool) -> dict:
    from datasets import load_dataset

    print("  Loading ARC-Challenge dataset…")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

    # Filter to letter-labeled questions only (A/B/C/D) for consistent parsing
    ds = ds.filter(lambda x: set(x["choices"]["label"]).issubset(_LETTER_LABELS))

    if sample and sample < len(ds):
        import random
        indices = random.sample(range(len(ds)), sample)
        ds = ds.select(indices)
        print(f"  Sampled {len(ds)} questions")
    else:
        print(f"  Running full set: {len(ds)} questions")

    total    = 0
    correct  = 0
    no_parse = 0

    t0 = time.time()
    for i, row in enumerate(ds):
        question   = row["question"]
        choices    = row["choices"]
        labels     = choices["label"]
        texts      = choices["text"]
        answer_key = row["answerKey"].strip().upper()

        prompt = _format_prompt(question, labels, texts)

        try:
            with _no_web():
                response = router.chat(prompt) or ""
        except Exception as e:
            response = ""
            if verbose:
                print(f"    [!] chat() error on q{i}: {e}")

        predicted = _parse_answer(response, labels, texts)
        hit = predicted == answer_key

        total += 1
        if hit:
            correct += 1
        if predicted is None:
            no_parse += 1

        if verbose or (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            acc = correct / total if total else 0
            print(f"    [{i+1:>5}/{len(ds)}] acc={acc:.3f}  elapsed={elapsed:.0f}s"
                  + (f"  | Q: {question[:60]}… → {response[:40]}…" if verbose else ""))

    elapsed  = time.time() - t0
    accuracy = correct / total if total else 0.0

    print(f"\n  {'─'*_W}")
    print(f"  ARC-Challenge Results")
    print(f"  {'─'*_W}")
    print(f"  Questions : {total}")
    print(f"  Correct   : {correct}")
    print(f"  Accuracy  : {accuracy:.3f}  ({accuracy*100:.1f}%)")
    print(f"  No-parse  : {no_parse}  ({no_parse/total*100:.1f}% — response not parseable as A/B/C/D)")
    print(f"  Time      : {elapsed:.1f}s  ({elapsed/total*1000:.0f}ms/q)")
    print(f"  Random baseline: 25.0%")
    print(f"  {'─'*_W}")

    return {
        "arc_accuracy":        round(accuracy, 4),
        "arc_correct":         correct,
        "arc_total":           total,
        "arc_no_parse":        no_parse,
        "arc_elapsed_s":       round(elapsed, 1),
        "arc_ms_per_question": round(elapsed / total * 1000, 1) if total else 0,
    }


# ── entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Uchi ARC-Challenge Benchmark")
    parser.add_argument("--sample", type=int, default=200,
                        help="Questions to sample (0 = full test set, default 200)")
    parser.add_argument("--brain", default="brain.uchi",
                        help="Brain file path (default: brain.uchi)")
    parser.add_argument("--out", default=_DEFAULT_OUT,
                        help=f"Output JSON path (default: {_DEFAULT_OUT})")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each question and response")
    args = parser.parse_args()

    sample = args.sample if args.sample > 0 else 0

    print("\n" + "="*_W)
    print(" Uchi ARC-Challenge Benchmark — Reasoning Baseline")
    print("="*_W + "\n")

    from uchi.omni_router import OmniRouter

    brain_path = args.brain
    router = None
    if os.path.exists(brain_path):
        print(f"[*] Loading brain from {brain_path}…")
        try:
            with gzip.open(brain_path, "rb") as f:
                router = pickle.load(f)
        except Exception:
            try:
                with open(brain_path, "rb") as f:
                    router = pickle.load(f)
            except Exception as e:
                print(f"[!] Failed to load brain: {e}")

    if router is None:
        print("[*] No brain loaded — using cold router (bootstrap disabled).")
        OmniRouter._bootstrap_knowledge = lambda self, *a, **kw: None
        OmniRouter._bootstrap_persona   = lambda self, *a, **kw: None
        router = OmniRouter(use_bpe=False)

    router.web_search_enabled = False

    results = run_arc(router, sample=sample, verbose=args.verbose)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {args.out}")


if __name__ == "__main__":
    main()
