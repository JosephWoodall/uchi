"""
mmlu_benchmark.py
=================
Evaluates Uchi on the Massive Multitask Language Understanding (MMLU) benchmark.

MMLU tests OOD generalization across 57 academic subjects via 4-choice
multiple-choice questions. This is the primary exit criterion for v0.3.0:
improvements to OOD synthesis (HRR fallback, goal tracking, internal dialogue)
must move the MMLU score. If the score doesn't move, the release doesn't ship.

Scoring:
  Each question is formatted as a multiple-choice prompt. Uchi's response is
  parsed for A/B/C/D letter selection or answer-text match. Accuracy is
  reported overall and per-subject.

  Note: Uchi is a sequence predictor, not a zero-shot LLM. At baseline,
  MMLU accuracy reflects what Uchi has in its brain. The score is expected
  to be low before v0.3.0 architectural changes land — that is the point.
  Run this first to establish the number, then again after each deliverable
  to track progress.

Usage:
    python benchmarks/mmlu_benchmark.py
    python benchmarks/mmlu_benchmark.py --sample 500
    python benchmarks/mmlu_benchmark.py --sample 0          # full 14,042 questions
    python benchmarks/mmlu_benchmark.py --subjects abstract_algebra,astronomy
    python benchmarks/mmlu_benchmark.py --brain path/to/brain.uchi
    python benchmarks/mmlu_benchmark.py --out results/mmlu_v030_baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import contextlib
import sys
import time
from collections import defaultdict
from typing import Optional


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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_W = 80
_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
_DEFAULT_OUT  = os.path.join(_RESULTS_DIR, "mmlu_results.json")

_CHOICE_LABELS = ["A", "B", "C", "D"]


# ── prompt formatting ─────────────────────────────────────────────────────────

def _format_prompt(question: str, choices: list[str], subject: str) -> str:
    subject_fmt = subject.replace("_", " ")
    lines = [
        f"The following is a multiple choice question about {subject_fmt}.",
        "",
        question,
        "",
    ]
    for label, choice in zip(_CHOICE_LABELS, choices):
        lines.append(f"{label}. {choice}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


# ── answer parsing ────────────────────────────────────────────────────────────

def _parse_answer(response: str, choices: list[str]) -> Optional[int]:
    """Return the 0-indexed choice selected by Uchi's response, or None."""
    if not response:
        return None

    r = response.strip()

    # 1. Look for explicit letter at the start: "A", "A.", "A)", "(A)"
    letter_match = re.match(r"^\(?([A-Da-d])[.):\s]", r)
    if letter_match:
        return _CHOICE_LABELS.index(letter_match.group(1).upper())

    # 2. Look for "the answer is A" / "answer: B" patterns
    ans_match = re.search(r"(?:answer\s*(?:is)?[:\s]*)\(?([A-Da-d])\)?", r, re.IGNORECASE)
    if ans_match:
        return _CHOICE_LABELS.index(ans_match.group(1).upper())

    # 3. Look for any standalone letter A-D in the response
    letter_anywhere = re.findall(r"\b([A-Da-d])\b", r)
    if letter_anywhere:
        return _CHOICE_LABELS.index(letter_anywhere[0].upper())

    # 4. Check if response text is a substring of one of the choices
    r_lower = r.lower()
    for i, choice in enumerate(choices):
        if choice.lower() in r_lower or r_lower in choice.lower():
            return i

    return None


# ── main benchmark ────────────────────────────────────────────────────────────

def run_mmlu(router, sample: int, subjects: Optional[list[str]], verbose: bool) -> dict:
    from datasets import load_dataset

    print("  Loading MMLU dataset…")
    ds = load_dataset("cais/mmlu", "all", split="test")

    if subjects:
        ds = ds.filter(lambda x: x["subject"] in subjects)
        print(f"  Filtered to {len(ds)} questions across subjects: {subjects}")

    if sample and sample < len(ds):
        import random
        indices = random.sample(range(len(ds)), sample)
        ds = ds.select(indices)
        print(f"  Sampled {len(ds)} questions from {len(subjects) if subjects else 57} subjects")
    else:
        print(f"  Running full set: {len(ds)} questions")

    total = 0
    correct = 0
    no_parse = 0
    by_subject: dict[str, list[bool]] = defaultdict(list)

    t0 = time.time()
    for i, row in enumerate(ds):
        question  = row["question"]
        choices   = row["choices"]
        answer    = row["answer"]   # int 0-3
        subject   = row["subject"]

        prompt = _format_prompt(question, choices, subject)

        try:
            with _no_web():
                response = router.chat(prompt) or ""
        except Exception as e:
            response = ""
            if verbose:
                print(f"    [!] chat() error on q{i}: {e}")

        predicted = _parse_answer(response, choices)
        hit = predicted == answer

        total += 1
        if hit:
            correct += 1
        if predicted is None:
            no_parse += 1
        by_subject[subject].append(hit)

        if verbose or (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            acc = correct / total if total else 0
            print(f"    [{i+1:>5}/{len(ds)}] acc={acc:.3f}  elapsed={elapsed:.0f}s"
                  + (f"  | Q: {question[:60]}… → {response[:40]}…" if verbose else ""))

    elapsed = time.time() - t0
    accuracy = correct / total if total else 0.0

    # per-subject summary
    subject_scores = {
        subj: sum(hits) / len(hits)
        for subj, hits in by_subject.items()
    }
    worst5 = sorted(subject_scores.items(), key=lambda x: x[1])[:5]
    best5  = sorted(subject_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    print(f"\n  {'─'*_W}")
    print(f"  MMLU Results")
    print(f"  {'─'*_W}")
    print(f"  Questions : {total}")
    print(f"  Correct   : {correct}")
    print(f"  Accuracy  : {accuracy:.3f}  ({accuracy*100:.1f}%)")
    print(f"  No-parse  : {no_parse}  ({no_parse/total*100:.1f}% — response not parseable as A/B/C/D)")
    print(f"  Time      : {elapsed:.1f}s  ({elapsed/total*1000:.0f}ms/q)")
    print(f"  Random baseline: 25.0%")
    print(f"\n  Best subjects:")
    for subj, sc in best5:
        print(f"    {subj:<45} {sc*100:5.1f}%")
    print(f"\n  Worst subjects:")
    for subj, sc in worst5:
        print(f"    {subj:<45} {sc*100:5.1f}%")
    print(f"  {'─'*_W}")

    return {
        "mmlu_accuracy":        round(accuracy, 4),
        "mmlu_correct":         correct,
        "mmlu_total":           total,
        "mmlu_no_parse":        no_parse,
        "mmlu_elapsed_s":       round(elapsed, 1),
        "mmlu_ms_per_question": round(elapsed / total * 1000, 1) if total else 0,
        "mmlu_by_subject":      {k: round(v, 4) for k, v in subject_scores.items()},
        "mmlu_subjects_tested": list(subject_scores.keys()),
    }


# ── entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Uchi MMLU Benchmark")
    parser.add_argument("--sample", type=int, default=200,
                        help="Number of questions to sample (0 = full 14,042, default 200)")
    parser.add_argument("--subjects", type=str, default=None,
                        help="Comma-separated MMLU subjects to test (default: all 57)")
    parser.add_argument("--brain", default="brain.uchi",
                        help="Brain file path (default: brain.uchi)")
    parser.add_argument("--out", default=_DEFAULT_OUT,
                        help=f"Output JSON path (default: {_DEFAULT_OUT})")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each question and response")
    args = parser.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",")] if args.subjects else None
    sample   = args.sample if args.sample > 0 else 0

    print("\n" + "="*_W)
    print(" Uchi MMLU Benchmark — OOD Generalization Baseline")
    print("="*_W + "\n")

    import gzip, pickle
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

    results = run_mmlu(router, sample=sample, subjects=subjects, verbose=args.verbose)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {args.out}")


if __name__ == "__main__":
    main()
