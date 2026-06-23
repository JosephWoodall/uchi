#!/usr/bin/env python3
"""
evaluate_omni.py
================
Runner script for the Uchi Omni-Evaluator.

Instantiates a fresh OmniRouter (with persona bootstrap), runs the full
evaluation suite, prints formatted results to stdout, and persists them
to ``eval_metrics.json``.

Usage::

    python evaluate_omni.py                     # default output path
    python evaluate_omni.py --output results.json
    python evaluate_omni.py --quiet             # suppress per-challenge logs
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Uchi Omni-Evaluator suite.",
    )
    parser.add_argument(
        "-o", "--output",
        default="eval_metrics.json",
        help="Path to the JSON file where results are appended (default: eval_metrics.json)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress per-challenge diagnostic output.",
    )
    args = parser.parse_args()

    # ── Bootstrap the router ──────────────────────────────────────────────
    print("⏳  Bootstrapping OmniRouter (persona training) …")
    t0 = time.perf_counter()

    from uchi.omni_router import OmniRouter
    router = OmniRouter()
    
    import os
    persona_path = os.path.join(os.path.dirname(__file__), "persona.txt")
    if os.path.exists(persona_path):
        with open(persona_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Split by <|user|> but keep the token
            blocks = content.split("<|user|>")
            for block in blocks:
                if block.strip():
                    router.chat("<|user|> " + block.strip())

    elapsed = time.perf_counter() - t0
    print(f"✅  OmniRouter ready in {elapsed:.1f}s\n")

    # ── Run evaluation ────────────────────────────────────────────────────
    from uchi.omni_evaluator import OmniEvaluator

    evaluator = OmniEvaluator(router, verbose=not args.quiet)
    results = evaluator.run_full_evaluation()

    # ── Persist ───────────────────────────────────────────────────────────
    evaluator.save_metrics(results, args.output)
    print(f"📁  Results appended to {args.output}")


if __name__ == "__main__":
    main()
