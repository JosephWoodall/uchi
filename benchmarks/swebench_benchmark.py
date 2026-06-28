"""
swebench_benchmark.py
=====================
Evaluates Uchi on SWE-bench — a benchmark of real GitHub issues requiring
code patches to fix bugs in Python repositories.

IMPORTANT — What this harness measures:
  Full SWE-bench evaluation (% issues resolved) requires executing patch
  candidates against real test suites in containerized environments. That
  infrastructure is out of scope here. This harness measures a code generation
  quality proxy that is tractable to run locally:

    1. Code Generation Rate   — did Uchi produce any code at all?
    2. Syntax Validity        — is the generated Python syntactically valid?
    3. Patch Keyword Overlap  — does the code share vocabulary with the
                                expected patch (functions, variables, modules)?
    4. Problem Coverage       — does the response address words from the
                                problem statement?

  These proxy metrics establish a baseline for Uchi's code synthesis capability
  before v0.3.0 changes land. As the HRR fallback (item 7) and internal
  dialogue (item 9) improve, these scores should rise. The proxy is not a
  substitute for real SWE-bench evaluation; it is a trackable signal.

Usage:
    python benchmarks/swebench_benchmark.py
    python benchmarks/swebench_benchmark.py --sample 100
    python benchmarks/swebench_benchmark.py --sample 0        # full 2,294 instances
    python benchmarks/swebench_benchmark.py --brain path/to/brain.uchi
    python benchmarks/swebench_benchmark.py --out results/swe_baseline.json
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import re
import sys
import time
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
_DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "swebench_results.json")


# ── scoring helpers ───────────────────────────────────────────────────────────

def _extract_code_blocks(text: str) -> list[str]:
    """Pull fenced code blocks and bare indented blocks from a response."""
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if not blocks:
        # Fall back: any line starting with 4 spaces or a tab
        indented = [
            line for line in text.splitlines()
            if line.startswith("    ") or line.startswith("\t")
        ]
        if indented:
            blocks = ["\n".join(indented)]
    return [b.strip() for b in blocks if b.strip()]


def _is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _keyword_overlap(response: str, patch: str) -> float:
    """Fraction of patch identifiers that appear in the response."""
    identifiers = set(re.findall(r"\b[a-zA-Z_]\w{2,}\b", patch))
    # Filter out very common Python keywords that aren't meaningful
    stop = {"def", "return", "import", "from", "class", "self", "None",
            "True", "False", "pass", "raise", "except", "try", "with",
            "for", "while", "if", "else", "elif", "and", "not", "or",
            "print", "str", "int", "list", "dict", "set", "len"}
    identifiers = identifiers - stop
    if not identifiers:
        return 0.0
    matched = sum(1 for kw in identifiers if kw in response)
    return matched / len(identifiers)


def _problem_coverage(response: str, problem: str) -> float:
    """Fraction of meaningful words from the problem statement in the response."""
    words = set(re.findall(r"\b[a-z_]\w{3,}\b", problem.lower()))
    stop = {"that", "this", "with", "from", "when", "should", "would",
            "could", "have", "been", "will", "which", "there", "their",
            "into", "more", "also", "than", "then", "does", "some"}
    words = words - stop
    if not words:
        return 0.0
    r_lower = response.lower()
    return sum(1 for w in words if w in r_lower) / len(words)


def _score_instance(response: str, patch: str, problem: str) -> dict:
    code_blocks = _extract_code_blocks(response)
    generated_code = "\n\n".join(code_blocks)

    has_code     = len(code_blocks) > 0
    syntax_valid = _is_valid_python(generated_code) if has_code else False
    kw_overlap   = _keyword_overlap(response, patch)
    coverage     = _problem_coverage(response, problem)

    # Composite proxy score: weighted sum
    # syntax_valid carries the most weight — generating broken code is worse than none
    composite = (
        0.25 * float(has_code) +
        0.35 * float(syntax_valid) +
        0.25 * kw_overlap +
        0.15 * coverage
    )

    return {
        "has_code":     has_code,
        "syntax_valid": syntax_valid,
        "kw_overlap":   round(kw_overlap, 3),
        "coverage":     round(coverage, 3),
        "composite":    round(composite, 3),
    }


# ── main benchmark ────────────────────────────────────────────────────────────

def run_swebench(router, sample: int, verbose: bool) -> dict:
    from datasets import load_dataset

    print("  Loading SWE-bench dataset…")
    ds = load_dataset("princeton-nlp/SWE-bench", split="test")

    if sample and sample < len(ds):
        import random
        indices = random.sample(range(len(ds)), sample)
        ds = ds.select(indices)
        print(f"  Sampled {len(ds)} instances")
    else:
        print(f"  Running full set: {len(ds)} instances")

    totals = {
        "has_code": 0,
        "syntax_valid": 0,
        "kw_overlap": 0.0,
        "coverage": 0.0,
        "composite": 0.0,
    }
    n = 0
    by_repo: dict[str, list[float]] = {}

    t0 = time.time()
    for i, row in enumerate(ds):
        repo    = row["repo"]
        problem = row["problem_statement"]
        patch   = row["patch"]

        # Truncate very long problem statements to keep prompts reasonable
        prompt = (
            f"Fix the following bug in the {repo} repository:\n\n"
            + problem[:1500]
            + ("\n\n[problem truncated]" if len(problem) > 1500 else "")
            + "\n\nProvide a Python code fix:"
        )

        try:
            response = router.chat(prompt) or ""
        except Exception as e:
            response = ""
            if verbose:
                print(f"    [!] chat() error on instance {i}: {e}")

        scores = _score_instance(response, patch, problem)
        n += 1

        for k in totals:
            totals[k] += float(scores[k])

        repo_short = repo.split("/")[-1]
        if repo_short not in by_repo:
            by_repo[repo_short] = []
        by_repo[repo_short].append(scores["composite"])

        if verbose or (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            avg = totals["composite"] / n
            print(f"    [{i+1:>4}/{len(ds)}] composite={avg:.3f}  elapsed={elapsed:.0f}s"
                  + (f"\n      repo={repo}  code={scores['has_code']}  "
                     f"syntax={scores['syntax_valid']}  kw={scores['kw_overlap']:.2f}" if verbose else ""))

    elapsed  = time.time() - t0
    averages = {k: round(v / n, 4) for k, v in totals.items()} if n else {}

    repo_avg = {r: round(sum(v)/len(v), 3) for r, v in by_repo.items()}
    worst5 = sorted(repo_avg.items(), key=lambda x: x[1])[:5]
    best5  = sorted(repo_avg.items(), key=lambda x: x[1], reverse=True)[:5]

    print(f"\n  {'─'*_W}")
    print(f"  SWE-bench Code Generation Proxy Results")
    print(f"  {'─'*_W}")
    print(f"  Instances         : {n}")
    print(f"  Code generated    : {int(totals['has_code'])}/{n}  ({totals['has_code']/n*100:.1f}%)")
    print(f"  Syntax valid      : {int(totals['syntax_valid'])}/{n}  ({totals['syntax_valid']/n*100:.1f}%)")
    print(f"  Avg keyword overlap : {averages.get('kw_overlap', 0):.3f}")
    print(f"  Avg problem coverage: {averages.get('coverage', 0):.3f}")
    print(f"  Avg composite score : {averages.get('composite', 0):.3f}")
    print(f"  Time              : {elapsed:.1f}s  ({elapsed/n*1000:.0f}ms/instance)")
    print(f"\n  NOTE: composite score is a proxy metric. Real SWE-bench resolution")
    print(f"  rate requires containerized test execution (out of scope for this harness).")
    print(f"\n  Best repos:")
    for repo, sc in best5:
        print(f"    {repo:<40} {sc:.3f}")
    print(f"\n  Worst repos:")
    for repo, sc in worst5:
        print(f"    {repo:<40} {sc:.3f}")
    print(f"  {'─'*_W}")

    return {
        "swebench_instances":         n,
        "swebench_code_rate":         round(totals["has_code"] / n, 4) if n else 0,
        "swebench_syntax_valid_rate": round(totals["syntax_valid"] / n, 4) if n else 0,
        "swebench_avg_kw_overlap":    averages.get("kw_overlap", 0),
        "swebench_avg_coverage":      averages.get("coverage", 0),
        "swebench_avg_composite":     averages.get("composite", 0),
        "swebench_elapsed_s":         round(elapsed, 1),
        "swebench_ms_per_instance":   round(elapsed / n * 1000, 1) if n else 0,
        "swebench_by_repo":           repo_avg,
    }


# ── entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Uchi SWE-bench Benchmark")
    parser.add_argument("--sample", type=int, default=50,
                        help="Number of instances to sample (0 = full 2,294, default 50)")
    parser.add_argument("--brain", default="brain.uchi",
                        help="Brain file path (default: brain.uchi)")
    parser.add_argument("--out", default=_DEFAULT_OUT,
                        help=f"Output JSON path (default: {_DEFAULT_OUT})")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each instance and scores")
    args = parser.parse_args()

    sample = args.sample if args.sample > 0 else 0

    print("\n" + "="*_W)
    print(" Uchi SWE-bench Benchmark — Code Generation Quality Proxy")
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

    router.web_search_enabled = False
    results = run_swebench(router, sample=sample, verbose=args.verbose)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {args.out}")


if __name__ == "__main__":
    main()
