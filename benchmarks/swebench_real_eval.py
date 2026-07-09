"""swebench_real_eval.py — REAL SWE-bench execution evaluation (0.5.0 Item 9).

Unlike ``swebench_benchmark.py``'s proxy metric (code-generation quality —
does a response *look* code-shaped — never executes anything, explicitly
scoped that way because "full SWE-bench evaluation... requires executing
patch candidates against real test suites" was out of scope at the time),
this harness runs the actual mechanism stack built this release:
``AgenticRepairAgent`` (``uchi/agentic_repair.py``) attempts each instance
against a real, locally cloned repo (``uchi/repo_fetch.py``), graded by
``ExecutionSandbox``'s real FAIL_TO_PASS/PASS_TO_PASS test execution
(``uchi/execution_sandbox.py``) — the genuine SWE-bench resolution rate,
not a lexical-overlap stand-in.

**Honest expectation, stated once rather than re-litigated per run**:
FLUX (``uchi/flux/checkpoints/flux_best.pt``) was never trained
specifically for multi-file agentic code repair at any real scale —
0.4.0's CommitPackFT SFT slice taught single-file diff *reading*, not
producing a ReAct-formatted tool-use trajectory that ends in a resolving
patch. A near-zero resolution rate here is the expected, honest starting
point this harness exists to measure, not a bug in the harness — see
``tasks/todo.md`` Item 9: "report the real number regardless of where it
lands," same discipline as every prior release. This harness is what makes
that number real instead of a proxy.

Usage:
    python -m benchmarks.swebench_real_eval --sample 3
    python -m benchmarks.swebench_real_eval --sample 20 --dataset SWE-bench/SWE-bench_Lite
"""
from __future__ import annotations

import argparse
import json
import os
import time

from uchi.agentic_repair import AgenticRepairAgent, Outcome
from uchi.execution_sandbox import ExecutionSandbox
from uchi.repo_fetch import RepoFetchError, ensure_local_clone

DEFAULT_DATASET = "SWE-bench/SWE-bench_Lite"
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "swebench_real_results.json")


def _as_list(value) -> list[str]:
    """SWE-bench's FAIL_TO_PASS/PASS_TO_PASS ship as a JSON-encoded string
    column on the HF Hub, not a native list — decode once, here, rather
    than at every call site."""
    if isinstance(value, str):
        return json.loads(value)
    return list(value)


def run(sample: int, dataset_id: str, max_attempts: int, max_iterations: int, verbose: bool) -> dict:
    from datasets import load_dataset

    from uchi.flux.inference_engine import build_generate_fn

    print(f"  Loading {dataset_id} ...")
    ds = load_dataset(dataset_id, split="test")
    if sample and sample < len(ds):
        import random
        ds = ds.select(random.sample(range(len(ds)), sample))
    print(f"  Running {len(ds)} instance(s) through the real execution harness")

    print("  Loading FLUX generate_fn ...")
    generate_fn = build_generate_fn()

    sandbox = ExecutionSandbox(timeout=120.0)
    agent = AgenticRepairAgent(
        generate_fn, sandbox, max_iterations=max_iterations, max_attempts=max_attempts,
    )

    tallies = {Outcome.PASS: 0, Outcome.FAIL: 0, Outcome.ABSTAIN: 0}
    per_instance = []
    t0 = time.time()

    for i, row in enumerate(ds):
        instance_id = row["instance_id"]
        repo, base_commit = row["repo"], row["base_commit"]

        try:
            repo_path = ensure_local_clone(repo, base_commit)
        except RepoFetchError as e:
            tallies[Outcome.ABSTAIN] += 1
            per_instance.append({"instance_id": instance_id, "outcome": "abstain", "note": f"clone failed: {e}"})
            print(f"    [{i+1}/{len(ds)}] {instance_id}: clone failed ({e})")
            continue

        result = agent.repair(
            repo_path=repo_path,
            problem_statement=row["problem_statement"],
            fail_to_pass=_as_list(row["FAIL_TO_PASS"]),
            pass_to_pass=_as_list(row["PASS_TO_PASS"]),
            base_commit=base_commit,
        )
        tallies[result.outcome] += 1
        per_instance.append({
            "instance_id": instance_id, "repo": repo,
            "outcome": result.outcome.value, "attempts": result.attempts,
        })
        if verbose or (i + 1) % 5 == 0 or (i + 1) == len(ds):
            elapsed = time.time() - t0
            print(f"    [{i+1}/{len(ds)}] {instance_id}: {result.outcome.value} "
                  f"(attempts={result.attempts})  elapsed={elapsed:.0f}s")

    n = len(per_instance)
    elapsed = time.time() - t0
    resolve_rate = tallies[Outcome.PASS] / n if n else 0.0

    print(f"\n  {'─'*70}")
    print("  SWE-bench REAL Execution Results")
    print(f"  {'─'*70}")
    print(f"  Instances     : {n}")
    print(f"  Resolved      : {tallies[Outcome.PASS]}  ({resolve_rate*100:.1f}%)")
    print(f"  Failed        : {tallies[Outcome.FAIL]}")
    print(f"  Abstained     : {tallies[Outcome.ABSTAIN]}  (timeout / sandbox error / clone failure)")
    if n:
        print(f"  Time          : {elapsed:.1f}s  ({elapsed/n:.1f}s/instance)")
    print(f"  {'─'*70}")

    return {
        "dataset": dataset_id, "instances": n,
        "resolved": tallies[Outcome.PASS], "failed": tallies[Outcome.FAIL],
        "abstained": tallies[Outcome.ABSTAIN], "resolve_rate": round(resolve_rate, 4),
        "elapsed_s": round(elapsed, 1), "per_instance": per_instance,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Uchi SWE-bench REAL execution evaluation")
    parser.add_argument("--sample", type=int, default=3, help="Instances to sample (0 = full dataset)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    print("\n" + "=" * 70)
    print(" Uchi SWE-bench Benchmark — REAL Execution (0.5.0 Item 9)")
    print("=" * 70 + "\n")

    results = run(args.sample, args.dataset, args.max_attempts, args.max_iterations, args.verbose)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
