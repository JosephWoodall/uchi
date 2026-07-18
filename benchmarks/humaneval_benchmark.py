"""humaneval_benchmark.py — 0.5.0 Item 9: real execution-graded HumanEval
pass@1 -- the benchmark this repo has never had, not even a proxy metric
(unlike SWE-bench, which had `swebench_benchmark.py`'s lexical-overlap
proxy before `swebench_real_eval.py`'s real-execution version).

Modeled directly on `swebench_real_eval.py`'s structure (argparse CLI,
`--sample`/`--checkpoint`, JSON results file) -- the current FLUX-based
real-execution template, not the older brain/trie-based `arc_benchmark.py`.

Dataset: `openai/openai_humaneval` (164 rows: `task_id`, `prompt`,
`canonical_solution`, `test`, `entry_point` -- confirmed by a real load;
the older unnamespaced `openai_humaneval` id no longer resolves on HF).

Prompt + extraction reuse `swebench_benchmark.py`'s established FLUX-facing
code-generation convention rather than reinventing one: an instruction
frame ("Provide the complete function implementation:") and
`_extract_code_blocks` (fenced-block-first, with indentation/def-class
fallback heuristics).

Grading reuses `sandbox_isolation.run_isolated` (0.5.0 Item 5 Stage 2) --
the same isolation primitive `execution_sandbox.py`'s `run_tests` uses for
SWE-bench-shaped test execution, applied here to HumanEval's simpler
single-file shape: `candidate_code + test_code + check(entry_point)`,
executed as one program, pass@1 = exit code 0. Deliberately NOT
`code_engine.REPLOracle.execute()`, which assumes a `def run():`
skill-shaped convention -- the wrong calling shape for HumanEval's
`prompt`-defines-the-function / `test`-defines-`check(candidate)` grading
contract.

Same discipline `swebench_real_eval.py`'s docstring states for SWE-bench:
report the real pass@1 plainly. A near-zero rate against an unfinished or
incoherent checkpoint is the expected, honest starting point this harness
exists to measure, not a bug in the harness.

Usage:
    python -m benchmarks.humaneval_benchmark --sample 10
    python -m benchmarks.humaneval_benchmark --sample 0 --checkpoint path/to/ckpt.pt
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from benchmarks.swebench_benchmark import _extract_code_blocks
from uchi.sandbox_isolation import run_isolated
from uchi.workspace import DEFAULT_ROOT

DEFAULT_DATASET = "openai/openai_humaneval"
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "humaneval_results.json")
_SANDBOX_SUBDIR = "humaneval_sandbox"


def _build_prompt(problem_prompt: str) -> str:
    return (
        "Complete the following Python function:\n\n"
        f"{problem_prompt}\n\n"
        "Provide the complete function implementation:"
    )


def grade_candidate(
    candidate_code: str, test_code: str, entry_point: str,
    timeout: float = 10.0, isolate: bool = True, root: str = DEFAULT_ROOT,
) -> tuple[bool, str]:
    """Runs `candidate_code + test_code + check(entry_point)` as one real
    Python program. Returns (passed, output) -- passed = exit code 0.

    Isolated via `sandbox_isolation.run_isolated` by default (falls back
    to plain subprocess gracefully if bwrap isn't installed, same as that
    module's own contract) -- HumanEval candidates are model-generated
    code same as SWE-bench patches, deserving the same hardening.
    """
    workdir = Path(root) / _SANDBOX_SUBDIR / uuid.uuid4().hex
    workdir.mkdir(parents=True, exist_ok=True)
    program = f"{candidate_code}\n\n{test_code}\n\ncheck({entry_point})\n"
    program_path = workdir / "candidate.py"
    program_path.write_text(program)
    try:
        cmd = [sys.executable, "candidate.py"]
        if isolate:
            result = run_isolated(cmd, cwd=workdir, timeout=timeout, env={"PYTHONDONTWRITEBYTECODE": "1"})
        else:
            result = subprocess.run(
                cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout,
            )
        return result.returncode == 0, (result.stdout + result.stderr)
    except subprocess.TimeoutExpired:
        return False, "TimeoutExpired"
    except Exception as e:
        return False, str(e)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run(
    sample: int, dataset_id: str, timeout: float, isolate: bool, verbose: bool,
    checkpoint: str | None = None, think: bool = False, max_tokens: int = 300,
    device: str | None = None,
) -> dict:
    from datasets import load_dataset

    from uchi.flux.inference_engine import build_generate_fn

    print(f"  Loading {dataset_id} ...")
    ds = load_dataset(dataset_id, split="test")
    if sample and sample < len(ds):
        import random
        ds = ds.select(random.sample(range(len(ds)), sample))
    print(f"  Running {len(ds)} HumanEval problem(s) through real execution grading")

    print(f"  Loading FLUX generate_fn ({checkpoint or 'default flux_best.pt'}, device={device or 'auto'}) ...")
    generate_fn = build_generate_fn(checkpoint=checkpoint, device=device)

    n_pass = 0
    per_problem = []
    t0 = time.time()

    for i, row in enumerate(ds):
        task_id = row["task_id"]
        prompt = _build_prompt(row["prompt"])
        response = generate_fn(prompt, max_tokens=max_tokens, think=think)
        blocks = _extract_code_blocks(response)
        candidate = blocks[0] if blocks else response

        passed, output = grade_candidate(
            candidate, row["test"], row["entry_point"], timeout=timeout, isolate=isolate,
        )
        n_pass += int(passed)
        per_problem.append({"task_id": task_id, "passed": passed})
        if verbose or (i + 1) % 5 == 0 or (i + 1) == len(ds):
            elapsed = time.time() - t0
            print(f"    [{i+1}/{len(ds)}] {task_id}: {'PASS' if passed else 'fail'}  elapsed={elapsed:.0f}s")

    n = len(per_problem)
    elapsed = time.time() - t0
    pass_at_1 = n_pass / n if n else 0.0

    print(f"\n  {'─'*70}")
    print("  HumanEval REAL Execution Results")
    print(f"  {'─'*70}")
    print(f"  Problems      : {n}")
    print(f"  Passed        : {n_pass}  (pass@1 = {pass_at_1*100:.1f}%)")
    if n:
        print(f"  Time          : {elapsed:.1f}s  ({elapsed/n:.1f}s/problem)")
    print(f"  {'─'*70}")

    return {
        "dataset": dataset_id, "n": n, "n_pass": n_pass,
        "pass_at_1": round(pass_at_1, 4), "elapsed_s": round(elapsed, 1),
        "per_problem": per_problem,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Uchi HumanEval Benchmark — REAL Execution (0.5.0 Item 9)")
    parser.add_argument("--sample", type=int, default=10, help="Problems to sample (0 = full 164)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--no-isolate", action="store_true", help="Disable bwrap isolation (debugging only)")
    parser.add_argument("--think", action="store_true", help="Use FLUX's <|think|> CoT lead-in")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--checkpoint", default=None,
                        help="FLUX checkpoint to evaluate (default: production flux_best.pt)")
    parser.add_argument("--device", default=None,
                        help="Force cpu/cuda (default: auto-detect). Useful to avoid VRAM "
                             "contention with a concurrent training run on the same GPU.")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    print("\n" + "=" * 70)
    print(" Uchi HumanEval Benchmark — REAL Execution (0.5.0 Item 9)")
    print("=" * 70 + "\n")

    results = run(
        args.sample, args.dataset, args.timeout, not args.no_isolate, args.verbose,
        checkpoint=args.checkpoint, think=args.think, max_tokens=args.max_tokens,
        device=args.device,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
