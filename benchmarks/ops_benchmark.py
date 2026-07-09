"""
ops_benchmark.py
================
Uchi Operations Per Second (OPS) Benchmark Harness (0.4.0 Item 15).

Measures how fast Uchi's autonomous tool-calling loop can execute a
complete inner-monologue cycle: one ``Core.ask()`` round trip that ends
in a successful, syntactically valid tool call being dispatched.

This is an infrastructure throughput metric (parse -> dispatch ->
loop-guard check -> log -> goal-state record -> splice — Items 4/5/6),
not a model-quality metric: FLUX hasn't been trained to emit the
tool-call grammar yet (that's a training-data question for a later FLUX
iteration, see Item 0). This harness drives the real Items 4-10 machinery
with a scripted step sequence standing in for FLUX's output, exactly the
way every item in this pass was verified end-to-end through the real
``Core.ask()`` path rather than by calling internals directly.

A single operation = one ``ask()`` call whose swarm response contains
exactly one ``<|tool_call|>``, dispatched successfully.

Usage:
    python benchmarks/ops_benchmark.py
    python benchmarks/ops_benchmark.py --steps 25
    python benchmarks/ops_benchmark.py --out benchmarks/ops_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_RESULTS_DIR = os.path.dirname(__file__)


def _scripted_step(i: int) -> str:
    """One synthetic inner-monologue step: a syntactically valid tool call
    Uchi's swarm response "would" emit. Deterministic and sandboxed
    (Python scratchpad only), so infrastructure throughput is measured,
    not FLUX's judgment.
    """
    return f'<|tool_call|> run_python(code="print({i} * 2)") <|end_tool|>'


def run_ops_benchmark(n_steps: int = 10, warmup: int = 1) -> Dict[str, Any]:
    """Drive *n_steps* real ``Core.ask()`` cycles through the actual
    Item 4-10 tool-calling pipeline and report Operations Per Second."""
    from uchi import Core

    core = Core()
    core.start_goal("OPS benchmark task")

    # Warm-up excluded from timing: cold-start costs (predictor
    # construction, first subprocess spawn) shouldn't skew the rate.
    for i in range(warmup):
        core.swarm.answer = lambda q, callback=None, i=i: _scripted_step(i)
        core.ask(f"warmup step {i}")

    core.tools.log.clear()
    core.goal_state.raw_log.clear()
    core.goal_state.successful_steps.clear()

    per_step: List[float] = []
    t_start = time.perf_counter()
    for i in range(n_steps):
        core.swarm.answer = lambda q, callback=None, i=i: _scripted_step(i)
        t0 = time.perf_counter()
        core.ask(f"step {i}")
        per_step.append(time.perf_counter() - t0)
    elapsed = time.perf_counter() - t_start

    successes = sum(1 for e in core.tools.log if e.ok)
    ops = successes / elapsed if elapsed > 0 else 0.0

    return {
        "n_steps": n_steps,
        "successful_tool_calls": successes,
        "elapsed_seconds": elapsed,
        "ops": ops,
        "mean_step_seconds": sum(per_step) / len(per_step) if per_step else 0.0,
        # Exit criterion 3 sanity check: the original goal must still be
        # intact after every step, even once compaction has kicked in.
        "context_intact": core.goal_state.goal == "OPS benchmark task",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Uchi Operations Per Second (OPS) benchmark.")
    parser.add_argument("--steps", type=int, default=10, help="Number of autonomous steps to run.")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up steps excluded from timing.")
    parser.add_argument("--out", type=str, default=os.path.join(_RESULTS_DIR, "ops_results.json"))
    args = parser.parse_args()

    print(f"[*] Running OPS benchmark: {args.steps} steps ({args.warmup} warm-up)...")
    result = run_ops_benchmark(n_steps=args.steps, warmup=args.warmup)

    print(f"[+] {result['successful_tool_calls']}/{result['n_steps']} successful tool calls")
    print(f"[+] elapsed: {result['elapsed_seconds']:.3f}s")
    print(f"[+] OPS: {result['ops']:.2f} operations/sec")
    print(f"[+] mean step latency: {result['mean_step_seconds']*1000:.1f}ms")
    print(f"[+] goal context intact after {args.steps} steps: {result['context_intact']}")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[+] results written to {out_path}")


if __name__ == "__main__":
    main()
