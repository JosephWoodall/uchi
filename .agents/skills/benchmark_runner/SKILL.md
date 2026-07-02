---
name: benchmark_runner
description: Runs MMLU, SWE-bench, and ARC-Challenge benchmarks against the current FLUX+Uchi engine to evaluate factual recall, code generation, and reasoning capabilities.
---

# Benchmark Runner

With the v0.3.0 architecture (FLUX as Proposer, Uchi as Verifier), this skill runs the three core benchmarks to ensure our OOD generalization and reasoning chains remain state-of-the-art: MMLU (factual recall), SWE-bench (code generation), and ARC-Challenge (reasoning).

## Usage

Run each benchmark individually using its dedicated script:

```bash
# MMLU — factual recall across 57 academic subjects
python benchmarks/mmlu_benchmark.py --sample 200 --brain brain.uchi

# SWE-bench — code generation quality proxy
python benchmarks/swebench_benchmark.py --sample 50 --brain brain.uchi

# ARC-Challenge — general reasoning and reasoning chains
python benchmarks/arc_benchmark.py --sample 200 --brain brain.uchi
```

## Output
Each benchmark reports accuracy, reasoning chain integrity, speed, and saves the JSON results.

## Regression Policy
Because FLUX provides the OOD generalization and Uchi guarantees it, any drop in these benchmarks indicates a failure in either the Proposer's raw capability or the Verifier's logic. All scores must be ≥ the prior release baseline.
