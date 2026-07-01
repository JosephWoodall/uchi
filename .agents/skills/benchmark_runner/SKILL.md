---
name: benchmark_runner
description: Runs MMLU, SWE-bench, and ARC-Challenge benchmarks against the current Uchi engine to evaluate factual recall, code generation, and reasoning capabilities.
---

# Benchmark Runner

This skill runs the three core benchmarks against Uchi: MMLU (factual recall), SWE-bench (code generation), and ARC-Challenge (reasoning). All three must be run together on every release — they measure different capability axes and any regression on any one of them is a blocker.

**Brain Knowledge Prerequisite**: Before benchmarking, verify the brain has been incrementally built with the correct training data for each benchmark:
- **MMLU**: `mmlu_qna` source (auxiliary_train, ≥5000 examples)
- **SWE-bench**: `swebench` + `humaneval` sources (≥500 examples each)
- **ARC-Challenge**: `arc` source (ARC-Challenge train split, ≥500 examples)

If any source is missing, run the incremental builder first, then re-run GRPO before benchmarking.

## Usage

Run each benchmark individually using its dedicated script:

```bash
# MMLU — factual recall and reasoning accuracy across 57 academic subjects
python benchmarks/mmlu_benchmark.py --sample 200 --brain brain.uchi

# SWE-bench — code generation quality proxy (bug-fix patch generation)
python benchmarks/swebench_benchmark.py --sample 50 --brain brain.uchi

# ARC-Challenge — multi-concept reasoning on elementary science questions
# Primary reasoning signal: harder to game than MMLU, resists pattern matching
python benchmarks/arc_benchmark.py --sample 200 --brain brain.uchi
```

Results are saved to `benchmarks/mmlu_results.json`, `benchmarks/swebench_results.json`, and `benchmarks/arc_results.json` respectively.

## Output
Each benchmark reports:
1. Accuracy / composite score
2. No-parse rate (format failures — should be <10% after v0.3.0)
3. Per-question progress at every 50-question interval
4. Speed (ms/question)
5. JSON results file for archiving in `tasks/0.X.0 Itemized Deliverables.md`

## Regression Policy
All scores must be ≥ the prior release baseline. If any score drops:
- **MMLU accuracy drop**: check if `mmlu_qna` source is still in the brain
- **ARC-Challenge drop**: check if `arc` source is in the brain; ARC is the canary for reasoning regression
- **SWE code generation drop**: check if `swebench` + `humaneval` sources are in the brain
- **No-parse rate spike**: SSM weights may have regressed — re-run GRPO before releasing
