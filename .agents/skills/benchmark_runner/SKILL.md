---
name: benchmark_runner
description: Runs a miniature suite of MMLU and SWE-Bench tests against the current Uchi engine to evaluate correctness and inference latency.
---

# Benchmark Runner

This skill allows agents (or the user) to quickly run a suite of baseline tests against Uchi. It is designed to expose both factual reasoning capabilities (MMLU) and code-generation capabilities (SWE-Bench), while also rigorously tracking the MCTS inference latency.

## Usage
Run the script passing either the `--mmlu` or `--swe` flag. 

You can also pass the `--wipe` flag to completely delete all `.uchi`, `.db`, and `.pt` brain files from the directory before running the benchmark, guaranteeing you are testing a completely blank slate.

```bash
# Run the MMLU factual reasoning benchmark with a completely wiped brain
python scripts/benchmark_mini.py --mmlu --wipe

# Run the SWE-Bench code generation benchmark
python scripts/benchmark_mini.py --swe --wipe
```

## Output
The script will output:
1. The exact prompt given to Uchi.
2. Uchi's generated string.
3. The inference latency in seconds (useful for verifying the Virtual Loss Batched Evaluation patch).
4. A PASS/FAIL status based on substring matching.
5. A final aggregate score and total time elapsed.
