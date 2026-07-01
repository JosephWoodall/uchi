---
name: Release Readiness
description: Executes the release readiness checklist - run regression tests, verify dual-layer documentation, check CI/CD pipelines, and prepare the release commit for the user to push.
---

# Release Readiness Checklist

When invoked to perform a release readiness check (or "release checklist"), follow these exact steps in order:

## 1. Regression Testing
Run the comprehensive test suite to ensure zero regressions. 
- Execute `make test` (or `pytest tests/`).
- Verify that all tests pass successfully. If any tests fail, stop and fix them before proceeding.

## 2. Performance & Capability Benchmarks
Run the compounding benchmarking suite to measure Uchi's intelligence across reasoning (MMLU, ARC-Challenge), patching (SWE-bench), agency (OPS/WebArena), and synthesis (HumanEval). All benchmarks must be run on every release to ensure zero regression.

**CRITICAL — Brain Knowledge Prerequisite**: Before running any benchmark, verify the brain contains training data for each benchmark domain. The following sources must have been ingested via `incremental_builder` before results are meaningful:
- **MMLU**: `mmlu_qna` (auxiliary_train split, minimum 5000 examples)
- **SWE-bench**: `swebench` + `humaneval` (minimum 500 examples each)
- **ARC-Challenge**: `arc` (ARC-Challenge train split, minimum 500 examples)

If any source is missing, run `python -m uchi.incremental_builder --sources <missing_source> --limit 500` and re-run GRPO before benchmarking.

- Execute `python benchmarks/run_benchmarks.py --mini`
  - `--mini` mode runs **5 instances of each benchmark** — fast enough for release-gate CI. For a full publication-quality run, omit `--mini`.
  - **MMLU** (language understanding): measures factual recall and reasoning accuracy across 57 academic subjects.
  - **ARC-Challenge** (reasoning): measures multi-concept reasoning on elementary science questions specifically designed to resist pattern matching. This is the primary reasoning signal — a higher bar than MMLU.
  - **SWE-bench** (coding): measures ability to resolve GitHub issues via repository patches.
  - **OPS** (Operations Per Second): measures agentic latency and throughput for autonomous loops.
  - **WebArena**: measures autonomous web navigation and task completion.
  - **HumanEval**: measures pure Python algorithmic synthesis and unit test passing.
- Run individual benchmark scripts for detailed per-benchmark output:
  - `python benchmarks/mmlu_benchmark.py --sample 200 --brain brain.uchi`
  - `python benchmarks/arc_benchmark.py --sample 200 --brain brain.uchi`
  - `python benchmarks/swebench_benchmark.py --sample 50 --brain brain.uchi`
- Verify that **ALL** scores (MMLU, ARC-Challenge, SWE-bench, OPS, WebArena, HumanEval) are ≥ the prior baseline. If any score drops, treat it as an architectural regression and fix before proceeding.
- **CRITICAL — Publish Benchmarks to README**: After every benchmark run, update the `README.md` "Benchmarks" section with the new scores. This is mandatory, not optional. Specifically:
  - Update the narrative sections for each benchmark (MMLU, ARC-Challenge, SWE-bench, OPS, WebArena, HumanEval).
  - Update the `<!-- BENCHMARK_TABLE_START --> ... <!-- BENCHMARK_TABLE_END -->` summary table with rows for all active benchmarks.
  - The prior scores must be replaced, not appended — the table always reflects the current brain, not history. Benchmark history belongs in `tasks/0.X.0 Itemized Deliverables.md`.

## 3. Documentation Verification
Ensure the documentation architecture is pristine:
- **README Core Mission**: Verify the root `README.md` core mission section reflects the current version (not a prior release). It must open with the `Uchi` class as the primary entry point, show the compounding mechanism (`ask()` → string → `learn()`), and not reference a stale version number. Update it if anything is outdated.
- **Dual-Layer Strategy**: Verify that the root `README.md` remains a short, clean "elevator pitch" (logo, badges, brief summary, pip install, quickstart, and a link to `docs/`).
- **Comprehensive Docs**: Ensure the `docs/` directory contains the exhaustive markdown files and that `mkdocs.yml` accurately maps to them. Verify `docs/python-api.md` exists as the canonical `Uchi` API reference.
- **Enterprise Layout Requirement**: Verify that every module/service documented in `docs/` explicitly contains a `### Realistic Use Cases` section (with exactly 3 concrete examples) followed by a `### The Ultimate Benefit` paragraph.
- **Changelog**: Verify that `CHANGELOG.md` is updated with the latest version features.
- **Full API Example in README**: Verify that the `README.md` Python API section contains a single, runnable code block that demonstrates the *complete* public facade — every method and property of the `Uchi` class in order: `Uchi()` constructor variants, `learn()`, `ingest()`, `ask()` (natural language + all slash commands), the compounding pattern, `predictor` (fit/train/partial_fit/predict_next/generate/score), `stream()`, `web_search` (get/set), `save()`, and `router`. If this block is absent, outdated, or missing any surface, update it before proceeding.
- **Codebase Hygiene Sweep**: Scan the entire repository for outdated, non-necessary, or irrelevant files and remove them before release. Specifically check:
  - Root-level scratch/debug scripts (`debug_*.py`, `test_*.py` at root, `evaluate_*.py`, `fix_*.py`, `patch_*.py`, `api_*.py`) — remove any that are not part of the public interface.
  - `scripts/` directory: remove one-off tools, old patch scripts, and any script whose function is now covered by a module in `uchi/`. Keep only scripts referenced in `README.md` (e.g., bootstrap scripts).
  - `benchmarks/` directory: remove superseded benchmark runners. Keep only `run_benchmarks.py`, `mmlu_benchmark.py`, `arc_benchmark.py`, and `swebench_benchmark.py` plus their result JSON files.
  - `examples/` directory: remove any example that uses the old `OmniRouter`-direct or `UniversalPredictor`-direct API. All examples must use `from uchi import Uchi`.
  - Temporary/runtime artifacts on disk: delete `*.tmp`, `*.pkl` caches, `replay.db`, `ssm_dynamics.pt`, `uchi_cpu_memory_*.{json,bin,npy}`, `uchi_procedural_memory.json`, and the `site/` build directory. These are all listed in `.gitignore` and must not exist in a clean release state.
  - Verify `.gitignore` covers all generated artifacts so they cannot be accidentally committed.

## 4. Engineering Best Practices & CI/CD
Ensure the repository adheres to elite open-source standards:
- Verify that `.github/workflows/ci.yml` exists and runs `pytest`, `ruff`, and `mypy`.
- Ensure `.pre-commit-config.yaml`, `CITATION.cff`, `SECURITY.md`, and `.github/FUNDING.yml` are present.
- Ensure the `pyproject.toml` contains the necessary `dev` and `test` dependencies.
- **CRITICAL**: Ensure that the `pyproject.toml` file is updated where it needs to be on each release readiness (e.g., bumping the version number, adding new dependencies).

## 5. Fresh Install Test
Ensure that the codebase works perfectly for a new user:
- Verify that all core dependencies required to run the CLI, API Harness, and tests are explicitly defined in `pyproject.toml` (e.g., `fastapi`, `uvicorn`, `requests`, `beautifulsoup4`, `tqdm`).
- Run a simulated installation (`pip install -e .`) to guarantee that the `uchi` entrypoint binds correctly without missing external packages.
- **Normalizer smoke test**: Run the following and verify the response contains no `.n.01` / `.v.02` synset markers, no `<|...|>` control tokens, starts with a capital letter, and ends with punctuation:
  ```python
  from uchi import Uchi
  u = Uchi()
  reply = u.ask("What is photosynthesis?")
  print(repr(reply))
  ```

## 6. Prepare Release Commit
Once all checks pass, stage and commit the release. **Do NOT push** — the user runs the final push themselves.
- Stage relevant files (be explicit; avoid `git add .` which can accidentally include runtime artifacts):
  ```
  git add uchi/ tests/ docs/ benchmarks/ README.md CHANGELOG.md pyproject.toml
  ```
- Commit with a descriptive message (e.g., `chore: vX.Y.Z release readiness sweep`).
- Print the exact push command for the user to run:
  ```
  git push -u origin <current-branch-name>
  ```
- *Note:* If GitHub is expected to reject the push due to workflow scope restrictions, instruct the user to un-stage `.github/workflows/` before pushing and update the workflow via the GitHub Web UI instead.
