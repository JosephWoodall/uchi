---
name: Release Readiness
description: Executes the release readiness checklist - run regression tests, verify dual-layer documentation, check CI/CD pipelines, and safely push to the active branch.
---

# Release Readiness Checklist

When invoked to perform a release readiness check (or "release checklist"), follow these exact steps in order:

## 1. Regression Testing
Run the comprehensive test suite to ensure zero regressions. 
- Execute `make test` (or `pytest tests/`).
- Verify that all tests pass successfully. If any tests fail, stop and fix them before proceeding.

## 2. Performance & Capability Benchmarks
Run the benchmarking suite to measure Uchi's language understanding (MMLU) and coding ability (SWE-bench), the two authoritative public benchmarks reported on the README.

- Execute `python benchmarks/run_benchmarks.py --mini`
  - `--mini` mode runs **5 MMLU questions** and **5 SWE-bench instances** — fast enough for release-gate CI. For a full publication-quality run use `--mmlu-samples 500 --swe-samples 50` (drops `--mini`).
  - **MMLU** (language understanding): samples questions from the Massive Multitask Language Understanding dataset across 57 academic subjects. Reports accuracy per subject and overall accuracy.
  - **SWE-bench** (coding): runs a subset of real GitHub issue→patch tasks. Reports resolve rate (% of issues where Uchi's generated patch passes all tests).
- Verify MMLU accuracy ≥ the prior baseline and SWE-bench resolve rate ≥ prior baseline. If either drops, treat it as a regression and fix before proceeding.
- **CRITICAL**: Update the `README.md` "Benchmarks" section with the latest scores (MMLU accuracy, SWE-bench resolve rate) produced by the script.

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
  - `benchmarks/` directory: remove superseded benchmark runners. Keep only `run_benchmarks.py`, `mmlu_benchmark.py`, and `swebench_benchmark.py` plus their result JSON files.
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

## 6. Final Push
Once all checks pass, push the changes to GitHub:
- Stage all changes (`git add .`).
- Commit with a descriptive message (e.g., "chore: final release readiness sweep").
- Push to the current branch (`git push`).
- *Note:* If GitHub rejects the push due to workflow scope restrictions, un-stage `.github/workflows/`, push the rest, and instruct the user to update the workflow via the GitHub Web UI.
