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

## 2. Performance Benchmarks
Run the benchmarking suite to verify the model maintains its compute and memory efficiency advantages.
- Execute `python benchmarks/run_benchmarks.py`
- Verify that inference latency and training throughput meet architectural expectations.

## 3. Documentation Verification
Ensure the documentation architecture is pristine:
- **Dual-Layer Strategy**: Verify that the root `README.md` remains a short, clean "elevator pitch" (logo, badges, brief summary, pip install, quickstart, and a link to `docs/`).
- **Comprehensive Docs**: Ensure the `docs/` directory contains the exhaustive markdown files and that `mkdocs.yml` accurately maps to them.
- **Enterprise Layout Requirement**: Verify that every module/service documented in `docs/` explicitly contains a `### Realistic Use Cases` section (with exactly 3 concrete examples) followed by a `### The Ultimate Benefit` paragraph.
- **Changelog**: Verify that `CHANGELOG.md` is updated with the latest version features.

## 3. Engineering Best Practices & CI/CD
Ensure the repository adheres to elite open-source standards:
- Verify that `.github/workflows/ci.yml` exists and runs `pytest`, `ruff`, and `mypy`.
- Ensure `.pre-commit-config.yaml`, `CITATION.cff`, `SECURITY.md`, and `.github/FUNDING.yml` are present.
- Ensure the `pyproject.toml` contains the necessary `dev` and `test` dependencies.

## 4. Fresh Install Test
Ensure that the codebase works perfectly for a new user:
- Verify that all core dependencies required to run the CLI, API Harness, and tests are explicitly defined in `pyproject.toml` (e.g., `fastapi`, `uvicorn`, `requests`, `beautifulsoup4`, `tqdm`).
- Run a simulated installation (`pip install -e .`) to guarantee that the `uchi` entrypoint binds correctly without missing external packages.

## 5. Final Push
Once all checks pass, push the changes to GitHub:
- Stage all changes (`git add .`).
- Commit with a descriptive message (e.g., "chore: final release readiness sweep").
- Push to the current branch (`git push`).
- *Note:* If GitHub rejects the push due to workflow scope restrictions, un-stage `.github/workflows/`, push the rest, and instruct the user to update the workflow via the GitHub Web UI.
