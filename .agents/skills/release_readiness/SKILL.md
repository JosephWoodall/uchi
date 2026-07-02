---
name: Release Readiness
description: Executes the release readiness checklist for Uchi — regression tests, trustworthiness benchmarks, documentation verification, CI/CD checks, a fresh-install behavior test, and the release commit. Gates a merge to main.
---

# Release Readiness Checklist

Uchi v0.3.0 uses **FLUX as the Proposer and Uchi as the Verifier.** This means we evaluate both raw OOD capability AND trustworthiness.

## 1. Regression Testing
- Run `make test`. All tests must pass.
- Tests must reflect the Proposer/Verifier logic, simplified API (REST, TUI, SDK), and compounding knowledge.

## 2. Performance & Capability Benchmarks
With FLUX proposing and Uchi verifying, capability benchmarks are **BACK**:

- **MMLU:** `python benchmarks/mmlu_benchmark.py` (Factual recall)
- **SWE-bench:** `python benchmarks/swebench_benchmark.py` (Code generation)
- **ARC-Challenge:** `python benchmarks/arc_benchmark.py` (Reasoning chains)
- **Trustworthiness:** `python benchmarks/trustworthiness.py` (SQuAD 2.0 hallucination tracking)

**Regression rule:** Scores across MMLU, SWE, and ARC must not drop compared to the previous baseline. Hallucination rate must not rise.

## 3. Documentation Verification
- Check that 5 non-negotiables are mentioned: Compounding effect, simplified public api (SDK, TUI, REST API), general reasoning / chains, human-readable I/O, OOD generalization.
- Verify `README.md`, `docs/`, `CHANGELOG.md` properly explain FLUX as the proposer and Uchi as the verifier.

## 4. CI/CD & APIs
- Ensure all Simplified APIs are functional: TUI loads, REST API responds, Python SDK passes tests.
- Confirm `pyproject.toml` version is updated.

## 5. Fresh Install Test
- Install via `pip install -e .` and verify the basic SDK behaves correctly.

## 6. Prepare Release Commit
Stage and commit changes, and provide the user with the git push command.
