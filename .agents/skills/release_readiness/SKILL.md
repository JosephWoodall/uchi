---
name: Release Readiness
description: Executes the release readiness checklist for Uchi — regression tests, trustworthiness benchmarks, documentation verification, CI/CD checks, a fresh-install behavior test, and the release commit. Gates a merge to main.
---

# Release Readiness Checklist

Uchi v0.3.0 uses **FLUX as the Proposer and Uchi as the Verifier.** FLUX is a
small (~116M), from-scratch model — capability benchmarks are a **dashboard to
watch the proposer improve**, not a target it's expected to win at this scale.
The real release gate is trustworthiness (grounded or honest abstention) and
that all three interfaces (SDK/TUI/REST) actually work end-to-end.

## 1. Regression Testing
- Run `make test` (`pytest tests/`). All tests must pass.
- Tests must reflect the Proposer/Verifier logic, simplified API (REST, TUI, SDK), and compounding knowledge.
- `CUDA_VISIBLE_DEVICES=""` when running tests if a training/inference job is
  using the GPU — the suite is CPU-only by design (conftest forces
  `FluxProposer.load()` -> None) and must never contend with a live GPU job.

## 2. Performance & Capability Dashboard (not a pass/fail gate)
With FLUX proposing and Uchi verifying, capability benchmarks are runnable
again (all construct `Uchi()` directly — no `brain.uchi` pickle, no OmniRouter):

- **MMLU:** `python benchmarks/mmlu_benchmark.py --sample 200` (Factual recall)
- **SWE-bench:** `python benchmarks/swebench_benchmark.py --sample 50` (Code generation proxy)
- **ARC-Challenge:** `python benchmarks/arc_benchmark.py --sample 200` (Reasoning chains)
- **Trustworthiness:** `python benchmarks/trustworthiness.py` (SQuAD 2.0 — coverage, precision@answered, honest-abstention, hallucination-rate)

**Regression rule:** record the numbers in the release commit, but do not
block a release on MMLU/SWE/ARC moving — at 116M params they stay near
baseline regardless of engineering quality (see `docs/training.md`). The
**hallucination rate must not rise** — that is the real, enforceable gate.
`benchmarks/run_benchmarks.py` is legacy (v0.2.0-era OmniRouter internals —
`.predictor._pred`, `.tokenizer.tokenize`, `.stream()` — that no longer exist
on `Uchi`); it is not part of this checklist.

## 3. Premade Brain Sanity Check
`uchi/data/embeddings.pt` ships a general-knowledge seed (skip-gram vocabulary
+ pre-embedded passages) so a fresh `Uchi()` can retrieve without any `learn()`
call. Verify it loads and actually retrieves:
```python
from uchi import Uchi
u = Uchi()
assert len(u.index.w2i) > 0 and len(u.index.passages) > 0
u.index.retrieve("What is photosynthesis?", k=3)  # must return non-empty results
```
If `w2i`/`passages` are empty, `learn()` becomes a silent no-op system-wide
(`SemanticIndex._vec()` only embeds words already in `w2i` — it never learns
new vocabulary) — this is a release blocker, not a degraded-mode fallback.

## 4. Documentation Verification
- Check that 5 non-negotiables are mentioned: Compounding effect, simplified public api (SDK, TUI, REST API), general reasoning / chains, human-readable I/O, OOD generalization.
- Verify `README.md`, `docs/`, `CHANGELOG.md` properly explain FLUX as the proposer and Uchi as the verifier, and that capability-benchmark claims match the dashboard-not-target framing (no "0% hallucination" or "benchmarks are back/conquered" language).
- `docs/training.md` must document the training pipeline and the `flux_best.pt` artifact.
- README must show exactly how to train FLUX from scratch, and how to obtain pretrained weights if not training locally.

## 5. CI/CD & APIs
- Ensure all Simplified APIs are functional: `uchi tui` loads, `uchi serve` responds on `/health`+`/ask`, Python SDK passes tests.
- Confirm `pyproject.toml` version is updated.

## 6. Fresh Install Test
- Install via `pip install -e .` and verify the basic SDK behaves correctly with zero setup (no manual `learn()` needed for a general-knowledge question, thanks to the premade brain).

## 7. Prepare Release Commit
Stage and commit changes, and provide the user with the git push command.
