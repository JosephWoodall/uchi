---
name: Release Readiness
description: Executes the release readiness checklist for Uchi — regression tests, dynamic edge-case testing (existing + new functionality), trustworthiness benchmarks, PyPI-vs-local-branch parity, documentation verification, CI/CD checks, a fresh-install behavior test, and the release commit. Gates a merge to main.
---

# Release Readiness Checklist

Uchi uses **FLUX as the Proposer and Uchi as the Verifier**, wrapped in the
**`MetaUchi` orchestrator** (the default `from uchi import Uchi` facade —
`uchi.Core` is the raw, unorchestrated single-instance engine underneath).
FLUX is a small (~116M-class), from-scratch model — capability benchmarks are
a **dashboard to watch the proposer improve**, not a target it's expected to
win at this scale. The real release gate is trustworthiness (grounded or
honest abstention), that all three interfaces (SDK/TUI/REST) actually work
end-to-end, and — starting with 0.4.0 — that the autonomy surface (tool
calling, the scratchpad, goal tracking, checkpointing, HitL yielding) holds
up under adversarial input the same way the core Q&A path does.

**0.4.0's mission, in one line:** give Uchi hands. `MetaUchi` wraps the
grounded Q&A engine with tool calling, a sandboxed Python scratchpad, web
search, autonomous multi-step goals, checkpoint/resume, and macro
distillation — closing the capability gap with general assistants
(ChatGPT/Claude) *without* giving up the core differentiator: grounded
answers or honest abstention, never confabulation. Every release-readiness
pass should ask, for anything new: does this still hold the line on that
differentiator, or did autonomy get bolted on at the cost of trustworthiness?

## 1. Regression Testing
- Run `make test` (`pytest tests/`). All tests must pass.
- Tests must reflect the Proposer/Verifier logic, simplified API (REST, TUI, SDK), compounding knowledge, **and the full 0.4.0 autonomy surface** (tool calling, scratchpad, loop guard, goal state/compaction, HitL yielding, checkpointing, macro distillation, skill sharing, data silos).
- `CUDA_VISIBLE_DEVICES=""` when running tests if a training/inference job is
  using the GPU — the suite is CPU-only by design (conftest forces
  `FluxProposer.load()` -> None) and must never contend with a live GPU job.

## 2. Dynamic Edge-Case Testing (`tests/test_edge_cases.py`)
Regression tests only re-check behavior someone already thought to write a
test for. This step exists to catch what nobody thought of yet — for BOTH
the existing surface and whatever's new this release.

- Run `pytest tests/test_edge_cases.py -v`. It black-box tests every public
  `Core` method (empty/`None`/non-str inputs, huge/unicode/null-byte
  strings, SQL/prompt-injection-shaped strings, malformed slash commands,
  rapid repeated calls, nonexistent paths) — the way a real external caller
  hitting the API cold would, not the internal module unit tests.
- **It's "dynamic" because it checks its own completeness**:
  `test_public_api_has_edge_case_coverage` introspects `Core`'s actual
  public methods against a maintained coverage map and fails loudly if a
  new public method has no corresponding edge-case tests. A release that
  adds a new public method must add its edge-case coverage in the same
  patch, or this test blocks the release — that's what keeps this from
  quietly rotting into a fixed checklist as the API grows.
- When this suite finds a genuine hole (a crash where there should be a
  clean error, a silent no-op where there should be a signal), the default
  is: **write a test documenting current behavior so it's a tracked,
  visible finding, and separately decide whether to fix it now or file it**
  — don't let a real finding disappear because "the test would fail."
  Real example from the 0.4.0 pass: `ask("/")` (bare slash) raised an
  unhandled `IndexError` from deep inside slash-command parsing — found by
  this suite, fixed same-session, now a permanent regression test.
- Extend `_COVERAGE` and add a test class whenever `Core` gains a new
  public method (tool calling, goal state, checkpointing, skill sharing,
  telemetry, streaming are all covered as of 0.4.0 — the next release's new
  surface needs the same treatment).

## 3. PyPI-vs-Local-Branch Parity Check
The published package and the working branch can drift — a real user only
ever sees what's on PyPI, not what's in the repo. Before any release:

- `pip install` the **currently published** package into a clean venv
  (`python3 -m venv /tmp/uchi_pypi_test && pip install uchi-python`) and run
  the same edge-case battery against it, `CUDA_VISIBLE_DEVICES=""` to avoid
  contending with any live training job.
- Compare against the local branch's behavior on the same inputs. Two real
  bugs were caught this way in the 0.4.0 pass and are now permanent checks:
  - **Packaging:** `uchi/skills/*.md` was missing from `[tool.setuptools.
    package-data]` in `pyproject.toml`, so every documented slash command
    (`/classify`, `/forecast`, etc.) silently failed on every `pip install`
    with zero warning (`SkillRegistry._load_dir`'s `isdir` guard swallows a
    missing directory silently). **Verify with a real build**, not just
    reading the config: `python -m build --wheel --outdir /tmp/wheel_check
    .` then `unzip -l dist/*.whl | grep skills/` — confirm all skill `.md`
    files are present before publishing.
  - **Output quality:** the documented README quickstart examples
    (factorial, `/classify data.csv`) should be spot-checked against the
    real published package's actual output, not assumed to work because
    they're "just" a documented example — a nonsensical answer to a
    textbook example is exactly the kind of thing a new user hits first
    and is exactly the kind of finding a regression suite alone won't
    catch (nothing there was "wrong" in the unit-test sense; the answer
    was just unrelated to the question).
- Watch for stray directories shadowing the package during ad-hoc testing
  (a script sitting directly in a dir that also contains a bare `<pkgname>`
  subdirectory can get shadowed by Python's import resolution) — run
  throwaway test scripts from an isolated scratch directory, not `/tmp`
  directly.

## 4. Performance & Capability Dashboard (not a pass/fail gate)
With FLUX proposing and Uchi verifying, capability benchmarks are runnable
again (all construct `Uchi()` directly — no `brain.uchi` pickle, no OmniRouter):

- **MMLU:** `python benchmarks/mmlu_benchmark.py --sample 200` (Factual recall)
- **SWE-bench:** `python benchmarks/swebench_benchmark.py --sample 50` (Code generation proxy)
- **ARC-Challenge:** `python benchmarks/arc_benchmark.py --sample 200` (Reasoning chains)
- **Trustworthiness:** `python benchmarks/trustworthiness.py` (SQuAD 2.0 — coverage, precision@answered, honest-abstention, hallucination-rate)
- **OPS (0.4.0):** `python benchmarks/ops_benchmark.py --steps 10` — Operations Per Second for the autonomous tool-calling loop (a complete inner-monologue cycle ending in a successful tool call). New metric class for the autonomy surface; track it the same dashboard way as MMLU/SWE/ARC, not as a launch blocker yet.

**Regression rule:** record the numbers in the release commit, but do not
block a release on MMLU/SWE/ARC moving — at this model scale they stay near
baseline regardless of engineering quality (see `docs/training.md`). The
**hallucination rate must not rise** — that is the real, enforceable gate.
`benchmarks/run_benchmarks.py` is legacy (v0.2.0-era OmniRouter internals —
`.predictor._pred`, `.tokenizer.tokenize`, `.stream()` — that no longer exist
on `Uchi`); it is not part of this checklist.

## 5. Premade Brain Sanity Check
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

## 6. Documentation Verification
- Check that the non-negotiables list is current for the release (v0.3.0's
  5 non-negotiables — compounding effect, simplified public API, general
  reasoning/chains, human-readable I/O, OOD generalization — plus, from
  0.4.0 on, the autonomy contract: tool calls are logged and loop-guarded,
  goal state compacts instead of losing intent, HitL yields instead of
  guessing, and none of that changes the grounded-or-abstain guarantee).
- Verify `README.md`, `docs/`, `CHANGELOG.md` properly explain FLUX as the
  proposer and Uchi as the verifier, that capability-benchmark claims match
  the dashboard-not-target framing (no "0% hallucination" or "benchmarks
  are back/conquered" language), and that the `MetaUchi`/`Core` split is
  documented (which one `from uchi import Uchi` returns, and when to reach
  for `uchi.Core` instead).
- `docs/training.md` must document the training pipeline and the
  `flux_best.pt` artifact, including any new data sources or architecture
  changes (vocab pruning, fused scan) folded into the current training run.
- README must show exactly how to train FLUX from scratch, and how to
  obtain pretrained weights if not training locally.
- **The "How It Connects" architecture diagram in `README.md` must be
  updated whenever the architecture it depicts changes.** As of this
  writing it's still the v0.3.0 diagram — `Uchi.__init__` picking a
  checkpoint, `GenerateAndGround`, then SDK/TUI/REST fanning out from a
  single `Uchi()` construction. It has no `MetaUchi`/`Core` split, no tool
  calling, no scratchpad, no goal state, no `/ask/stream` or
  `/v1/chat/completions`. Do not let a release ship with a diagram that
  describes last release's architecture: redraw it (same ASCII-box style)
  to show what `from uchi import Uchi` actually constructs and connects to
  as of the release being cut, and confirm every box in it is a real,
  current class/function name — grep for each one before publishing, the
  same way the rest of this checklist insists on verifying claims against
  running code rather than assuming docs kept up.

## 7. CI/CD & APIs
- Ensure all Simplified APIs are functional: `uchi tui` loads, `uchi serve`
  responds on `/health` + `/ask` + `/ask/stream` (SSE) + `/v1/chat/completions`
  (OpenAI-compatible), Python SDK passes tests.
- Confirm `pyproject.toml` version is updated, and re-run the wheel-contents
  check from Section 3 after any `package-data` change.

## 8. Fresh Install Test
- Install via `pip install -e .` (or the built wheel from Section 3) and
  verify the basic SDK behaves correctly with zero setup (no manual
  `learn()` needed for a general-knowledge question, thanks to the premade
  brain; slash commands work, confirming skills were actually bundled).

## 9. Prepare Release Commit
Stage and commit changes, and provide the user with the git push command.
