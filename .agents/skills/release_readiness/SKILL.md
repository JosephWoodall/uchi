---
name: Release Readiness
description: Executes the release readiness checklist for Uchi — regression tests, trustworthiness benchmarks, documentation verification, CI/CD checks, a fresh-install behavior test, and the release commit. Gates a merge to main.
---

# Release Readiness Checklist

Uchi is a grounded, no-LLM assistant whose value is **trustworthiness** — it grounds
factual answers, reasons in verified steps, and abstains rather than confabulate.
The release gate must protect that promise. Run these steps in order; any failure
stops the release.

## 1. Regression Testing

- Run `make test` (or `.venv/bin/python -m pytest tests/ --ignore=tests/tests.py -q`).
- **All tests must pass** (expected: ~150 passing).
- The suite must be *current*: if any test references removed machinery (the retired
  "Family C": `convergent_engine`, `tree_search_engine`, `grpo`, `grpo_offline_trainer`,
  `calibration`, `grammar_mask`, `omni_evaluator`, SSM QA-discrimination / GRPO training),
  **update or delete that test** — do not skip it. Tests for the live architecture
  (`generate_and_ground`, `oracle`, `retrieval`, `answerability`, `decoder`,
  `intent_router`, `conversation`, `reasoning`, `skill_registry`) must exist and pass.
- If a behavioral change is intentional (e.g. `ask()` now abstains where it once
  guessed), update the test's expectation to the new **honest** behavior — never
  loosen an assertion to hide a real regression.

## 2. Performance & Capability Benchmarks

Uchi is measured on **trustworthiness**, not LLM-style accuracy. The old
MMLU/ARC-Challenge/SWE-bench accuracy benches are **retired** — they measure the
wrong axis (this is a no-LLM system; those are at random and not what it is for).

- **Headline — SQuAD 2.0 trustworthiness**: `python benchmarks/trustworthiness.py --sample 800`
  - Reports: **coverage** (% answerable answered), **precision@answered**,
    **honest-abstention** (% unanswerable correctly declined), **hallucination-rate**
    (% of emitted answers that are wrong — the number that matters).
  - **Regression rule (flipped from accuracy thinking):** hallucination-rate must
    NOT rise and honest-abstention must NOT fall versus the prior release at the
    default answerability threshold. Coverage may trade for caution.
- **Reasoning (separate skill)**: `python experiments/arc_dsl.py --split evaluation`
  — the program-synthesis reasoner. Report solve-rate + precision (a verified
  program must reproduce the demos). Honest scope: grid tasks only, not general QA.
- **Retrieval diagnostic** (optional): recall@k of the answer-bearing passage — the
  primary quality lever; regressions here explain precision drops.
- **Publish HONEST numbers to README**: update the trustworthiness KPI table with the
  new run. Do **not** publish, or reintroduce, any "0% hallucination" claim — the
  measured rate is not zero. State limitations plainly (see §3).

## 3. Documentation Verification

- **Honest-claims gate (mandatory).** Grep `README.md`, `docs/`, `CHANGELOG.md`, and
  release copy for disproven claims and FAIL if any are present:
  `grep -rniE "0% hallucination|zero hallucination|teaches it to reason|reasons like a brain|hallucination-free" README.md docs/ CHANGELOG.md`
  Uchi does not achieve 0% hallucination and does not reason like an LLM. Claims must
  match the benchmarks.
- **README** opens with `from uchi import Uchi`, the compounding `learn()`/`ask()`
  contract, the 3-lane router (factual / social / skill), the honest trustworthiness
  table, and the stated retrieval/generation limitation. No stale version number.
- **docs/** contains no docs for deleted modules (e.g. `convergent-engine.md` must be
  gone) and `mkdocs.yml` nav matches the files present. Core docs
  (`architecture.md`, `generate-and-ground.md`, `reasoning.md`, `benchmarks.md`,
  `python-api.md`, `index.md`) reflect the current architecture.
- **CHANGELOG.md** has an entry for this version describing the real changes.
- **python-api.md** documents the live `Uchi` surface: `learn()`, `ingest()`, `ask()`
  (NL 3-lane + slash commands), `predictor`, `save()`, `router`.

## 4. Engineering Best Practices & CI/CD

- `.github/workflows/ci.yml` exists and runs `pytest`, `ruff`, and `mypy`.
- `.pre-commit-config.yaml`, `CITATION.cff`, `SECURITY.md`, `.github/FUNDING.yml` present.
- `pyproject.toml` version is bumped, description matches the current product, and all
  runtime deps are declared (`torch`, `numpy`, `datasets`, `fastapi`, `uvicorn`,
  `sympy`, `spacy`, `requests`, `beautifulsoup4`, `tqdm`).
- `uchi/__init__.__version__` matches `pyproject` version.

## 5. Fresh Install Test

- `pip install -e .` binds the `uchi` entrypoint with no missing packages.
- **Bundled artifacts present (release blocker).** Confirm these ship in the package:
  - `uchi/data/brain.uchi` — a built brain with a populated semantic index
    (NOT the absent/stub state; build it via `python -m uchi.incremental_builder` on a
    broad corpus if missing).
  - `uchi/data/{skipgram_emb.pt, decoder.pt, answerability.pt, chat_decoder.pt}`.
- **Behavior smoke test** — the three lanes and honesty must work out of the box:
  ```python
  from uchi import Uchi
  u = Uchi()
  print(u.ask("What is photosynthesis?"))              # grounded answer OR honest abstain
  print(u.ask("Who is the emperor of Neptune?"))        # must ABSTAIN (never confabulate)
  print(u.ask("hi there!"))                             # a conversational reply
  print(u.ask("What is 12 times 15; then add 8"))       # → 188, verified steps
  ```
  Verify: no `.n.01`/`.v.01` synset markers, no `<|...|>` control tokens in output;
  the unknown question abstains; the multi-step question reasons.

## 6. Prepare Release Commit

Once all checks pass, stage and commit. **Do NOT push** — the user runs the final push.
- Stage explicitly (avoid `git add .` which can pull in runtime artifacts):
  ```
  git add uchi/ tests/ docs/ benchmarks/ README.md CHANGELOG.md pyproject.toml mkdocs.yml
  ```
- Commit with a descriptive message (e.g. `chore: vX.Y.Z release readiness`).
- Print the push command for the user:
  ```
  git push -u origin <current-branch-name>
  ```
- If GitHub rejects the workflow scope, instruct the user to un-stage `.github/workflows/`
  and update it via the web UI.
