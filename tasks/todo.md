# Uchi 0.4.0 — Execution Order (locked)

Supersedes the previous version of this file, which tracked a completely
different, now-superseded architecture (`omni_router`, "Family C", etc. —
none of that exists in the current codebase). This is the current,
accurate order of remaining work, most recent session first. Full detail
for every item lives in `tasks/0.4.0 Itemized Deliverables.md` — this file
is the compressed, sequenced checklist, not a duplicate of it.

## Architecture, end to end

Two separate diagrams below — **training-time** (what produces the
checkpoints) and **runtime** (what `Core.ask()` actually does with them).
They're separate because they run at different times, on different
resources, and (for the proposer and verifier) are deliberately trained
with zero shared weights.

### Training-time: what feeds into what to produce each checkpoint

```
                                    ┌─────────────────────────────────────┐
                                    │      PROPOSER (FLUX) — 4 phases      │
                                    └─────────────────────────────────────┘

Phase 0  FineWeb-Edu (→OpenWebText fallback)
              │
              ▼  scripts/pretokenize.py
         data/{train,val}.bin  (uint32 memmap, GPU-bound throughput)
              │
              ▼  uchi/flux/train_v2.py   (--pruned-vocab)
Phase 1  ckpt_best.pt   [64.0M params, pruned 32,018-token vocab]
              │
              ▼  uchi/flux/sft_train.py   (--pruned-vocab)
         SQuAD + Dolly + CodeAlpaca + UltraChat-200k  ──┐
         (fair-budgeted, independent per-source share)   │→ sft_best.pt
              │                                          │  (conversational
              ▼                                          │   tone added)
Phase 2  sft_best.pt
              │
              ▼  uchi/flux/cot_distill.py   (--pruned-vocab)
         GSM8K + OpenOrca + Magicoder + CommitPackFT  ───┐
         (fair-budgeted, 4-way independent share)         │→ cot_best.pt
              │                                            │ (reasoning +
              ▼                                            │  code-change
Phase 3  cot_best.pt                                       │  understanding)
              │
              ▼  uchi/flux/qat_train.py   (ternary, CoT-preserving mix)
Phase 4  qat_best.pt
              │
              ▼  cp (train_all.sh's final step)
         flux_best.pt   ◄── canonical artifact Core() loads by default


                                    ┌─────────────────────────────────────┐
                                    │   VERIFIER (EntailmentClassifier)    │
                                    │   own embedding table, FULL (not     │
                                    │   pruned) tokenizer, never shares    │
                                    │   weights with the proposer above    │
                                    └─────────────────────────────────────┘

nyu-mll/glue (mnli) + stanfordnlp/snli  ──┐
(fair-budgeted, independent share)         │
                                            │
uchi/verifier_flywheel.py's exported ──────┼──► uchi/flux/verifier_train.py
real user corrections (jsonl, grows        │         │
over time, empty on a first run)          ─┘         │
                                                       ├──► verifier_best.pt
                                                       │    (the classifier)
                                                       │
                                                       └──► verifier_best.ood.pt
                                                            (Mahalanobis mean +
                                                            precision, fit on a
                                                            sample of the SAME
                                                            training run's
                                                            pooled latent reps)
```

### Runtime: what `Core.ask(question)` actually does, in order

```
from uchi import Uchi  →  MetaUchi(Core)         ◄── what users actually construct
                             │
                             ▼
                    Core.__init__ loads, in order:
                    1. SemanticIndex   ← premade brain (uchi/data/embeddings.pt)
                                          or whatever ingest()/learn() added
                    2. FactCheckOracle ← auto-detects verifier_best.pt if present
                                          (entailment_checker + its .ood.pt gate);
                                          numeric_checker stays off until
                                          fit_numeric_plausibility_checker() is
                                          explicitly called (real cost, opt-in)
                    3. FluxProposer    ← auto-detects flux_best.pt (→qat→cot→sft
                                          fallback order), tokenizer auto-matches
                                          pruned vs full vocab from the
                                          checkpoint's own embedding shape
                    4. ToolRegistry, GoalState, EpisodicMemory, VerifierFlywheel
                       (flywheel inert unless the same entailment_checker above
                       was actually loaded)

Core.ask(question)
    │
    ├─ fold in: episodic memory context (last N turns) + active goal state
    │           context + pending HitL-yield answer, if any
    │
    ├─ VerifierFlywheel.check_for_correction(prior_Q, prior_A, question)
    │     — does THIS turn contradict the LAST answer? (reuses the same
    │       entailment checker below; purely observational, logged only)
    │
    ├─ SwarmSynthesizer.answer()  — IQ-router gate (skip decompose if
    │     the question is atomic) → loop-guard check (skip if this exact
    │     question already failed delegation) → decompose → parallel
    │     sub-answers → aggregate
    │
    │     each sub-answer runs the FULL loop below:
    │
    │     GenerateAndGround.answer(question)
    │       │
    │       ├─ is the question asking anything SPECIFIC (proper noun/number)?
    │       │     no  → skip the two honesty gates below, proceed straight
    │       │           to candidate generation (a greeting doesn't need
    │       │           retrieval to have "worked" to be answerable)
    │       │     yes → both gates apply at full strength
    │       │
    │       ├─ gate 1: known_fraction(question) ≥ min_known?  (else ABSTAIN)
    │       ├─ retrieve top-k evidence from SemanticIndex
    │       ├─ gate 2: best evidence similarity ≥ min_sim?
    │       │     no + web_search enabled → live search → learn() it → retry
    │       │     no (still)                → ABSTAIN
    │       ├─ ev_texts = only evidence that cleared min_sim (weak/irrelevant
    │       │   matches don't count as "evidence" for the oracle either)
    │       │
    │       ├─ FluxProposer.propose(question, ev_texts, think=True)
    │       │     → candidate text (n_votes candidates, reflection retries)
    │       │
    │       ├─ FactCheckOracle.is_grounded(candidate, ev_texts)
    │       │     1. deterministic word-overlap (or no-evidence relaxation:
    │       │        only the candidate's SPECIFIC content needs support)
    │       │     2. entailment_checker.is_contradiction(evidence, candidate)?
    │       │          — gated by OOD detector first: input far from the
    │       │            classifier's training distribution → treated as
    │       │            "no opinion", verdict falls back to step 1's result
    │       │     3. numeric_checker.is_plausible(number)? (if fitted)
    │       │     — 2 and 3 can only turn a PASS into a REJECT, never the
    │       │       reverse; if 1 already rejected, 2/3 are moot
    │       │
    │       ├─ Devil's Advocate critique (logical soundness, separate axis
    │       │   from grounding — same proposer, adversarial framing)
    │       │
    │       └─ if nothing grounds: Empirical Synthesis Loop (REPLOracle
    │           executes proposer-written Python) → else extractive
    │           fallback → else honest ABSTAIN
    │
    ├─ <|tool_call|> / <|tool_call_async|> dispatch (ToolRegistry, LoopGuard-
    │     checked, logged) — filesystem, scratchpad, web search, macros
    │
    ├─ HitL yield check (explicit <|yield_to_user|>, or loop-guard auto-
    │     escalation) — pauses instead of guessing if genuinely stuck
    │
    └─ save turn to EpisodicMemory (unless an external conversation_context
         was supplied, e.g. the REST API's per-request isolation)
              │
              ▼
      response_normalizer.normalize(raw)  →  returned to the caller
```

### The one cycle that closes training back into runtime

```
 runtime: VerifierFlywheel logs a confirmed correction
              │
              ▼
 export_training_examples()  →  corrections.jsonl  (real, never fabricated)
              │
              ▼
 next `train_verifier.sh` run:  --flywheel-path corrections.jsonl
              │                  (folded in as a genuine 3rd fair-budgeted
              │                   source alongside MNLI/SNLI)
              ▼
 a better-calibrated verifier_best.pt  →  promoted  →  back into runtime
```

## Where things actually stand right now

- Phase 1 (pretrain) — **done**. 64.0M params, pruned 32,018-token vocab.
- Phase 2 (SFT) — **running now**, step ~250/742, healthy, no errors.
  `nvidia-smi` confirms the GPU is fully committed to this.
- Phase 3 (CoT), Phase 4 (QAT) — **not started**.
- Item 17 (verifier upgrade) — code, data pipeline, and tests **fully
  built and CPU-verified**; training **not yet run** (needs the GPU, which
  Phase 2–4 has first).

## 1. Let training finish — no action, just don't interrupt it

- [ ] Phase 2 (SFT) completes
- [ ] Phase 3 (CoT distillation) — now includes CommitPackFT as a 4th
      fair-budgeted source alongside GSM8K/OpenOrca/Magicoder
- [ ] Phase 4 (ternary QAT)

## 2. Promotion (manual step, not automatic)

- [ ] Copy the final artifact from the isolated `v040_phaseN/` directories
      into the default `uchi/flux/checkpoints/` path — `Core()` only
      searches the default path, so nothing happens on its own
- [ ] Re-verify the inference-time pruned-vocab fix
      (`build_generate_fn`/`EntailmentChecker.load`-style auto-detection)
      against the real final artifact, not just Phase 1's intermediate one

## 3. Benchmarking — real, unstarted work, not a formality

- [ ] Zero-regression vs. v0.3.0 (MMLU/SWE-bench through `MetaUchi`) —
      named 0.4.0 exit criterion, never run
- [ ] Web Navigation Baseline — named exit criterion, never run this
      entire session
- [ ] Trustworthiness benchmark (SQuAD 2.0) on the new model — the real
      release gate per this project's own doctrine, not MMLU/SWE-bench
- [ ] Resume the PyPI-vs-local-branch parity check — explicitly paused
      mid-session when training started; never resumed

## 4. Verifier training (Item 17) — sequenced after Phase 4, same GPU

- [ ] Run `scripts/train_verifier.sh` for real (MNLI+SNLI, ~200K examples)
- [ ] Fit the OOD detector on the trained model's own latent space
      (already wired into `verifier_train.py`, runs automatically at the
      end of training)
- [ ] Held-out adversarial validation (the "330m vs 500m" style test set)
      — **hard gate**, exit criterion #7 in the deliverables doc. Nothing
      below this line happens until it passes.
- [ ] Once validated: flip it on (`Core.__init__` already auto-detects the
      checkpoint at `uchi/flux/checkpoints/verifier/verifier_best.pt` —
      no code change needed) and watch `oracle.layered_veto_log` closely
      in real use before fully trusting it
- [ ] Only after the verifier has a real production track record: revisit
      verifier-as-tool (0.7.0-adjacent, pulled forward if it still makes
      sense) and the rejection-filter proposer-training idea — **not**
      before, and **not** joint/shared-weight training, RL-style reward
      optimization against the verifier, or shared latent spaces (all
      explicitly rejected, not just deferred)

## 5. Documentation — real updates once the model actually changes

- [ ] README/docs currently describe FLUX as "~116M-class" throughout —
      that's v0.3.0. Update once the new (64M, pruned-vocab) model is
      validated and promoted, not before
- [ ] Re-check the "How It Connects" diagram still matches reality

## 6. Release readiness — one full pass, not the individual pieces checked ad hoc

- [ ] Run the full `.agents/skills/release_readiness/SKILL.md` checklist
      end-to-end against the finished, promoted model
- [ ] Item 16 bonus objectives remain explicitly non-blocking — ship
      whatever fraction is done, don't gate on the rest

## 7. Release commit

- [ ] Prepare the release commit
- [ ] **Do not push or merge to main without explicit confirmation** —
      same standing rule as every other hard-to-reverse action this session
