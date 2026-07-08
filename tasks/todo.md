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
              ▼  uchi/flux/qat_train.py   (--pruned-vocab, ternary, CoT-preserving mix)
Phase 4  qat_best.pt   [running — data-bin blocker resolved, see
                        "Where things actually stand" below]
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
                                          explicitly called (real cost, opt-in);
                                          relational_checker (Item 18) always on,
                                          no training needed
                    3. FluxProposer    ← auto-detects flux_best.pt (→qat→cot→sft
                                          fallback order), tokenizer auto-matches
                                          pruned vs full vocab from the
                                          checkpoint's own embedding shape
                    4. TaskConfigCache ← always on, no training needed (ODUSP-
                                          backed dynamic-N vote recall)
                    5. ToolRegistry, GoalState, EpisodicMemory, VerifierFlywheel
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
    │       ├─ n_votes = dynamic, not a fixed 3: baseline from iq_router's
    │       │     complexity score, refined by TaskConfigCache.recall_n()
    │       │     if it has a confident recommendation for this question's
    │       │     STRUCTURAL shape (comparison/multi-part/enumeration/length
    │       │     -- not its exact text, which is what makes this recur
    │       │     often enough to be useful, unlike claim-level memoization)
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
    │       │     4. relational_checker.is_contradicted(candidate, ev_texts)?
    │       │          — deterministic transitive-relation veto (Item 18),
    │       │            active unconditionally, no training/fitting needed
    │       │     — 2, 3, and 4 can only turn a PASS into a REJECT, never
    │       │       the reverse; if 1 already rejected, the rest are moot
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
- Phase 2 (SFT) — **done**. 743 steps, 787.6 min, val loss 3.42, PPL 30.5
  (down from Phase 1's PPL 61.1 — a real improvement, not just more training).
- Phase 3 (CoT) — **done.** Proof run succeeded first (all 4 sources
  loaded exactly fairly, pruned-vocab tokenizer confirmed working), then
  the full run completed cleanly: 4/4 epochs, val loss 2.30, PPL 10.0 at
  the final (best) epoch, monotonically improving epoch over epoch
  (13.9 → 10.9 → 10.2 → 10.0) — no divergence, no red flags. Final
  checkpoint: `uchi/flux/checkpoints/v040_phase3/cot_best.pt`.
- Phase 4 (QAT) — **done. All four phases of 0.4.0's FLUX retraining are
  now complete.** The `--data-bin` blocker was resolved while Phase 3 was
  still training, so Phase 4 launched immediately once the GPU freed:
  proof run (20 steps) succeeded first, then the full run (600 steps)
  completed cleanly — final: CoT val loss 2.29 (PPL 9.9, best), text val
  loss 4.72 (PPL 112.4). Checkpoints at
  `uchi/flux/checkpoints/v040_phase4/` (`qat_00200.pt`, `qat_00400.pt`,
  `qat_00600.pt`, `qat_best.pt`). GPU is now fully idle.
  **Next real step is Section 2 (Promotion) below — not automatic, needs
  a decision on when to copy the final artifact to the default path.**
- Item 17 (verifier upgrade) — **training done.** Proof run first (300
  examples, 1 epoch, GPU): confirmed all 3 sources load and fair-budget
  correctly, no crashes, OOD detector fit succeeded. Full run
  (`--max-examples 200000 --epochs 2`, matching `train_verifier.sh`'s
  real parameters): 66,666 examples each from MNLI/SNLI/synthetic
  multi-hop (fair-budgeted correctly), epoch 1 val acc 54.0%, epoch 2 val
  acc 59.0% (best), OOD detector fit on 5,000 representations. Checkpoint
  loads correctly via `EntailmentChecker.load()`, full suite still 346
  passing. **Real, concrete finding, not a formality**: a quick sanity
  check against the flagship motivating case from Item 17's own design
  ("The Eiffel Tower is 330 meters tall" vs "...25 meters tall", same
  evidence) came back `is_contradiction() == False` — **the freshly
  trained verifier missed it.** Unsurprising at 59% val accuracy (barely
  above the 33% random baseline for 3-way classification), but a concrete
  signal that held-out adversarial validation (below) is doing real
  work, not a checkbox — proceed to it before drawing any conclusion
  about whether this classifier is usable.
  **Not wired into the live oracle yet** — held-out adversarial
  validation (exit criterion #7) is a hard gate that must pass first,
  per Section 4 below.
- **`verifier_train.py` gained a third, unconditional data source since
  the entry above was written: synthetic multi-hop transitive examples**
  (`generate_multihop_examples`, RuleTaker/ProofWriter-style), fair-budgeted
  alongside MNLI/SNLI. Motivation: MNLI/SNLI are both single-premise —
  neither teaches "combine two stated facts into a transitive conclusion"
  as a skill at all, which is exactly the gap Item 18 (below) found and
  closed deterministically. This is the corresponding neural-side fix,
  discussed explicitly as the "smarter, neural approach" to semantic
  comparison the user asked about, with the honest caveat that a small
  from-scratch classifier trained on this won't match a frontier LLM's
  reasoning reliability — it's a real improvement over zero multi-hop
  training signal, not a solved problem.
  **A real bug was found and fixed during development, caught by
  cross-verification rather than trusted on sight**: the relation was
  originally being re-picked on every `fact()` call instead of once per
  generated example, so a premise pair could end up about two *unrelated*
  attributes (e.g. "richer" for one fact, "higher" for the other) —
  silently producing wrong contradiction labels, since transitivity
  across unrelated attributes means nothing. Caught by checking the
  generator's own output against `RelationalTransitivityChecker`
  (Item 18) before trusting it, exactly the "verify, don't assume"
  discipline used everywhere else this session. Fixed, then re-verified
  with **zero mismatches across 1000+ generated examples**, restricted to
  the vocabulary the deterministic checker can actually parse (the
  generator deliberately uses a broader relation vocabulary than the
  checker's hardcoded list, so the neural model sees variety the checker
  doesn't know — comparing against words the checker *can* parse is the
  fair, apples-to-apples validation).
  **This debugging pass also found two real, independent gaps in
  `relational_reasoning.py` itself** (Item 18), not just the generator —
  see the Item 18 entry below.
- **Sentence-level MCTS verifier cascade (post-0.4.0 design) — scoring
  rule pinned down, latency benchmark harness built and CPU-validated;
  real GPU measurement blocked until Phase 3/4 free the GPU.**
  Design lands as a generalization of dynamic-N voting: instead of one
  flat round of N candidate answers, a PUCT tree search over
  *complete-sentence* nodes (never partial tokens — the entailment
  classifier is trained on complete MNLI/SNLI propositions, so scoring
  a fragment is out of its distribution the same way a raw softmax
  threshold is meaningless before calibration). **Scoring rule, pinned
  down and encoded in the benchmark harness:** the verifier always
  scores the *full accumulated path* `s_1..t`, never the isolated new
  sentence — scoring the increment alone swaps the fragment mismatch
  for a coreference mismatch ("It was built in 1889" has no antecedent
  without the prior sentence).
  `scripts/benchmark_mcts_latency.py` — exhaustive depth×branching tree
  (no PUCT pruning yet; this is the safe worst-case upper bound to size
  K against, since real PUCT only ever calls the proposer/verifier
  *fewer* times than full expansion), forced to `device="cpu"`
  explicitly so it can run safely while proposer training saturates the
  GPU. Checked `nvidia-smi` before running anything on shared hardware:
  100% GPU compute utilization from Phase 3, only 41% memory — meaning
  no compute headroom for a concurrent GPU job even though there was
  memory room, so this stayed CPU-only rather than risk slowing down
  training. **CPU numbers (depth=3, branching=3, real Phase 2
  `sft_best.pt` checkpoint, random-weight verifier of the real
  production shape — architecture determines latency, not whether
  weights are trained):** 39 nodes, proposer 39 calls / 58.1s total
  (1.49s/call avg), verifier 39 calls / 3.35s total (86ms/call avg),
  61.5s wall time. **Proposer cost dominates verifier cost by ~17x** —
  the verifier is comparatively free; latency-reduction effort belongs
  on proposer calls, not verifier calls. Candidate reductions to
  evaluate once real GPU numbers are in: batch sibling proposer calls
  (all `branching` children of a node share the same prefix — a single
  batched forward pass instead of `branching` serial ones is the
  highest-leverage lever, since GPUs specifically reward batched
  parallel work over serial small calls), KV-cache reuse across the
  shared prefix, non-uniform branching (wider at the root, narrower
  deeper), and a real PUCT budget (K simulations, not full expansion)
  in place of this benchmark's exhaustive baseline. **Next action**:
  re-run `scripts/benchmark_mcts_latency.py --device cuda` the moment
  Phase 3/4 finish and the GPU is free, before writing the PUCT loop
  itself.
- **Dynamic-N self-consistency voting — done, active by default, no
  training needed.** `uchi/task_config_cache.py`: ODUSP (`UniversalPredictor`)
  recalls a recommended vote count keyed by a question's structural
  signature (multi-part/comparison/enumeration/length), not its exact
  text — this is a better fit for ODUSP's pattern-recall strength than the
  original claim-memoization idea, since structural shapes recur across
  completely different topics far more often than exact factual claims
  do. Verified empirically before wiring it in (same discipline as the
  numeric-plausibility rewrite): trained on a handful of examples, it
  correctly recalled N=1 for a *never-seen* simple question and N=8 for a
  *never-seen* complex/comparison question, confirming it generalizes on
  structure rather than memorizing text. Wired into
  `GenerateAndGround.answer()`, replacing the old hardcoded `n_votes=3`;
  falls back to a complexity-score baseline (`iq_router.estimate_complexity`)
  when unfitted or unconfident. Safe in both directions — unlike the
  oracle's layered vetoes, a vote count is a compute-budget knob, not a
  correctness gate, since every candidate still goes through the full,
  unchanged oracle cascade regardless of how many were generated. Wired
  into `Core.__init__` by default (needs no training, unlike the
  entailment classifier). 11 new tests
  (`test_task_config_cache.py`, `test_dynamic_n_voting.py`), 317 passing
  overall.
- **Relational transitivity veto (0.4.0 Item 18) — done, active by
  default, no training needed.** Closed a real gap, demonstrated
  empirically before writing any code: given evidence "A is taller than
  B" and "B is taller than C", `FactCheckOracle` accepted the valid
  conclusion ("A is taller than C"), the reversed/wrong one ("C is
  taller than A"), and outright nonsense ("A is taller than A") —
  identically, all three, because word-overlap and the entailment
  classifier both check surface consistency against evidence, not the
  logical validity of a conclusion synthesized from combining premises.
  Separately confirmed generation-side: FLUX at the current Phase
  2/SFT-only checkpoint didn't even attempt the comparison when asked
  directly, producing generic off-topic text instead — a second, distinct
  gap from the verification-side one. `uchi/relational_reasoning.py`:
  `RelationalTransitivityChecker`, pure deterministic pattern-matching +
  per-relation transitive closure, explicitly narrow in scope (simple
  comparative relations only — taller/shorter, older/younger,
  before/after, etc. — not general logical-inference verification, which
  is a much harder, open problem). Wired into `FactCheckOracle` as a
  third additive-only veto layer and into `Core.__init__` unconditionally
  (no training/fitting needed, unlike the entailment classifier). Fixed
  the exact motivating scenario end-to-end (verified through the real
  `Core()` instance, not just the isolated checker). Added as Item 18
  in `tasks/0.4.0 Itemized Deliverables.md` with a new exit criterion (#8).
  **Extended twice since the initial build:**
  - **Numeric-attribute extraction** (`extract_attribute_value`): real
    evidence is far more likely to state "Building A is 442 meters tall"
    and "Building B is 330 meters tall" separately than to state an
    explicit comparative sentence — the original comparative-sentence-only
    version missed this, probably the more common real case. Now derives
    comparative edges directly from pairs of bare numeric measurements on
    the same attribute, composing into the *same* transitive-closure graph
    as explicit comparative sentences (verified: a 3-entity chain mixing
    one numeric-derived edge and one comparative-sentence-derived edge
    resolves correctly).
  - **Two regex gaps found and fixed** while cross-verifying the synthetic
    multi-hop generator (above) against this checker: (1) a trailing
    clause (e.g. ", based on the available data.") was getting swallowed
    into the entity name, so "Building A" and "Building A, based on the
    available data" failed to match as the same graph node — object
    capture groups now stop at the first comma/terminal punctuation
    instead of requiring end-of-string; (2) the "Compared to Y, X is
    COMP." phrasing (no literal word "than" at all) wasn't recognized —
    added `_COMPARED_TO_RE` alongside the existing `_COMPARATIVE_THAN_RE`.
    Both are real, independent improvements to real-world coverage, not
    just artifacts of testing — found because the verification step was
    taken seriously rather than skipped.
  - Hardcoded-word-list concern raised directly by the user ("I never want
    to hardcode anything") — addressed honestly: the syntactic patterns
    (regexes) are unavoidable for a deterministic system, but the semantic
    knowledge (antonym pairs like taller/shorter) is currently a small,
    manually-typed list. Proposed fix, not yet built: source a
    comprehensive antonym table from WordNet once, offline, and ship it as
    a static data file rather than a live NLTK dependency — deferred as
    polish, not blocking, since it doesn't unblock anything else.
  29 tests total across `test_relational_reasoning.py` (24, up from 14)
  and the new `test_verifier_multihop.py` (5), 346 passing overall.

**Bugs found and fixed transitioning into Phase 3** (same class as the
pruned-vocab fixes already made to `sft_train.py`/`build_generate_fn`
earlier in this session, just not yet applied to Phases 3–4): both
`cot_distill.py` and `qat_train.py` unconditionally used the full ~100K
tokenizer regardless of the checkpoint's actual vocab — since Phase 2's
`sft_best.pt` uses the pruned 32,018-token vocab, running Phase 3 without
this fix would have corrupted or crashed training immediately on the first
out-of-range embedding index. Both scripts now accept `--pruned-vocab`
(same pattern as `sft_train.py`) and `--checkpoint-dir` (isolate
proof/experimental runs from production checkpoints). Verified via
`--help` and the full test suite (306 passing) before launching anything.

**Phase 4 blocker — fixed while Phase 3 was training (CPU/IO-only work,
verified not to compete with the GPU or the training process's own CPU
usage — 20 cores, load average had headroom).** `scripts/pretokenize.py`
now accepts `--pruned-vocab` (same pattern as `sft_train.py`/
`cot_distill.py`/`qat_train.py`): the existing fast batched Rust encode
path (`enc.encode_ordinary_batch`) is unchanged for speed, with the
pruned-vocab remap (`PrunedVocab.remap`) applied as a cheap post-hoc
step on the returned ids, before the special-token shift. Verified, not
assumed: smoke-tested both paths (pruned and unpruned) end-to-end,
loaded the real output `.bin` files back and confirmed every token id
is within the pruned vocab's actual range (`max 31940 < 32018`) —
exactly the property that matters, since an out-of-range id is what
crashes the embedding lookup. Unpruned path re-verified unchanged (max
id 99988, full vocab). Full test suite: 317 passing.
Real, full-scale re-tokenization (80M train / 1M val tokens, matching
Phase 1's original scale) **launched in the background** while Phase 3
trains, writing to a new `uchi/flux/data_pruned/` directory (not
overwriting the existing full-vocab `.bin` files at `uchi/flux/data/`,
in case anything else still references them) — log at
`/tmp/pretokenize_pruned_full.log`. This means Phase 4 should have its
`--data-bin` ready the moment Phase 3 finishes and Phase 4 is next,
rather than needing this run started only then.

**Proprioception (Approach 1) — corrected, re-validated, and integrated
into the main pipeline.** Second validation pass with corrected
methodology found a real problem in the first pilot: the reference set
used a "Context:...Question:...Answer:" template that matches neither
`_ANSWER` nor (more importantly) the real evidence-grounded generation
path, which sends FLUX the raw question alone with no wrapper at all
(confirmed directly in `propose()` — deliberate, already-documented
behavior, not a bug). Rebuilt `uchi/proprioception.py` +
`scripts/fit_proprioception.py` using real questions from FLUX's actual
CoT training sources (GSM8K/OpenOrca/Magicoder/CommitPackFT, 480 fit +
120 held-out for calibration), matching the exact raw-question shape.
Result: clean ~4x separation (in-distribution ~50–60, genuinely OOD
~201–228), calibrated threshold (123.30, from the 95th percentile of
real held-out distances, not a reused default) sitting cleanly between
them — a real, validated result, not just a promising lead.
**Wired into `Core.__init__`/`GenerateAndGround` as an additive-only
signal**: loads a second, standalone FLUX instance for hidden-state
access (since `FluxProposer` only exposes a `generate_fn` closure, not
the raw model — a real memory cost, only paid if the fitted artifact
exists); an "unfamiliar" verdict can only raise `n_votes` to the max
bucket, never lower it below whatever complexity/`TaskConfigCache`
already decided. All three of proprioception/model/tokenizer must be
present or it's silently inert — same graceful-degradation contract as
every other optional component. 7 new tests
(`tests/test_proprioception.py`), 353 passing overall.
**Reconfirmed once more, explicitly, because this keeps needing
restating**: this is a pre-generation, question-only gate. It never
sees a generated claim, so it structurally cannot check factual
correctness — the verifier remains required regardless of how well this
performs. See `tasks/proprioception_experiment.md` for the full
reasoning and the decisive test that established this.
**[x] Fully verified end-to-end, not just in isolation**: saved artifact
at `uchi/flux/checkpoints/proprioception.pt`; loaded through a real
`Core()` instance (`proprioception`/`_proprioception_model` both
populated); tested real, complete reference-shaped questions (correctly
`unfamiliar=False`) against genuinely OOD input (correctly
`unfamiliar=True`) through the live-loaded detector, not just the
fitting script's own printed output. One false alarm caught and
resolved during this check: an initial manual test used a truncated
version of a CommitPackFT question (missing the diff text the real
question always includes), which produced a different distance and
looked like a bug — re-tested with the actual complete question text
sampled from the same source and it matched expectations exactly. Full
suite: 353 passing.

## 1. Let training finish — no action, just don't interrupt it

- [x] Phase 2 (SFT) completes
- [x] Phase 3 (CoT distillation) — done, 4/4 epochs, val loss 2.30/PPL 10.0,
      included CommitPackFT as a 4th fair-budgeted source alongside
      GSM8K/OpenOrca/Magicoder
- [x] Resolve Phase 4's `--data-bin` pruned-vocab blocker (see above) —
      `pretokenize.py` fixed and verified; full re-tokenization completed
      cleanly (verified: real output token IDs within pruned vocab range)
- [x] Confirm the background re-tokenization run finished cleanly before
      pointing Phase 4 at `uchi/flux/data_pruned/` — confirmed
- [x] Phase 4 (ternary QAT) — **done.** 600/600 steps, CoT val loss 2.29
      (PPL 9.9, best), text val loss 4.72 (PPL 112.4). All four proposer
      phases complete. GPU now fully idle.

## 2. Promotion (manual step, not automatic)

- [x] Copy the final artifact from the isolated `v040_phaseN/` directories
      into the default `uchi/flux/checkpoints/` path. **Found before
      promoting, not assumed away**: the default path already held a
      full, live set of checkpoints from an earlier (Jul 2–4) attempt —
      `flux_best.pt` there was the **full 100,300-token vocab**
      architecture (pre-pruning), confirmed by directly inspecting
      `embedding.weight`'s shape before touching anything. Backed up
      (moved, not deleted) to `uchi/flux/checkpoints/pre_pruning_backup/`
      rather than overwritten, so the promotion stays reversible. Promoted
      the full new lineage for fallback-chain consistency, not just
      `flux_best.pt` alone: `v040_phase2/sft_best.pt` → `sft_best.pt`,
      `v040_phase3/cot_best.pt` → `cot_best.pt`, `v040_phase4/qat_best.pt`
      → both `qat_best.pt` and `flux_best.pt`. Checksummed every copy
      against its source before trusting it (all matched).
- [x] Re-verify the inference-time pruned-vocab fix against the real
      final artifact — confirmed `flux_best.pt`'s `embedding.weight` is
      `[32018, 768]` (pruned, not `[100300, 768]`), then loaded a real
      `Core()` instance and ran a live `ask()` end-to-end successfully.
      Full test suite re-run after promotion: 346 passing, unaffected.

## 3. Benchmarking — real, unstarted work, not a formality

- [ ] Zero-regression vs. v0.3.0 (MMLU/SWE-bench through `MetaUchi`) —
      named 0.4.0 exit criterion, never run
- [ ] Web Navigation Baseline — named exit criterion, never run this
      entire session
- [ ] Trustworthiness benchmark (SQuAD 2.0) on the new model — the real
      release gate per this project's own doctrine, not MMLU/SWE-bench
- [x] Resume the PyPI-vs-local-branch parity check — **done, clean.**
      Published `uchi-python` 0.3.0 matches local `pyproject.toml`'s
      version. Downloaded the real wheel, compared file lists and diffs
      against the local branch: **zero files present in the PyPI package
      but missing locally** (nothing accidentally deleted/regressed), 26
      new files locally not yet published — all recognizable as this
      session's and the broader 0.4.0 cycle's real work (`verifier_model.py`,
      `proprioception.py`, `relational_reasoning.py`, `task_config_cache.py`,
      `meta.py`, `goal_state.py`, `tool_calling.py`, etc.). Diff'd the 51
      shared files too: the largest changes (`simple.py` 412 lines,
      `oracle.py` 172, `generate_and_ground.py` 114) all correspond
      directly to documented work (layered vetoes, dynamic-N, proprioception
      wiring) — purely additive/enhancing drift, not accidental. Also
      confirmed the dev environment itself is clean: `uchi-python` is
      **not** pip-installed in this venv at all; `import uchi` resolves
      directly to the local branch's source — no risk that any testing
      this session ran against a stale installed package instead of the
      real, current code.

## 4. Verifier training (Item 17) — sequenced after Phase 4, same GPU

- [x] Run verifier training for real — **done**. 66,666 examples each
      from MNLI/SNLI/synthetic multi-hop, epoch 1 val acc 54.0%, epoch 2
      val acc 59.0% (best). `--flywheel-path` would make it a fourth
      source once real corrected conversations exist.
- [x] Fit the OOD detector on the trained model's own latent space —
      done automatically at the end of training, fit on 5,000
      representations, saved to `verifier_best.ood.pt`.
- [ ] Temperature-scaling calibration on the trained classifier — needed
      both as its own exit criterion and as a hard precondition for
      Item 5's PUCT search (below) to safely use the classifier's score
- [x] Held-out adversarial validation — **run, and it FAILED. Hard gate
      not passed. Do not wire this into the live oracle.**
      16 hand-built cases across numeric substitution, semantic negation,
      and held-out multi-hop transitive chains — deliberately different
      entities/topics than any training data, synthetic generator output,
      or demo examples used this session. Result: **0 of 8 real
      contradiction cases caught.** The "50% accuracy" is coincidental —
      the classifier said "no contradiction" for literally every case,
      which happened to be correct for the 8 true non-contradictions and
      wrong for all 8 real contradictions.
      **Root cause has two independent parts, both confirmed by
      inspecting raw probabilities/distances directly, not guessed:**
      1. The OOD gate fires on every single test case (distances
         15.5–18.1 vs. threshold 3.0) — ordinary, in-domain sentences are
         being treated as out-of-distribution and suppressed to "no
         opinion." Same threshold-miscalibration pattern found in the
         proprioception experiment's Approach 1, now confirmed in the
         verifier's own freshly-fit detector too — the default threshold
         doesn't transfer to a real fitted distribution and needs its
         own calibration, not reuse.
      2. **Even bypassing the OOD gate, the raw classifier is wrong**:
         "Golden Gate Bridge spans 4,200 meters" (should score
         contradiction highest) instead scores entailment=0.41 vs.
         contradiction=0.30. Same pattern on two other spot-checked
         cases. Not an OOD artifact — the classifier itself isn't yet
         discriminating these adversarial cases correctly, consistent
         with only 59% overall validation accuracy after 2 epochs.
      **Consistent with the earlier warning sign**: a quick sanity check
      right after training (330m vs. 25m Eiffel Tower case) already
      missed the same way — this isn't a fluke of the specific test
      cases chosen.
      **Next steps, not yet done**: recalibrate the OOD threshold against
      real validation data (cheap, should happen first, clearly wrong as
      configured); separately, the classifier's own discriminative
      accuracy likely needs more training (more epochs/examples) or
      architecture attention before revisiting this gate — calibration
      alone won't fix a classifier that ranks entailment above
      contradiction on clear contradiction cases.
      **Live safety action taken, not just noted**: `Core.__init__`
      auto-detects any `verifier_best.pt` present at the default path
      with no gate checking whether it's been validated — confirmed
      directly that constructing `Core()` right after training completed
      would have silently loaded this failed checkpoint into the live
      oracle. Moved the checkpoint and its OOD detector aside to
      `uchi/flux/checkpoints/verifier/failed_validation_v1/` (not
      deleted — fully reversible) so `Core()` gracefully falls back to
      `entailment_checker=None`, its own designed degradation path.
      Re-verified: `Core()` now correctly shows `entailment_checker=None`,
      `ask()` still works, full suite still 346 passing.
- [x] Found and fixed the actual root cause of the ~3-hour first run:
      `load_verifier_examples()` tokenized every premise and hypothesis
      one at a time via `tokenizer.encode_text()` — 2×N individual
      tiktoken calls for N examples. Batched via `encode_ordinary_batch`
      (same fast Rust path `pretokenize.py` already uses), reassembling
      special-token wrapping and padding after. Verified correct (proof
      run, 3,000 examples, ~2 minutes total including HF streaming +
      training — dramatically faster) before trusting it, full suite
      still 346 passing.
- [~] Retraining now in progress with the fix: 600,000 examples (200K
      each MNLI/SNLI/synthetic multi-hop — using far more of the ~390K
      real MNLI / ~550K real SNLI pool than the first attempt's 66K
      each), 6 epochs (up from 2) — addressing the likely undertraining
      root cause behind 59% val accuracy and the failed adversarial
      validation. Log `/tmp/verifier_full_v2.log`.
      **Progress checked against `scripts/verifier_adversarial_validation.py`
      as each epoch completes, not just at the end:**
      - Epoch 2 (no OOD detector attached yet, pure classifier signal): 0/8 real contradictions caught.
      - Epoch 3: **6/8 real contradictions caught (75%), 81.2% overall accuracy** — real,
        substantial improvement. One new false positive appeared ("drug
        reduced symptoms" incorrectly flagged) — less severe than a
        missed contradiction given the additive-only design (costs an
        unnecessary abstention, not a wrong acceptance), but worth
        tracking across the remaining epochs.
      - Epochs 1-3 pacing: ~4h25m, ~4h24m, ~5h01m — consistent, no
        stalls (each independently confirmed via utime/network checks
        when timing looked off).
      **Once all 6 epochs done**: recalibrate the OOD threshold against
      real validation data (the reused default of 3.0 was confirmed
      wrong on the first attempt), then re-run the same held-out
      adversarial validation set before any promotion.
- [ ] Once a retrained version actually passes: promote it back to
      `uchi/flux/checkpoints/verifier/verifier_best.pt` (`Core.__init__`
      already auto-detects it there — no code change needed) and watch
      `oracle.layered_veto_log` closely in real use before fully trusting it
- [ ] Only after the verifier has a real production track record: revisit
      verifier-as-tool (0.7.0-adjacent, pulled forward if it still makes
      sense) and the rejection-filter proposer-training idea — **not**
      before, and **not** joint/shared-weight training, RL-style reward
      optimization against the verifier, or shared latent spaces (all
      explicitly rejected, not just deferred)

## 5. Sentence-level MCTS verifier cascade (post-0.4.0 design, sequenced after Item 4)

Full design reference — everything needed to pick this back up without
re-deriving it. Depends on Item 4 (verifier trained and validated) and
real GPU latency numbers (blocked on Phase 3/4 finishing, see the
"Where things actually stand" entry above for the CPU-benchmarked
numbers already in hand). Generalizes the dynamic-N self-consistency
voting already shipped: instead of one flat round of N candidate
answers, a PUCT tree search over complete-sentence nodes, reusing the
proposer and verifier exactly as trained — no new training objective,
no shared weights, no gradient flow between them at any point.

**One correction made before this got written down**: an earlier
discussion of this design conflated two distinct, already-separate
mechanisms under the name "ODUSP." Keeping them straight matters for
whoever picks this up next:
- `OODDetector` (`uchi/flux/verifier_model.py`) — Mahalanobis distance
  over the entailment classifier's own pooled latent representation.
  This is the actual, already-built OOD gate, already wired into
  `EntailmentChecker.is_contradiction()`. **This is what prunes MCTS
  branches below**, not ODUSP.
- ODUSP / `UniversalPredictor` (via `uchi/task_config_cache.py`) — a
  separate, trie-based sequence-credibility mechanism, currently used
  *only* for dynamic-N vote-count recall. It plays no role in this
  design as currently scoped. (A future idea, not yet built or
  scoped: using ODUSP's sequence-prediction credibility as an
  *additional*, complementary plausibility signal over the discrete
  text path itself, alongside `OODDetector`'s latent-space check —
  worth keeping distinct from this item if pursued.)

### Architecture

```
                    [USER QUESTION] + [RETRIEVED EVIDENCE]
                                   │
                                   ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │                    MetaUchi (orchestrator)                        │
   │  1. K (simulation budget) = min(TaskConfigCache-recalled N,       │
   │     hard ceiling from the real measured latency benchmark —       │
   │     never an unmeasured "IQ-quotient" number)                     │
   │  2. Strictly discrete text handoffs between Proposer and          │
   │     Verifier -- no shared gradients, no shared embedding table    │
   │                                                                    │
   │  ┌───────────────────── PUCT loop, repeated up to K times ─────┐  │
   │  │                                                              │  │
   │  │   current path s_t = "The Eiffel Tower is in Paris."         │  │
   │  │              │                                               │  │
   │  │              ▼                                               │  │
   │  │      FluxProposer.propose(...)  (unchanged, Cross-Entropy    │  │
   │  │      trained) proposes candidate NEXT COMPLETE SENTENCES,    │  │
   │  │      not tokens -- matches the verifier's own MNLI/SNLI      │  │
   │  │      training distribution (complete propositions)           │  │
   │  │              │                                               │  │
   │  │     a1: "It is 330m tall."      a2: "It was built in 1999."  │  │
   │  │              │                          │                    │  │
   │  │      full path assembly:        full path assembly:          │  │
   │  │      s_t ⊕ a1 (s_t and a1        s_t ⊕ a2                    │  │
   │  │      concatenated -- NEVER                                   │  │
   │  │      score a1 alone: no                                      │  │
   │  │      antecedent for "It")                                    │  │
   │  │              │                          │                    │  │
   │  │              ▼                          ▼                    │  │
   │  │      OODDetector.is_ood?         OODDetector.is_ood?          │  │
   │  │        no  → continue             yes → Ω = -inf, PRUNED     │  │
   │  │              │                                               │  │
   │  │              ▼                                               │  │
   │  │      EntailmentChecker scores s_t ⊕ a1 (full path) against    │  │
   │  │      evidence -- TEMPERATURE-SCALED first (not yet built;     │  │
   │  │      hard precondition, see below), giving a calibrated v     │  │
   │  │              │                                               │  │
   │  │              ▼                                               │  │
   │  │      PUCT backup: update Q(s_t, a1) using v                  │  │
   │  └───────────────────────────────────────────────────────────────┘ │
   │  3. After K simulations: return the path with the highest visit  │
   │     count N(s,a), not just the highest single v (standard        │
   │     AlphaZero move-selection rule, more robust than argmax-v)     │
   └────────────────────────────────┬──────────────────────────────────┘
                                    ▼
                     [FINAL, DISCRETE TEXT RESPONSE]
```

### The math, exactly as verified (no unresolved errors, unlike the
### rejected continuous-latent-space version earlier this design cycle)

```
State s_t:  the accumulated text so far, complete sentences only,
            never a partial token sequence.
Action a:   one candidate next complete sentence.
Transition: s_{t+1} = s_t ⊕ a   (concatenation)

1. Proposer prior (unchanged training, Cross-Entropy):
       P(a | s_t) = FluxProposer.propose(s_t, evidence)

2. Verifier value -- SCORES THE FULL ACCUMULATED PATH, pinned down:
       v(s_t ⊕ a) = 1 - P_contradiction(s_t ⊕ a | evidence)
   Never v(a | evidence) alone -- that swaps the fragment-mismatch
   problem for a coreference-mismatch problem (a later sentence's
   pronouns have no antecedent without the accumulated prefix).
   REQUIRES temperature scaling on a held-out set before v can safely
   drive search -- in PUCT, v isn't just a threshold, it directly
   drives which branches get exploited, so an uncalibrated,
   overconfident classifier doesn't just misjudge one candidate, it
   biases where the whole search commits its budget. Not yet built.

3. OOD pruning mask (OODDetector, Mahalanobis distance, already built):
       Ω(s) = 0     if OODDetector.distance(pooled(s)) <= threshold
       Ω(s) = -inf  if OODDetector.distance(pooled(s)) >  threshold

4. PUCT selection (which branch to expand next):
       U(s,a) = Q(s,a) + c * P(a|s) * sqrt(N(s)) / (1 + N(s,a)) + Ω(s⊕a)
   Q(s,a): exploitation (mean verifier score seen so far this branch)
   c*P*sqrt(N(s))/(1+N(s,a)): exploration (proposer-favored, under-
     visited branches), standard AlphaZero PUCT, correctly stated
   Ω: hard prune -- an OOD branch is never explored, full stop

5. Backup after a branch is scored (standard incremental mean):
       Q_new(s,a) = [N(s,a)*Q_old(s,a) + v_final] / [N(s,a)+1]
       N(s,a) <- N(s,a) + 1

6. Budget K: clamped by the REAL measured latency benchmark
   (scripts/benchmark_mcts_latency.py), not an unmeasured "IQ-quotient"
   number. CPU numbers already in hand (see above): proposer cost
   dominates verifier cost ~17x, so K should be tuned primarily against
   proposer-call cost, and batching sibling proposer calls is the
   highest-leverage latency lever once real GPU numbers are in.

Honest final claim -- the one actually worth writing down, replacing
every "mathematically guaranteed / impossible / structurally
impossible" phrasing this design went through before landing here:
  This structurally rules out continuous-space gibberish, because every
  node is a complete sentence built from the proposer's real
  vocabulary. It returns the path that empirically minimizes
  contradiction probability under a specific, fallible, calibrated
  classifier. That is a real, useful property. It is not a guarantee
  of truth, and no phrasing of this design should claim it is.
```

### Residual risk, not a blocker but worth monitoring once this runs

PUCT's exploitation term explicitly searches for whatever maximizes the
verifier's score — structurally adjacent to the RL-reward-hacking risk
rejected at the start of this whole design arc (training the proposer
against a fixed verifier). It is not the same risk: RL training gets
unbounded iterations to find and permanently bake an exploit into the
proposer's weights; MCTS gets a bounded, per-query search budget from a
fixed, frozen prior, and nothing found in one query's search persists
into the next. Real but bounded, not equivalent. **Concrete test once
live**: compare the flywheel's correction rate on MCTS-selected answers
vs. flat-voted answers. A materially higher rate on MCTS answers is the
number that tells you this risk is showing up in practice, not just in
theory.

### On OOD generalization and coverage vs. accuracy (accurate version)

Neural classifiers generalize poorly outside their training
distribution, and can be confidently wrong there rather than
uncertain. `OODDetector` is the existing, deliberate answer to this: an
MCTS branch whose accumulated path lands far from the classifier's
training distribution gets pruned (`Ω = -inf`) rather than scored,
regardless of how confident the classifier's raw output would have
been. The real trade-off this creates: on a genuinely novel-domain
question, most or all branches may get pruned as OOD, and the system
abstains rather than answers — trading coverage for not confidently
lying. `verifier_flywheel.py`'s exported corrections, folded into the
next `verifier_train.py` run (already built, already the mechanism —
see the training-time diagram above), is what pushes that boundary
outward over time as `OODDetector` gets refit on an expanded training
set. This bounds risk meaningfully; it does not make anything "100%
safe" — the deterministic Stage 1 floor and the classifier itself,
even on in-distribution input, both retain their own known, nonzero
error rates. "Meaningfully safer, honestly bounded" is the accurate
claim; "100% safe" is the same overclaim pattern this whole design
cycle kept producing and kept getting corrected out of.

### What's already done vs. still open

- [x] Scoring rule pinned down (full accumulated path, never the
      isolated increment) — encoded directly in
      `scripts/benchmark_mcts_latency.py`'s `score_path()`
- [x] Latency benchmark harness built, CPU-validated (real Phase 2
      checkpoint, real production-shape verifier, `nvidia-smi`-checked
      before running anything so it couldn't contend with proposer
      training)
- [ ] Real GPU latency numbers — blocked on Phase 3/4 finishing;
      re-run `scripts/benchmark_mcts_latency.py --device cuda` the
      moment the GPU is free
- [ ] Temperature-scaling calibration on the entailment classifier —
      hard precondition for `v` to safely drive PUCT search, not yet
      built (this is Item 4's calibration work, needed here too)
- [ ] The PUCT loop itself — not yet written; everything above is the
      design this item exists to preserve until it's time to write it
- [ ] Batched sibling proposer calls (or other latency reduction, if
      the real GPU number requires it) — evaluate once real numbers
      are in, not before

## 6. Documentation — real updates once the model actually changes

**Split into two genuinely different things, only one of which needs to
wait**: performance claims (parameter count, benchmark numbers) depend on
validated results and correctly stay blocked; architectural description
(what components exist, how they interact) is just a factual account of
what's built and doesn't depend on benchmark results at all — done now,
GPU-independent, while the verifier retrains.

- [x] **`docs/architecture.md` rewritten** to actually describe the
      current system, not v0.3.0's. Added: the full verification cascade
      (word-overlap floor → entailment classifier + OOD gate → numeric
      plausibility → relational transitivity, explicit about the
      additive-only property), dynamic-N self-consistency voting +
      `TaskConfigCache`, proprioception (explicitly marked experimental
      and NOT a verifier substitute), and `SwarmSynthesizer`'s real
      decomposition behavior. **Found and fixed a real, separate
      problem while doing this, not just staleness**: the old doc
      described a "`ReasoningEngine`" producing "mathematically verified
      steps" — verified directly against the codebase and confirmed
      `ReasoningEngine` doesn't exist anywhere at all, not renamed, not
      moved, just never real. Replaced with an accurate description of
      what actually handles multi-step reasoning (`SwarmSynthesizer`
      decomposition + FLUX's CoT-trained `think=True` path). Verified
      `SwarmSynthesizer`'s decomposition-gating claim directly against
      `swarm.py`/`iq_router.py` before writing it, not assumed.
      Deliberately left the "~116M" parameter mentions untouched — that's
      the performance-claim half, still correctly blocked on validated
      benchmarks.
- [ ] README/docs currently describe FLUX as "~116M-class" throughout —
      that's v0.3.0. Update once the new (64M, pruned-vocab) model is
      validated and promoted, not before
- [ ] Re-check the "How It Connects" diagram still matches reality

## 7. Release readiness — one full pass, not the individual pieces checked ad hoc

- [ ] Run the full `.agents/skills/release_readiness/SKILL.md` checklist
      end-to-end against the finished, promoted model
- [ ] Item 16 bonus objectives remain explicitly non-blocking — ship
      whatever fraction is done, don't gate on the rest

## 8. Release commit

- [ ] Prepare the release commit
- [ ] **Do not push or merge to main without explicit confirmation** —
      same standing rule as every other hard-to-reverse action this session
