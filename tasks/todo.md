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
Phase 4  qat_best.pt   [blocked: --data-bin needs a pruned-vocab re-tokenization
                        first — see "Where things actually stand" below]
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
- Phase 2 (SFT) — **done**. 743 steps, 787.6 min, val loss 3.42, PPL 30.5
  (down from Phase 1's PPL 61.1 — a real improvement, not just more training).
- Phase 3 (CoT) — proof run **succeeded**: all 4 sources loaded exactly
  fairly (100 each), pruned-vocab tokenizer confirmed working, val loss
  4.26/PPL 70.8 (a different, harder task shape than SFT, not a red flag).
  **Full run now running** (default 10,000 examples, 4 epochs).
- Phase 4 (QAT) — not started, and has a **real, newly-discovered blocker**
  (see below) that needs resolving before it can launch.
- Item 17 (verifier upgrade) — code, data pipeline, and tests **fully
  built and CPU-verified**; training **not yet run** (needs the GPU, which
  Phase 3–4 has first).
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

**Real blocker found for Phase 4, not yet resolved**: `qat_train.py`
requires `--data-bin`, a pre-tokenized memmap produced by
`scripts/pretokenize.py` — which *also* has no `--pruned-vocab` support,
and the `train.bin`/`val.bin` that already exist on disk (from July 2nd)
were tokenized with the full vocab. Using them as-is for Phase 4's
general-text-recovery data would hit the same out-of-range issue even with
`qat_train.py`'s own fix in place. Needs either pruned-vocab support added
to `pretokenize.py` and a fresh `.bin` re-tokenization, or another source
of general-text-recovery data consistent with the pruned vocab. Not
blocking Phase 3 — only needs solving once Phase 3 is done and Phase 4 is
actually next.

## 1. Let training finish — no action, just don't interrupt it

- [x] Phase 2 (SFT) completes
- [ ] Phase 3 (CoT distillation) — now includes CommitPackFT as a 4th
      fair-budgeted source alongside GSM8K/OpenOrca/Magicoder — proof run
      passed cleanly, full run in progress
- [ ] Resolve Phase 4's `--data-bin` pruned-vocab blocker (see above) before
      attempting to launch it
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
