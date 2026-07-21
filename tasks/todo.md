# Uchi 0.5.0 — Execution Order (locked)

Supersedes the previous version of this file (0.4.0's, which tracked that
release's phased retraining and verifier work — all done, commit `838bf0e`
onward). Full detail for every item lives in
`tasks/0.5.0 Itemized Deliverables.md` — this file is the compressed,
sequenced checklist, not a duplicate of it.

**Target:** ≥85% SWE-bench, without regressing any of 0.4.0's verified
capabilities (ODUSP recall, cascade precision, retrieval accuracy). Honest
calibration from the itemized doc stands: this exceeds current publicly
reported SWE-bench Verified numbers from frontier LLM-based agentic
systems, pursued here with a from-scratch, no-LLM, ~64M-parameter model.
Report the real number in Item 9 regardless of where it lands.

## Where things stand right now

Not started. This is the plan going in, not a status report — update each
section's checkboxes as work actually lands, same discipline as every
prior release's todo.md.

## Dependency map (who actually blocks whom)

```
Item 1 (corpus + decontamination) ──► Item 2 (compaction + context extension)
                                            │
Item 3 (physics/world retrieval)  ─┐        │
Item 3.5 (repo-scale code RAG)     ─┤──► Item 4 (retrieval scaling validation)
                                    │        │
Item 5 (execution sandbox) ────────┴────────┼──► Item 6 (GRPO self-play) ──┐
        │                                    │        │                    │
        └──────────────────────────────► Item 7 (live verify + repair)     │
                                             (parallel to Item 6)           │
                                                        │                   │
                                             Item 8 (j-space MCTS) ◄────────┘
                                                        │
                                             Item 9 (full re-validation)
```

Only three things are true dependencies from the start: Item 2 needs
Item 1's decontaminated corpus; Item 4 needs both retrieval tracks built;
Item 9 needs everything else finished. Items 1, 3, 3.5, and 5 have no
upstream dependency at all — they start in parallel, today.

## 1. Corpus + Decontamination (Item 1) — starts immediately, no dependency

- [x] Source (a)/(b)/(c) general-code portion — **The Stack v2 (BigCode),
      Python subset**, decided and documented in `uchi/corpus_sources.py`'s
      module docstring (rejected CodeSearchNet as too thin, rejected
      reusing 0.4.0's Magicoder/CodeAlpaca/CommitPackFT as SFT-scale, not
      pretraining-scale). `iter_general_code_python()` streams it. HF token
      now authenticated + gate terms accepted — **verified end-to-end
      against real data**, correcting two wrong pre-auth assumptions:
      v2-dedup is a **Software Heritage index, not a content dataset**
      (rows carry `blob_id`/`src_encoding`, no `content` field — content is
      fetched separately via SWH's public S3 mirror, anonymous/unsigned,
      `_fetch_swh_content`/`_swh_s3_client`, `boto3` added to `dev` extras)
      and it is **not** pre-filtered to permissive licenses (both
      `"permissive"`/`"no_license"` rows present live — filtered
      client-side now, alongside `is_generated`/`is_vendor` exclusion).
      Live smoke test: `python -m uchi.corpus_sources --source stack
      --limit 3` pulled 3 real files (MutopiaProject/mupub,
      uktrade/digital-workspace-v2, thomaspurchas/readthedocs.org) through
      the decontamination gate end-to-end. `bigcode/the-stack-dedup` (v1)
      remains the documented fallback but its own gate is still
      pending/denied for this token — its column names are still a
      best-recollection candidate list, not live-verified (see
      `uchi/corpus_sources.py` docstring). Tests:
      `tests/test_corpus_sources.py` (12 offline incl. real gzip/decode
      against a fake SWH client, 2 live `eval` incl. a real SWH S3 fetch,
      all passing).
- [x] New required source: real GitHub issue→diff pairs at scale —
      **SWE-Gym** (`SWE-Gym/SWE-Gym`), `iter_issue_diff_pairs()` in
      `uchi/corpus_sources.py`. Ungated, confirmed live: schema matches
      SWE-bench's own (`repo`/`problem_statement`/`patch`/`base_commit`/
      `FAIL_TO_PASS`/`PASS_TO_PASS`), sampled repos disjoint from
      SWE-bench's eval set. `SWE-Gym/SWE-Gym-Raw` (~66K instances)
      available via `dataset_id=` for later scale-up.
- [x] **Decontamination filter, hard gate before Item 2**: exclude every
      repo in SWE-bench's fixed eval set (astropy, django, flask,
      matplotlib, pytest, requests, scikit-learn, sympy, + full
      full/verified/lite list) at the **repo level**, not just
      instance-level dedup. `uchi/corpus_decontamination.py`, resolved
      live from the HF Hub (never a hardcoded snapshot). Tests in
      `tests/test_corpus_decontamination.py` (7 offline + 2 live `eval`,
      all passing).
- [x] Delegate: standalone decontamination script, reusable by any future
      corpus addition, not a one-off inline filter — both
      `corpus_decontamination.py` and `corpus_sources.py` are independent
      CLIs (`python -m uchi.corpus_decontamination`,
      `python -m uchi.corpus_sources --source {stack,swe-gym,swe-gym-raw}`)
      that compose: source streams a `{"repo": ...}`-shaped record,
      the decontamination gate filters it, downstream tokenization
      (`scripts/pretokenize.py`-style) consumes the filtered JSONL.
      **Pulled to disk (2026-07-09)**: `.uchi/corpus/swe_gym_full.jsonl`
      (all 2,438 curated SWE-Gym instances, 0 excluded) +
      `.uchi/corpus/stack_v2_sample.jsonl` (14,991/15,000 files kept, 9
      excluded by decontamination, **13,959 unique repos** — broad, not
      concentrated — ~18.8M tokens, 77MB, real ~27min wall-clock).
      User's explicit scope call: "a good representative sample, not the
      full dataset" — this is that, not literally the full Stack v2/all
      of SWE-Gym-Raw. Sufficient to start Item 2; not "pull more" unless a
      future need specifically calls for it.

## 2. Model Compaction + Context Extension (Item 2) — depends on 1

- [ ] Train from scratch on Item 1's decontaminated corpus — **corpus
      tokenized, training about to launch at a reduced scope**.
      `scripts/pretokenize_0_5_0.py` mixed Item 1's local pull (~23.3M
      tokens: 14,991 Stack v2 files + 2,438 SWE-Gym issue/diff pairs, both
      already decontaminated) with FineWeb-Edu up to an 80M-token budget
      (matching 0.4.0's Phase 1 scale), interleaved not block-concatenated.
      Refactored `scripts/pretokenize.py`'s `build_bin` to accept a
      pluggable `text_stream` so this didn't duplicate the tokenize/write
      logic. **Run for real**: `uchi/flux/data_0_5_0/{train,val}.bin`,
      80M/1M tokens, both verified (correct token counts, valid uint32
      memmap).
      **Two real, unexpected findings from launch smoke-testing, both
      fixed in `train_v2.py`, not just discovered**:
      (1) OOM at `micro_batch=2, seq_len=1024` with the *full* 12-layer
      model — the seq_len benchmark's isolated single-layer measurements
      didn't capture whole-model memory pressure. Fixed by running with
      `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the OOM error's
      own suggestion) rather than dropping to `micro_batch=1` (confirmed
      working but ~40% slower).
      (2) **A pre-existing 10× throughput-logging bug**: `tok_per_sec`
      always multiplied token count by `log_interval` (10), but the very
      first log line (step 0) only spans 1 real step, not 10 — so every
      run's first printed rate was ~10× inflated (a "7,655 tok/s" step 0
      reading was actually ~744 tok/s once measured honestly). Fixed by
      tracking `steps_since_log` explicitly instead of assuming
      `log_interval` always applies. This changed the real plan: original
      scope (matching 0.4.0's ~491M total training tokens) would have
      taken **~7.6 days** at the corrected ~750 tok/s, not the
      hours-to-a-day originally estimated. Also found: **seq_len barely
      affects this** (805 tok/s at 512 vs 744 at 1024, not the
      near-2× you'd expect) — something closer to fixed per-step/
      per-layer overhead (likely the sequential Python-loop scan's
      kernel-launch cost under gradient checkpointing) dominates over
      seq_len-scaling cost in this range, consistent with the seq_len
      benchmark's own finding that the SSM scan (not attention) is what
      dominates. User's call given this: keep seq_len=1024 (the context-
      extension goal), scope down to **~20M tokens (~305 steps, ~7-8hrs)**
      instead of a full 80M-token epoch (~30hrs) or the original ~7.6-day
      plan. Added `--eval-interval`/`--checkpoint-interval` CLI overrides
      to `train_v2.py` (previously hardcoded at 500/5000 — silently never
      firing on any run shorter than that, meaning no `ckpt_best.pt` and
      no mid-run validation signal) so this shorter run actually produces
      periodic checkpoints and a monitorable val-loss curve.
      **COMPLETE** (launched 2026-07-09 15:01, finished 22:25, ~7h24m):
      305/305 steps, clean run, no crashes/NaN. Val PPL 1433.8 (step 50)
      → **490.0** (step 300, best). Checkpoints in
      `uchi/flux/checkpoints/v050_phase1/` (`ckpt_best.pt`/`ckpt_final.pt`
      /`ckpt_latest.pt`) — isolated from production `flux_best.pt`, which
      is untouched. Log: `.uchi/corpus/train_0_5_0_phase1.log`. User
      explicitly said skip the non-regression-checkpoint bullet below and
      continue the training pipeline instead — see Phase 2 note under
      "Addendum" at the bottom of this file.
      **Real bug found in the completion summary, not yet fixed**:
      printed "305 steps in 0.1h" — obviously wrong (real time ~7.4h);
      cosmetic only, doesn't affect the checkpoints or loss curve, low
      priority.
- [x] Micro-benchmark attention FLOPs/memory at 512/1024/2048 tokens vs.
      current 256 — `scripts/benchmark_seq_len.py`, real hardware (RTX
      5070), the actual `AttentionBlock`/`TSSMBlock` classes at the real
      checkpoint's shape (d_model=768, 12 layers, 3 attention/9 SSM),
      under the same `torch.utils.checkpoint` gradient-checkpointing path
      training uses — not a theoretical FLOPs formula. **Result: attention
      never gets close to dominant** at any tested seq_len (1.6% share at
      256, 2.1% at 1024) — the checkpointed SSM scan's per-layer cost
      dwarfs attention's throughout the range that fits in 12GB VRAM at
      batch=4 (2048 OOMs outright). Recommendation: **1024** is the
      largest candidate that both fits in memory and keeps attention
      subdominant — 2048 needs a smaller batch or grad accumulation to
      even test, separate from the attn-vs-ssm question.
      **Real, unresolved finding surfaced by this benchmark, worth
      flagging before Item 2's actual training run**: `ssm.py`'s
      documented `UCHI_FUSE_SSM_SCAN=1` (torch.compile-fused scan, profiled
      3.08× *forward-only* speedup in that module's own docstring)
      measured **slower**, not faster, once wrapped in the same gradient
      checkpointing training actually uses (88.92ms vs 63.71ms unfused at
      256; 1055ms vs 294ms at 1024) — checkpointing recomputes the forward
      during backward, and that recompute path likely isn't hitting
      torch.compile's cached graph, paying a large compile-scale cost on
      every step instead of once. Not investigated further here (out of
      this bullet's "arithmetic, not a research phase" scope) — training
      should default `UCHI_FUSE_SSM_SCAN` **off** until this is understood,
      contrary to what the flag's own docstring would suggest.
- [x] Re-initialize HiPPO A_fast/A_slow timescales proportional to the new
      max_seq_len (current 2–10 / 10–34 horizons are tuned for 256 tokens,
      do not just reuse them) — `SSMCell`/`TSSMBlock`/`HybridTSSM` gained
      optional `a_fast_range`/`a_slow_range` constructor params (default
      unchanged, `(0.0,2.0)`/`(2.0,3.5)`, so every existing caller and
      loading an old checkpoint is unaffected). `train_v2.py` computes
      them automatically from `--seq-len` (`scale = seq_len/256`,
      `fast=(0, 2*scale)`, `slow=(2*scale, 3.5*scale)`) — keeps the SSM
      state horizon covering the same ~13% fraction of context regardless
      of seq_len. At seq_len=1024: fast≈(0,8), slow≈(8,14). Smoke-tested:
      real forward pass with custom ranges, and default-unchanged
      construction both verified working.
- [ ] **Non-regression checkpoint**: MMLU/ARC/retrieval-accuracy suite
      against this model before it becomes the base for Items 6 and 8.

## 3. Physics & World-Knowledge Retrieval (Item 3) — parallel, no dependency

- [x] Unchanged from prior releases' retrieval work — confirmed, not just
      assumed: `uchi/retrieval.py`'s `SemanticIndex` is still the live
      index `uchi/generate_and_ground.py`'s `GenerateAndGround` retrieves
      evidence from (no drift since 0.4.0). Live-benchmarked for real in
      Item 4 below, not just re-confirmed by reading the import.

## 3.5. Repo-Scale Code Retrieval (Item 3.5) — parallel, no dependency

- [x] Delegate: `uchi/code_retrieval.py`'s `CodeIndex` — **not a literal
      FAISS/sentence-transformer/cross-encoder port**, deliberately, per
      the project's no-pretrained-model constraint: from-scratch
      per-repo-trained skip-gram (SGNS, hand-written NumPy) in place of a
      sentence-transformer, plain NumPy cosine in place of `hnswlib`/FAISS
      (a per-task repo index is thousands of chunks, not millions — brute
      force is microseconds), a deterministic lexical-overlap +
      import-graph rerank in place of a cross-encoder. Reasoning fully
      documented in the module docstring. Tests in
      `tests/test_code_retrieval.py` (13 passing, real on-disk fixture
      package, not mocks).
- [x] Per-repo index (file/function chunks) built at the start of each
      SWE-bench task (`CodeIndex.build`/`index_repo`, `ast`-based chunking:
      module/class/function/method granularity); import-graph edges
      (`import x`/`from x import y` resolved to in-repo files) track
      cross-file navigation via `neighbor_chunks`/`imports_of`/
      `imported_by`. Symbol-level call-graph tracking is deferred past v1,
      documented as such — import-graph edges already answer "the bug is
      in file A, the fix is in file B it imports."
- [x] Confirm this index is exempt from Item 1's decontamination list —
      it's the actual task repo at eval time, not training data. Stated
      explicitly in both `code_retrieval.py`'s and
      `corpus_decontamination.py`'s module docstrings, cross-referencing
      each other.

## 4. Retrieval Scaling Validation (Item 4) — depends on 3, 3.5

- [x] Benchmark Item 3's flat fact retrieval and Item 3.5's code retrieval
      **separately** — `benchmarks/retrieval_scaling_benchmark.py`, two
      independent tracks, both run against real data (not synthetic
      queries). **Item 3** (real SQuAD 2.0, 100 answerable questions,
      `SemanticIndex` — pre-trained embeddings, only `retrieve()` has a
      per-query cost): 465 passages indexed in 0.02s, **86% hit-rate@5**,
      0.23ms/query. **Item 3.5** (real `SWE-bench_Lite` instances,
      `CodeIndex` — skip-gram trained fresh per repo at index-build time):
      3 real repos (django×2, astropy), **avg 117s to build one repo's
      index** — confirms the cost-profile warning in this item's own
      wording: that number says nothing Item 3's 0.02s does, the two
      systems are not comparable. Hit-rate@10 (does retrieval surface a
      chunk from the file the *gold* patch actually touches): 1/3 — too
      small a sample for a real percentage, but the harness and the
      "would this retrieval have helped `agentic_repair.py`" question it
      answers are both real and working end-to-end (django__django-15814:
      miss, django__django-13401: miss, astropy__astropy-14995: HIT).
      Scaling the sample further is a pure compute-time tradeoff (~2
      min/repo) — deliberately left small here rather than run
      unsupervised for an hour+.

## 5. Execution Sandbox (Item 5) — starts immediately, no dependency

- [x] Port `rl_env.py`'s `ExecutionSandbox`, extended to apply a patch to
      a sandboxed repo copy and run its real test suite (pass/fail +
      partial credit) — `uchi/execution_sandbox.py`, graded against
      SWE-bench's own FAIL_TO_PASS/PASS_TO_PASS convention. Tests in
      `tests/test_execution_sandbox.py` (4 passing).
- [x] Stage 1 (unblocks Item 6 now): subprocess + timeout + explicit
      test-id list (FAIL_TO_PASS/PASS_TO_PASS already answers "which
      tests matter" for eval; per-test pytest invocation, no batched-XML
      nodeid-mapping ambiguity).
- [x] Stage 2 (before self-play scales to full-repo suites, not before
      Item 6 starts): container/VM-level isolation — `uchi/sandbox_isolation.py`,
      using `bwrap` (bubblewrap), the only sandboxing primitive actually
      present in this environment (checked directly: no `docker`,
      `runsc`/gVisor, `firejail`, or `nsjail` on this host). Unshares
      network + PID namespace, restricts the filesystem to `/usr` (merged-
      `/usr` bootstrap: `/lib`, `/lib64`, `/bin`, `/sbin` symlinked back
      in) plus the running venv (read-only) and *cwd* (read-write),
      `--die-with-parent` for cleanup. Graceful fallback to plain
      `subprocess.run` if `bwrap` isn't installed — never a hard new
      dependency. Wired into `ExecutionSandbox` via a new `isolate: bool`
      constructor flag (`run_tests`/`apply_patch` call
      `sandbox_isolation.run_isolated` when `True`) — **defaults to
      `False`**, so this is additive, not a behavior change for any
      existing caller; opt in once self-play actually runs at volume.
      **Two real bugs found empirically, not assumed away**: (1) `bwrap`
      resolves bind-source paths against its own process, not the
      caller's — a relative *cwd* (this repo's `DEFAULT_ROOT`-derived
      paths are relative by construction) silently produced a nonsensical
      nested path; fixed by resolving *cwd* to absolute before building
      the bind arguments. (2) mount order matters: `--tmpfs /tmp` before
      the `--bind <cwd> <cwd>` clobbers *cwd* whenever it happens to live
      under the host's `/tmp` (true of pytest's own `tmp_path` fixture, and
      plenty of real callers) — fixed by moving the tmpfs mount before the
      more specific bind, so the specific bind shadows the broad one
      instead of the reverse. Validated for real, not just unit-tested: a
      passing test passes, a failing test's real assertion text and exit
      code survive unchanged, and a live TCP connect attempt to `8.8.8.8:53`
      confirms network really is blocked. Tests: `tests/test_sandbox_isolation.py`
      (7 passing, skip gracefully if `bwrap` absent) + one added
      `tests/test_execution_sandbox.py` case (`isolate=True` resolves the
      existing fixture identically to the non-isolated path).
- [x] Stage 2: test-impact-analysis subsetting for the broader training
      corpus, where no FAIL_TO_PASS/PASS_TO_PASS annotation exists —
      static, not coverage-based (a coverage-instrumented approach would
      need a full instrumented suite run per repo first, out of this
      pass's scope). `code_retrieval.py` gains `extract_patch_files()`
      (same regex convention `retrieval_scaling_benchmark.py` already
      uses) and `CodeIndex.impacted_tests()`: BFS over the existing
      *single-hop* `_imported_by` graph (confirmed it doesn't do
      transitive closure on its own) to the transitive closure of files
      depending on a changed file, filtered to test-shaped paths, plus a
      same-basename fallback (`foo.py` changed → also considers
      `test_foo.py`/`tests/test_foo.py`) for fixture-driven tests the
      static import graph can't see. `grpo.py` gains `derive_test_lists()`:
      builds a `CodeIndex`, computes impacted tests, runs them **before**
      the patch for a real baseline pass/fail split (failing-at-baseline →
      FAIL_TO_PASS candidates, passing-at-baseline → PASS_TO_PASS
      candidates) — same semantics real SWE-bench annotation encodes,
      derived instead of supplied. Wired into `GRPOTrainer._sample_branch`
      as a fallback **only** when a record's `fail_to_pass`/`pass_to_pass`
      are both empty — existing annotated curriculum records (all of
      curated SWE-Gym today) are unaffected. **Real granularity caveat,
      not a bug**: the derived lists are file-shaped test ids
      (`"test_foo.py"`), coarser than SWE-bench's function-level
      annotation (`"test_foo.py::test_case"`) — `ExecutionSandbox.run_tests`
      grades a whole file as a node id exactly the same way it grades one
      function, so this is honestly coarser, not broken. Tests:
      `tests/test_code_retrieval.py` (+5), `tests/test_grpo.py` (+3,
      including a full `GRPOTrainer.train_step` run with empty annotation
      lists confirming real, non-zero grading via the fallback).

## 6. Execution-Verified Code Self-Play (Item 6) — depends on 2, 5

- [x] GRPO (`grpo.py`) using Item 5's sandbox output
      (`SandboxEvalResult.reward`) as reward — pure-math functions
      (`grpo_advantage`/`grpo_loss`/`grpo_agentic_advantage`/
      `AgenticBaseline`) ported near-verbatim from
      `efficient_llm_training/src/grpo.py`; `GRPOTrainer` samples
      `n_branches` real ReAct episodes per instance (reusing
      `agentic_repair.py`'s loop, extracted into module-level
      `run_react_episode` so both callers share one parser), grades each
      via the real sandbox, computes each branch's `sequence_log_prob`
      (masked-loss pattern reused from `sft_train.py`), backprops
      `grpo_loss`. Tests: `tests/test_grpo.py` (10 passing) — real
      throwaway git repo + real `ExecutionSandbox`, scripted fake
      `generate_fn`, verifying a full multi-branch step produces
      non-degenerate advantage and an actual gradient update on a stub
      policy. **Real bug found and fixed along the way**: the
      reconstructed transcript abbreviates `apply_patch`'s diff to a
      `<N-char diff>` placeholder (by design, for prompt-context size) —
      scoring `sequence_log_prob` against that summary instead of the raw
      generated text silently produced identical log-probs for a correct
      and an incorrect patch of the same length. Fixed: `GRPOTrainer`
      scores the raw per-turn `generate_fn` outputs, not the parsed
      transcript. **Not done**: an actual curriculum training *run*
      meant to improve the model — needs the Phase 1→4 chain to clear the
      decision gate first (a zero-reward proposer gives zero-variance
      advantage, no gradient); smoke-tested end-to-end against the live
      in-progress `v050_phase1_v2` checkpoint instead (executes cleanly,
      numbers not meaningful yet).
- [x] **Curriculum**: `load_curriculum()` buckets `.uchi/corpus/swe_gym_full.jsonl`
      records by patch line-count (small/medium/large, thresholds
      documented in the module docstring as a first-pass split, not a
      distribution fit) and `run_curriculum()` iterates easy → hard,
      resolving each record's `repo`/`base_commit` to a local clone via
      `repo_fetch.ensure_local_clone` (same mechanism
      `retrieval_scaling_benchmark.py`'s code track already uses).
- [x] Trajectory schema: `TrajectoryStep` dataclass (`kind`: thought/
      action/observation/final) + `parse_transcript()`, converting
      `agentic_repair.py`'s existing prefixed transcript lines into
      structured form — reused directly by Item 8's action space.
- [ ] **Non-regression checkpoint (diagnostic, not a gate)**: re-run fast
      MMLU/ARC/HumanEval subset; separately record whether held-out
      pass-rate curve is actually rising. Item 8 proceeds regardless of
      the result — this just tells you what Item 8 is working with. **Not
      built this pass** — deferred, no `--eval-subset` hook wired into
      `GRPOTrainer` yet; `benchmarks/arc_benchmark.py`/`mmlu_benchmark.py`
      exist and are the intended reuse, just not called from here yet.
- [x] **Real curriculum-training launcher — `uchi/grpo_train.py`** (2026-07-18,
      user's explicit direction to promote Phase 4 v2 regardless of the gate
      and move straight into Items 6/8/9). The gap found: `GRPOTrainer`/
      `run_curriculum` had no checkpointing/logging/trajectory-persistence
      hook — this script adds all three (own loop re-walking
      `load_curriculum`'s buckets, `--checkpoint-interval` model saves,
      per-branch trajectory JSONL for Item 8). Never greedy (`temperature`
      default 0.8) — greedy decoding would make GRPO's zero-variance-
      advantage degenerate case worse, not better, per the module's own
      docstring warning. Never `torch.compile`s (sidesteps the exact
      autotune-stall class that hit `cot_distill.py` this session; real
      ReAct generation loops are a bad fit for compile anyway).
      **Real correctness issue found and fixed as a prerequisite**:
      `build_generate_fn` always loads a *second*, independent, never-
      updated model copy — using it here would train one model while
      generating episodes from a frozen other one. Fixed by factoring its
      generation closure out into `build_generate_fn_from_model`
      (`uchi/flux/inference_engine.py`), which `grpo_train.py` binds
      directly to the model its optimizer updates; `build_generate_fn`
      itself is unchanged (calls the new factored function internally),
      confirmed via the existing inference/proposer/agentic_repair test
      suite (14 passed) with no regression.
      **Live-verified, not just unit-tested**: real 1-instance/2-branch dry
      run (CPU, `v050_phase3_v2/cot_best.pt`) completed end-to-end in
      78s/instance — real repo clone, real ReAct episodes, real sandbox
      grading (0/2 resolved, expected — this checkpoint was never trained
      for multi-file repair, same honest gap already documented elsewhere),
      real checkpoint + trajectory-JSONL files written. Projected full
      2,438-instance curriculum at this (CPU, unrepresentative) rate: 53.6h
      — a real GPU calibration run is still needed before sizing the actual
      training run, per this project's standing practice (Phase 1's token-
      budget calibration, the corpus-fetch parallelization benchmarking).
      Tests: `tests/test_grpo_train.py` (4 passing, curriculum-iteration
      cap + trajectory-JSONL round-trip).

## 7. Live Execution-Gated Verification + Agentic Repair (Item 7) — depends on 5, parallel to 6

- [x] Wire Item 5's sandbox into the oracle as a new cascade layer —
      `SandboxVerificationChecker` (`uchi/execution_sandbox.py`) +
      `FactCheckOracle.is_patch_verified()` (`uchi/oracle.py`). Own
      method rather than folded into `is_grounded()`'s claim+evidence
      signature — a patch + test-id lists isn't that shape. **Fails
      closed** on sandbox exceptions/timeouts, unlike the entailment/
      relational layers' fail-open — the asymmetry is deliberate, see
      the class docstring. Tests in `tests/test_sandbox_verification.py`
      (5 passing, including the fail-closed behavior explicitly).
- [x] Port `react_agent.py` + `agentic_harness.py`; register `run_tests`,
      `apply_patch`, `read_file`, `grep_repo` in `tools.py`'s pattern —
      `uchi/tools.py` (`RepoToolRegistry`, per-episode not global-singleton,
      since each SWE-bench instance gets its own checkout) +
      `uchi/agentic_repair.py` (`AgenticRepairAgent`). Fused the two source
      files into one module: same ReAct Thought/Action/Observation shape
      as `react_agent.py` (regex-parsed, kept close to verbatim), driven by
      `uchi/flux/inference_engine.py`'s real `build_generate_fn()` seam
      instead of the foreign source's streaming `InferenceEngine` —
      `agentic_harness.py`'s raw-token-loop and `<|action|>`/`<|observation|>`
      special tokens weren't ported, since FLUX was never trained with
      those tokens; plain-text `Thought:`/`Action:`/`Final Answer:` markers
      work with the model as-is. **Real correctness fix along the way**:
      `apply_patch`'s argument is a full unified diff, which routinely
      contains `]` (list literals, regex classes) — the source's
      `tool[arg]` bracket parsing silently truncates on the first `]` in
      that case, so `apply_patch` gets its own `<<<PATCH>>>...<<<END_PATCH>>>`
      delimited block instead (documented to the model in the system
      prompt); the other three short-arg tools keep the bracket form.
      **Second real bug found and fixed, in `execution_sandbox.py`, not
      just this new code**: `run_tests` could serve stale pass/fail
      results after a same-checkout `apply_patch` — Python's timestamp-
      based `.pyc` cache can treat a just-edited source file's stale
      bytecode as valid when two writes land in the same mtime window
      (Stage 1's original design never hit this — one `evaluate()` call
      always applied-then-tested exactly once). Fixed: `run_tests` now
      clears `__pycache__` and sets `PYTHONDONTWRITEBYTECODE=1` before
      each invocation. Confirmed via 5 repeated runs pre/post-fix (flaky
      before, stable after). Tests: `tests/test_tools.py` (13),
      `tests/test_agentic_repair.py` (10) — real throwaway git repos +
      real `ExecutionSandbox`, scripted fake `generate_fn` standing in for
      FLUX (same testing shape as mocking react_agent.py's engine), all
      passing.
- [x] **Tri-state outcome policy**: pass / fail / timeout-or-flaky —
      `agentic_repair.Outcome` (`PASS`/`FAIL`/`ABSTAIN`) +
      `classify_outcome()`. `ABSTAIN` (timeout-or-flaky) does not retry and
      is not meant to fall back to word-overlap/entailment/numeric/
      relational — documented in the module docstring as the caller's
      contract; no router/oracle wiring changed, since nothing yet calls
      this loop from `ask()`'s cascade (out of this bullet's scope, same
      as Item 7's first bullet leaving `is_patch_verified()` unwired from
      `ask()` too).
- [x] **Retry cap: 3–5 attempts**, hard number, no unbounded iteration —
      `AgenticRepairAgent(max_attempts=4)` default, whole-episode retries
      (fresh checkout + fresh ReAct exploration each time, previous
      attempt's real failure text spliced back in as feedback), distinct
      from `max_iterations` (the ReAct loop's own per-episode step cap).
- [x] Port `parallel_thought.py`'s vectorized batch generation so
      dynamic-N doesn't cost N× latency — landed in `uchi/flux/inference_engine.py`
      (`build_batch_generate_fn`, batch-dimension prefill+decode_step over
      N branches in one forward pass, per-branch temperature spread same
      as the source's `torch.linspace` mechanism) + `uchi/proposer.py`
      (`FluxProposer.propose_batch`, opt-in via `load(with_batch=True)` —
      loads a second model instance, so it's not a silent default GPU-memory
      change) + `uchi/generate_and_ground.py`'s `_candidates()` (batches
      only the first-attempt round across all `n_votes` branches, since
      that's the one round every branch shares an identical prompt for;
      later per-branch reflection stays sequential, unchanged). Not
      `task_config_cache.py` itself, despite the todo's original phrasing —
      that module only *recommends* N, it was never where N candidates get
      generated; `_candidates()` is. **Real bug found and fixed as a side
      effect of building this**: `build_generate_fn` defaults to
      `greedy=True`, so N sequential `propose()` calls against the same
      prompt were producing N byte-identical strings — self-consistency
      voting had zero actual diversity to vote between. Per-branch
      temperature spread fixes that too, not a separate change.
      **Live-verified through the real production path** (not just the
      raw batched-generate function): `GenerateAndGround._candidates()`
      with `n_votes=4` — 8.92s batched vs. 18.59s with the sequential
      fallback forced, ~2× faster, and genuinely diverse candidates
      instead of repeats. Full suite (422 passed, 5 skipped) confirms no
      regression to the unbatched default path.

## 8. Latent-Space Planning — j-space (Item 8) — depends on 2, 5, 6, built unconditionally

- [x] Port `world_model.py` (DynamicsHead, ValueHead, WorldModel) near-
      verbatim from `efficient_llm_training/src/world_model.py` — the
      architecture and `predict_next`/`rollout`/`update_value`/
      `rerank_sequences` are model-agnostic, and `HybridTSSM` already
      exposes `.embedding`/`.layers`/`.norm_f` under the exact names the
      source assumes. One real adaptation: `train_dynamics` no longer
      scans a project-specific shard directory that doesn't exist here —
      takes real token sequences directly instead (e.g. from Item 6's
      collected trajectories), same underlying MSE-on-next-state math.
      `mcts.py`'s `MCTSNode`/`MCTS` PUCT mechanics (`ucb`, `_select`,
      `_backprop`) ported as-is — the search algorithm doesn't care what
      an "action" is. Tests: `tests/test_world_model.py` (8 passing),
      `tests/test_mcts.py` (7 passing).
- [x] Redefine "action" as hypothesis/reasoning-step level, using Item 6's
      Thought-tagged trajectory segments as the discrete units (mean-pooled
      embeddings, same approach as `proprioception.py`) — `mcts.py`'s
      `sample_candidates` samples `top_k` whole ReAct continuations from
      `generate_fn`, embeds each via `proprioception.pooled_hidden_for_question`
      (reused directly, not reimplemented — pooling free-form text is the
      same operation either way), and derives priors from
      `grpo.sequence_log_prob` softmax-normalized across the group, in
      place of the source's top-k vocab-logit priors. **Real property
      surfaced while testing, not a bug in the port**: PUCT's exploration
      term only grows as `sqrt(n)`, so under perfectly symmetric priors
      (an edge case that essentially never occurs with real LM
      probabilities) it takes ~65+ simulations to overcome the arbitrary
      tie-broken first pick and actually surface the higher-value branch
      — confirmed empirically in `tests/test_mcts.py`, not just asserted.
- [x] Hook ValueHead to Item 5's real pass/fail as reward, trained via
      GRPO same as Item 6 — `WorldModel.update_value`/`update_value_from_state`
      take a real reward (`SandboxEvalResult.reward`) directly, same
      reward source `grpo.py` uses, no separate reward design. **Not
      done**: an actual training run on real collected trajectories — same
      Phase 1→4 gate as Item 6; smoke-tested end-to-end against the live
      in-progress checkpoint instead (`MCTS.select_action` runs cleanly,
      search quality not meaningful since `ValueHead` is untrained).
- [x] **Operational "no precedent" check**: `mcts.NoPrecedentCheck` reuses
      `OODDetector`'s Mahalanobis-distance mechanics
      (`uchi/flux/verifier_model.py`) exactly as `proprioception.py`
      already does for question familiarity — not a new distance metric.
      Diagnostic only (`{"distance", "close"}`), no automated branching,
      per todo.md's explicit instruction. **Real bug found and fixed**:
      first draft copied `proprioception.py`'s "start inert at
      threshold=inf until calibrated" convention verbatim — wrong here,
      since that made the "far" case unreachable by construction (nothing
      is ever `> inf`). Fixed to use `OODDetector`'s own sensible default
      (3.0 Mahalanobis std-devs) so the check actually functions
      out of the box; caught by `tests/test_mcts.py`'s far-query test
      failing, not by inspection.
- [ ] **Non-regression checkpoint**: re-run fast MMLU/ARC/HumanEval subset
      again — second sequential fine-tune on top of Item 6's, forgetting
      compounds across passes if unchecked. **Not built this pass**, same
      status as Item 6's non-regression bullet above.
- [x] **Real training launcher — `uchi/world_model_train.py`** (2026-07-18,
      same session as Item 6's launcher above). Reads Item 6's trajectory
      JSONL, trains `WorldModel.train_dynamics`/`update_value_from_state`
      against real token sequences + real rewards (base FLUX model stays
      frozen throughout — only the two small heads train), saves to
      `WorldModel.CKPT_PATH`, then smoke-validates with a real
      `MCTS.select_action()` call against the trained world model + a live
      `generate_fn`. **Live-verified**: 2-trajectory CPU dry run — dynamics
      loss computed (2.95, 20 calibration steps), value loss 0.00477,
      checkpoint saved, MCTS smoke-check returned a real (if unpolished —
      expected given this checkpoint's known limitations) candidate action
      without crashing. Tests: `tests/test_world_model_train.py` (4
      passing, trajectory-load missing/empty/round-trip cases).

## 9. Full Benchmark Re-Validation (Item 9) — depends on 2, 4, 6, 7, 8

- [ ] Re-run MMLU, SWE-bench, ARC against the finished model.
- [ ] Establish first baselines on TruthfulQA and HumanEval. **TruthfulQA
      MC1 harness built** (2026-07-18): `benchmarks/truthfulqa_benchmark.py`,
      modeled on `humaneval_benchmark.py`'s structure. Uses the MC1
      multiple-choice formulation (judge-free, exact-match) rather than the
      original paper's generation+GPT-judge variant — an LLM-judge grader
      is a non-starter given this project's hard no-LLM constraint. Scores
      each choice via `uchi.grpo.sequence_log_prob` (reused, not
      reimplemented). Tests: `tests/test_truthfulqa_benchmark.py` (3
      passing, mechanics-only against a stub policy — an untrained/stub
      model can't be expected to prefer the correct choice, that real
      result is this harness's job to measure for real, not fake here).
      **Real bug found and fixed while wiring this up**: both this new
      harness's sibling scripts, `swebench_real_eval.py` and
      `humaneval_benchmark.py`, called `build_generate_fn(checkpoint=...)`
      with no way to pass a pruned-vocab path — its own auto-detection
      silently falls back to the OLD 0.4.0 `pruned_vocab_32k.json` whenever
      omitted, and accepts it without complaint because both 0.4.0's and
      0.5.0's pruned vocabs happen to be size 32,018. Would have silently
      corrupted any eval against a 0.5.0 checkpoint with the *wrong* token
      mapping, no error, no warning. Fixed: added `--pruned-vocab` to both
      scripts' CLIs, threaded through to `build_generate_fn`. Existing
      `tests/test_humaneval_benchmark.py` (7 passing) confirms no
      regression. HumanEval harness now built and proven**, ahead of this item's formal
      dependencies (same precedent as `swebench_real_eval.py`):
      `benchmarks/humaneval_benchmark.py`, modeled directly on
      `swebench_real_eval.py`'s structure. Loads `openai/openai_humaneval`
      (confirmed working id/schema — the old unnamespaced
      `openai_humaneval` id no longer resolves on HF). Prompt +
      code-extraction reuse `swebench_benchmark.py`'s established
      FLUX-facing instruction framing and `_extract_code_blocks`, not
      reinvented. Grading reuses `sandbox_isolation.run_isolated` (this
      item's own Stage 2 work above) — `candidate + test + check(entry_point)`
      as one real program, pass@1 = exit code 0; deliberately not
      `code_engine.REPLOracle`, whose `def run():` convention doesn't
      match HumanEval's shape. **Live-run, not just unit-tested**:
      `--sample 3` against the real, live in-progress `v050_phase1_v2`
      checkpoint (CPU, to avoid VRAM contention with that same training
      job — added a `--device` flag for exactly this) completed
      end-to-end in 48s, 0/3 pass@1 — honest, expected result given the
      checkpoint hasn't cleared its own decision gate yet, not a harness
      bug. Tests: `tests/test_humaneval_benchmark.py` (7 passing, no
      network/dataset download needed — direct grading-function tests
      against hand-written correct/incorrect/crashing/syntax-error
      candidates). TruthfulQA baseline still not started.
- [x] Report the real SWE-bench number plainly against the 85% target —
      **the harness to do this now exists and is proven end-to-end**,
      ahead of the items it formally depends on, because Items 5 and 7
      already provide everything it needs: `benchmarks/swebench_real_eval.py`
      replaces `swebench_benchmark.py`'s lexical-overlap proxy (that
      script's own docstring said real execution was "out of scope" —
      it no longer is) with `AgenticRepairAgent` attempts against real,
      on-demand-cloned repos (`uchi/repo_fetch.py`, targeted shallow
      fetch-by-commit-SHA, not a full clone — some of these repos carry
      gigabyte-scale histories irrelevant to grading one commit), scored
      by `ExecutionSandbox`'s genuine FAIL_TO_PASS/PASS_TO_PASS execution.
      **Live-run, not just unit-tested**: `--sample 1` against real
      `SWE-bench/SWE-bench_Lite` (django/django, real clone, real FLUX
      `flux_best.pt` checkpoint via `build_generate_fn()`) completed
      end-to-end in 4.5s — outcome `fail` (patch attempt didn't resolve
      the issue). **Honest, expected result, not a bug**: FLUX was never
      trained for multi-file agentic repair at any real scale — 0.4.0's
      CommitPackFT slice taught diff *reading*, not producing a
      ReAct-formatted trajectory ending in a resolving patch. The number
      that matters (a full sample against SWE-bench_Lite/Verified) still
      waits on Items 2/4/6/8 actually improving the model this harness
      grades — this bullet is "the measurement instrument works and reports
      real, verified numbers," not "the target is hit." Full suite:
      `tests/test_repo_fetch.py` (5 offline + 1 live `eval`, all passing).
- [ ] Confirm non-regression contract held end-to-end vs. 0.4.0 baseline
      (ODUSP recall, cascade precision, retrieval accuracy).

## Addendum, done alongside this file

- [x] `tasks/core_principle.md` — added clarification that Item 6's GRPO
      reward (real test execution) is not the previously-rejected
      "RL-tune the proposer against the verifier's reward" failure mode.

## ReAct-format warmup fine-tune (2026-07-19) — ahead of Item 6, real result

Item 6's real GPU calibration (25 instances, 100 branches) got zero reward on
every branch. Diagnostic (`generate_fn` against a real system prompt + real
issue, `think=True`) showed complete structureless gibberish — the model had
never been shown the `Thought:/Action:/Observation:` tagged format in any
training phase. Built `uchi/flux/react_warmup_train.py`: fine-tunes the
current (QAT'd) `flux_best.pt` on real, mechanically-constructed ReAct
teacher traces (real tool calls — `read_file`/`apply_patch`/`run_tests` —
against real repos with real gold-standard SWE-Gym patches, no LLM anywhere;
only the Thought/Final-Answer English narration is templated), mixed with
CoT-recovery steps.

**Two real bugs found and fixed while building this**:
1. `uchi/grpo.py`'s `sequence_log_prob` returned `torch.zeros(())` — a
   tensor disconnected from the model's graph — whenever a response left no
   room in the token budget. If every branch in a GRPOTrainer group hit this
   (a real, observed case against this undertrained proposer), `grpo_loss`'s
   `.backward()` crashed with "element 0 of tensors does not require grad
   and does not have a grad_fn". Fixed to return a differentiable zero
   (`0.0 * model.embedding.weight.sum()`). Tests:
   `tests/test_grpo.py::test_sequence_log_prob_empty_response_is_still_differentiable`,
   `test_grpo_loss_backward_survives_an_all_degenerate_group` (new, 15/15
   passing total).
2. `uchi/execution_sandbox.py`'s `checkout_repo` used
   `shutil.copytree(..., symlinks=False)`, which crashes the *entire* copy
   on a single broken symlink — hit for real at 300-instance construction
   scale (`python/mypy`'s `mypyc/lib-rt` tree). Fixed to `symlinks=True`
   (also the safer default regardless — never dereference an untrusted
   repo's symlinks). Benefits every caller: Item 6, Item 7, Item 9's
   SWE-bench eval, not just this new script.

**Training complete** (2026-07-19, 300 steps, ~2.9h + ~1.9h real construction
time for 300 instances ≈ 4.8h total): ReAct val PPL improved every eval —
23.7 → 13.4 → 9.9 → 8.5 → 8.0 → **7.8 (best, final)**. Checkpoints:
`uchi/flux/checkpoints/v050_react_warmup/{react_warmup_best.pt,
react_warmup_final.pt}`. Tests: `tests/test_react_warmup_train.py` (3
passing, real-throwaway-repo fixture).

**Real result of the actual test that matters — honest, not spun**:
re-ran the exact raw-generation diagnostic that found the original bug
against `react_warmup_best.pt`. Checked with the real production parser
(`agentic_repair._THOUGHT_PATTERN`/`_ACTION_PATTERN`/`_FINAL_PATTERN`), not
eyeballing: **1 of 3 samples matched `_ACTION_PATTERN` (0 matched
`_THOUGHT_PATTERN` or `_FINAL_PATTERN`)**, vs. 0 of 3 matching anything
before. This is a real, measurable but **partial** improvement — the model
now sometimes produces a recognizable `Action:` line (previously never),
but still fails to reliably produce the full `Thought:`→`Action:`→
`Observation:`→...→`Final Answer:` structure, and even the one matching
sample's action argument was malformed. Not the clean "format now reliably
appears" result hoped for.

**User chose to re-run the Item 6 calibration anyway (2026-07-19) — real
answer, not spun: zero change.** Same 25 instances, same shape as before,
against `v050_react_warmup/react_warmup_best.pt`: **0/100 branches got any
reward**, identical to the pre-warmup calibration. The warmup's partial
format improvement (occasional `Action:` line where there was none before)
did not translate into even one successful patch resolution or any reward
variance. GRPO still cannot get a gradient at this checkpoint's current
capability level — the format gap, while measurably narrowed, was not
closed enough to matter for actual task-solving. `--checkpoint-dir
uchi/flux/checkpoints/v050_item6_calibration_v2/`, trajectories in
`.uchi/corpus/item6_calibration_v2_trajectories.jsonl`. Real per-instance
timing confirmed again: 19.4-22.5s/instance (~19.8s avg), full-curriculum
projection ~13.4h.

Honest state after v1: one round of a 300-example/300-step ReAct warmup
fine-tune measurably moved the needle on raw format-matching but did not
unblock Item 6.

## ReAct warmup v2 (2026-07-20) — different teaching format, real fix, still zero reward

User asked for a genuinely different teaching format, not just more dose.
Re-reading `agentic_repair.run_react_episode` against what v1 actually
trained on found a real training/inference mismatch: `run_react_episode`
calls `generate_fn` ONCE PER TURN expecting one short Thought+Action
completion (real Observation appended by the harness afterward, never
generated). v1 trained on ONE long completion containing the WHOLE
multi-turn trace, loss applied to fabricated Observation text too —
teaching the model to keep generating past an action and to
predict/hallucinate tool output, backwards from the real task. This is
standard multi-turn tool-use SFT practice getting applied correctly for the
first time (Toolformer/Gorilla/FireAct-style agent fine-tuning all mask
loss to the agent's own turns only) — not a novel invention.

**Rewrote `uchi/flux/react_warmup_train.py`'s data construction**:
`build_react_example` now returns per-turn `(prompt, target)` pairs
mirroring `run_react_episode`'s own prompt/memory construction exactly
(including the abbreviated `apply_patch[<N-char diff>]` memory form for
later turns, matching the harness's own abbreviation byte-for-byte) — loss
masked to each turn's own short target only, real Observations are context,
never a generation target. Bonus: this multiplies real training data from
the same real instances (300 instances → ~1000+ real per-turn examples).
Tests: `tests/test_react_warmup_train.py` (4 passing, checks against the
*real* `agentic_repair` regexes directly, not an approximation).

**A second real bug found while recalibrating for this**:
`tokenizer_v2.py`'s `encode_text` defaults to `max_length=1024` and
silently truncates (keeping the front, dropping the tail) whenever a
caller doesn't override it — `load_react_examples` never had, so **v1's
training data was also silently corrupted by this**, on top of the masking
bug. Fixed by passing `max_length=max_seq_len` explicitly. Real per-turn
token lengths measured properly this time (min 393, median 1446, p90 2719,
max 3095 across 40 real examples) — `--seq-len 3072` real-OOM'd on this
12GB GPU even with gradient checkpointing; settled on 2048 (already
proven to fit, keeps 82% of real examples).

**Training complete** (2026-07-20, 300 steps, continuing from `flux_best.pt`
again — not v1's checkpoint, to avoid compounding a flawed prior
adaptation): ReAct val PPL 20.5 → 7.7 → 5.0 → 4.3 → 4.0 → **3.9 (best,
final)** — converged faster and to a much lower PPL than v1's final 7.8.
Checkpoints: `uchi/flux/checkpoints/v050_react_warmup_v2/`.

**The real tests, in order, not spun**:
1. Single-turn generation diagnostic (5 real instances, one short
   `generate_fn` call each, matching real per-turn usage): **4/5 produced
   a parseable `Action:` line** (vs. ~1/3 in v1's cruder whole-response
   check, ~0/3 pre-warmup) — real, substantial improvement. `Thought:`
   itself still never appeared verbatim (model consistently drops just
   that literal word while keeping the rest of the sentence — an
   unexplained, minor residual quirk; doesn't block `run_react_episode`,
   which acts on Action/Final-Answer matches independently of Thought).
2. **Real multi-turn `run_react_episode`** against 3 real instances: 1/3
   now produces a genuine multi-turn episode (2 real turns — a failed
   `read_file` with a hallucinated path, then a failed `run_tests` with a
   hallucinated test id, both real tool calls with real failure
   Observations, episode correctly continuing to a second turn). 2/3 still
   produce zero parseable turns.
3. **The decisive test — Item 6 calibration, same 25 instances, same
   shape as both prior rounds: 0/100 branches got any reward. Identical to
   v1, identical to the pre-warmup baseline.** `--checkpoint-dir
   uchi/flux/checkpoints/v050_item6_calibration_v3/`. Per-instance timing:
   ~11.5-16.6s (~11.7s avg, full-curriculum projection ~7.9h).

**Honest conclusion**: the training/inference-mismatch fix was real and
worked as intended — format-matching improved substantially and
measurably, exactly as the root-cause analysis predicted it should. But it
still was not enough to produce even one successful patch resolution or
any reward variance. The remaining gap looks like a deeper capability
issue, not a format issue: even when the model produces a syntactically
valid action, its *content* (file paths, test ids) is still
hallucinated/wrong. Two well-motivated, properly-executed rounds of
ReAct-format warmup have now both failed to unblock Item 6 — this may be
approaching the limits of what this scale/approach can do without a larger
change (more capacity, much more real training compute at every phase, or
a different strategy entirely). Next direction is an open decision for the
user, not resolved here.

## Item 9 real run (2026-07-20) — a real, unrelated production bug found and fixed

User asked to run Item 9 (full benchmark re-validation) for real. MMLU/ARC
(`benchmarks/{mmlu,arc}_benchmark.py`, both boot the full production `Uchi()`
router — the ODUSP/Generate-and-Ground system, not FLUX directly) both
returned **0% accuracy, 100% no-parse rate**, and — the real red flag —
**the identical answer text for every question regardless of topic**
(oak trees, water phase changes, photosynthesis all got the same fixed
sentence in one run; capital-of-France and photosynthesis got a different
shared fixed sentence in another). Direct probing confirmed this wasn't
model incompetence alone — it was a real, previously-unknown bug in
`uchi/generate_and_ground.py`'s `GenerateAndGround.answer()`.

**Root cause, fully traced**: `_candidates()` always yields
`self._extractive(question, evidence)` (the retrieved passage that best
lexically matches the question, a real and deliberate helper) as an
unconditional final candidate. Since it *is* a piece of real evidence, it
trivially passes `FactCheckOracle.is_grounded()` (support against itself is
1.0) regardless of whether it actually answers the question. Confirmed
live: "What is the capital of France?" confidently returned a passage about
**New France** (the historical Quebec colony) — wrong entity, but "grounded"
because both share the literal word "france". Since FLUX's own generated
candidates are essentially always gibberish for prompts this complex (the
`think=True` path deliberately drops evidence from the prompt --
`FluxProposer.propose()` -- CoT was trained on the bare question, wrapping
it broke that format when tried previously), the extractive fallback wins
by default on essentially every non-trivial question -- a real trustworthiness
violation of the package's own stated principle ("Uchi never confabulates:
when it cannot ground an answer it says so").

**Fixed** in `generate_and_ground.py`'s `answer()`: track, by *position* (not
string equality -- a real proposer answer can legitimately be textually
identical to evidence for a simple direct lookup, and an early version of
this fix broke exactly that case, caught by the full test suite), whether
any candidate *other than* the guaranteed-final extractive one ever passed
grounding. If the extractive fallback is the *only* thing that grounds, that
means no real synthesis was ever verified -- treat it as unresolved and
abstain, instead of confidently emitting a possibly-wrong passage. Pure
extractive-only configurations (no proposer/decoder at all, a fully
legitimate documented mode) are unaffected -- the check only activates when
a real generator had a chance to produce something and everything it made
failed grounding.

Live-verified: "capital of France"/"photosynthesis" now correctly abstain
instead of confidently confabulating; "2+2" and "Hello!" unaffected. Tests:
`tests/test_generate_and_ground.py` (3 new, covering the bug, the fix, and
the extractive-only-mode non-regression). Full suite: 486 passed (up from
480), only the 3 pre-existing unrelated `ruff`-not-installed failures
remain.

MMLU/ARC real numbers with the fix in place, plus HumanEval/TruthfulQA/
SWE-bench, still to be run for real -- see below.

## Post-Item-2 FLUX pipeline phases (not separate 0.5.0 items, but real
## prerequisites for Item 6 — see below for why)

Item 6 (GRPO self-play) cannot run sensibly on a bare Phase 1 (pretrain-only)
checkpoint: Phase 1's corpus is flat text, no `<|user|>`/`<|assistant|>`/
`<|think|>` special-token structure at all — the model has never learned
what those tokens mean. `agentic_repair.py`'s ReAct loop and
`FluxProposer.propose(think=True)` both depend on that structure being
trained in. 0.4.0's precedent: Phase 1 -> Phase 2 (SFT) -> Phase 3 (CoT) ->
Phase 4 (QAT), chained. Continuing that chain for the new 0.5.0 Phase 1
base, user's explicit direction ("move onto the next phase of training").

- [x] **Phase 2 (SFT)** — `uchi/flux/sft_train.py`, unmodified (0.4.0's
      exact recipe: SQuAD 2.0 + Dolly-15K + CodeAlpaca + UltraChat-200K,
      fair-budgeted 12,500 each, seq_len=512 kept deliberately — SQuAD/
      Dolly/CodeAlpaca examples are capped at 300(context)+200(answer)
      tokens regardless, so 1024 would mostly pad, not add signal; the
      context-extension payoff matters more for CoT's longer reasoning
      traces. **COMPLETE** (2026-07-10 10:12 → 14:45, 743 steps, 4h33m):
      train loss 5.7330, **val loss 5.5115 (PPL 247.5, best)** — smooth,
      healthy convergence throughout, no crashes. Checkpoints:
      `uchi/flux/checkpoints/v050_phase2/{sft_best.pt,sft_epoch1.pt}`.
      **Known gap in this script, accepted not fixed**: checkpoints only
      save at end-of-epoch (no mid-run periodic save like Phase 1's
      `--checkpoint-interval`) — acceptable given both Phase 1 and this
      run went clean start to finish.
- [ ] Phase 3 (CoT distillation) — `uchi/flux/cot_distill.py`, unmodified
      (0.4.0's exact recipe: GSM8K + OpenOrca + Magicoder + CommitPackFT
      real teacher traces, fair-budgeted). Smoke-tested first (100
      examples): all 4 sources loaded, architecture handoff from the new
      SFT checkpoint clean, no crash. **Launched** (2026-07-10 14:48,
      detached/nohup, autonomous chain — user explicitly authorized
      auto-continuing Phases 3→4 without a check-in between each): `--base
      uchi/flux/checkpoints/v050_phase2/sft_best.pt`, checkpoints to
      `uchi/flux/checkpoints/v050_phase3/`. Log:
      `.uchi/corpus/train_0_5_0_phase3_cot.log`.
      **COMPLETE** (2026-07-10 14:49 → 17:59, 3h10m, 420 steps, 4 epochs):
      val PPL improved every epoch — 94.8 → 72.3 → 66.1 → **64.4 (best)**.
      Checkpoints: `uchi/flux/checkpoints/v050_phase3/cot_best.pt` (+ one
      per epoch). **Honest, expected gap vs 0.4.0**: 0.4.0's CoT run
      reached PPL 7.2 — this run's 64.4 is notably worse, because this
      whole chain inherits a deliberately-scoped-down Phase 1 (~20M
      tokens vs 0.4.0's much larger budget, chosen to fit a same-day
      timeline rather than 0.4.0's multi-day one). Not a bug — a known,
      accepted tradeoff from the timeline-scoping decision made earlier.
- [x] Phase 4 (QAT) — `uchi/flux/qat_train.py`, unmodified. Smoke-tested
      first (5 macro-steps): BitNet ternary quantization activated
      correctly, all 4 CoT sources loaded, clean handoff from the new CoT
      checkpoint. **COMPLETE** (2026-07-10 18:03 → 22:31, 4h28m, 600/600
      macro-steps, `cot_frac=0.5`): best CoT val loss 4.0068 (**PPL 55.0**),
      text val PPL 579.4. CoT reasoning quality held up well through
      quantization — minimal degradation from Phase 3's pre-quantization
      55-65 PPL range, the QAT recovery design (mixing CoT + general text
      every macro-step) worked as intended. Checkpoints:
      `uchi/flux/checkpoints/v050_phase4/{qat_best.pt,qat_00200.pt,
      qat_00400.pt,qat_00600.pt}`. **The full 4-phase pipeline (Phase 1
      pretrain → 2 SFT → 3 CoT → 4 QAT) is now done, one clean pass, no
      crashes at any stage.** Note: 5 of the ~7 background wait-monitors
      used to track this run got killed by something in the harness
      environment (not the actual detached training process, which was
      unaffected every time — confirmed by direct `ps`/log checks after
      each kill). Worth remembering: long-running background monitor
      loops here have a real lifetime limit; direct periodic checks are
      the reliable fallback, not a failure of the approach.

## Decision gate before Item 6/8/9 (2026-07-10, user's explicit direction)

Diagnostic finding (2026-07-10, real checkpoint comparison, not guesswork):
0.5.0's Phase 1 trained on ~20M tokens vs `train_v2.py`'s own design
default of ~491M (deliberately scoped down to fit a same-day timeline —
real measured throughput is ~750 tok/s regardless of seq_len on this GPU,
so the full budget would take ~7.6 days). CoT val PPL (64.4) is
correspondingly far worse than 0.4.0's (7.2) on the identical eval script/
data — expected consequence of the token-budget cut, not a bug; the
backbone architecture and Phases 2-4's recipes are unchanged from 0.4.0.

**Plan: don't preemptively pay for a multi-day retrain on a hunch — measure
first, then decide.**

- [x] Once Phase 4 finished: ran a real coding/proposer-quality check
      against `v050_phase4/qat_best.pt` — direct generation testing
      (added no CLI flags needed, used `build_generate_fn(checkpoint=...)`
      directly) + `benchmarks/swebench_real_eval.py --sample 2
      --checkpoint ...` (added a `--checkpoint` flag to the harness itself,
      a real reusable capability now, not a one-off hack — every future
      gate-check can point it at any candidate checkpoint).
      **Result: bad, clearly.** Generation test: "What is 2+2?" ->
      "The number of the total of $4.The answer is 1." (wrong answer,
      incoherent, though it does follow the trained "...The answer is X"
      format — not fully degenerate). The code-writing prompt produced
      **complete word salad, no code at all** — "The number of the given
      to be used in the first step, and then it is not only 1." SWE-bench
      check: **0/2 resolved**, both instances exhausted all 4 retry
      attempts without a working patch (7.8s/instance — fast because it's
      failing fast, not because it's doing anything well). This matches
      exactly what the PPL numbers predicted — not a surprise, a
      confirmation.
- [x] **Confirmed bad → Phase 1 restarted with a real budget** — user's
      explicit go-ahead given. Concretely, ahead of the original plan in
      one real way: pulling more corpus revealed **SWE-Gym-Raw alone is
      ~342.6M tokens** (not the ~126M estimated), so the local
      code/issue-diff corpus is ~365.9M tokens total (Stack v2 18.8M +
      SWE-Gym curated 4.5M + SWE-Gym-Raw 342.6M, all real, all already
      decontaminated — 854 SWE-Gym-Raw instances correctly excluded).
      User's call given that: use all of it, fill the remainder of a
      ~491M-token total with FineWeb-Edu (~125M) — **~75% code by token
      count this time**, a deliberate large shift from the first run's
      ~29%, matching the "considerably better at coding" goal directly.
      **Real, valuable optimizations found before launching, not assumed**:
      (1) parallelized the Stack v2 SWH fetch (`uchi/corpus_sources.py`'s
      `_iter_stack_v2`, bounded `ThreadPoolExecutor`) — sequential fetch
      was 4.1 files/s; found and fixed a real bug along the way
      (botocore's default `max_pool_connections=10` was silently capping
      concurrency regardless of thread count) — real confirmed rate after
      the fix: **26.2 files/s**, 6.4x. (2) Built a 0.5.0-specific pruned
      vocab (`scripts/build_pruned_vocab_0_5_0.py`, sampling the ACTUAL
      new corpus mix, not reusing 0.4.0's OpenWebText/Wikipedia-based one)
      — 32,018 tokens, 97.1% coverage, and this alone gave the model the
      exact same 64.0M-param size as 0.4.0's `flux_best.pt`. (3) The
      smaller pruned embedding table freed enough VRAM to double
      `--micro-batch` (2→4, same effective batch via `--grad-accum`
      16 vs 32) — confirmed real throughput 750→885→**1,313 tok/s**
      (calibrated at each step, not assumed), pushed until GPU memory
      left only ~2GB headroom, deliberately not pursued further to avoid
      OOM risk on an unattended multi-day run. **Net effect: same ~491M-token
      scope, real ETA cut from ~7.6 days to ~4.3 days**, zero training-scope
      compromise.
      **Launched** (2026-07-13 09:21, detached/nohup): 7,492 steps,
      `--eval-interval 250 --checkpoint-interval 250` (frequent enough for
      real resume-safety on a multi-day run), checkpoints to
      `uchi/flux/checkpoints/v050_phase1_v2/` (kept separate from the
      first attempt's `v050_phase1/`, both preserved). Log:
      `.uchi/corpus/train_0_5_0_phase1_v2.log`. Will re-run Phases 2-4 on
      this new base once it finishes, then re-clear the same decision
      gate before Item 6/8/9.
- [~] ~~If coding/overall performance is acceptable, proceed straight to
      Item 6/8/9 on the current chain~~ — **ruled out**, performance is
      bad (see above). `flux_best.pt` promotion, Item 6 (GRPO self-play),
      Item 8 (j-space MCTS), Item 9 (full re-validation: MMLU/ARC re-run,
      TruthfulQA/HumanEval baselines, the real SWE-bench number, and
      non-regression confirmation vs 0.4.0).

## Phase 1 v2 restart chain — Phases 1-3 status (2026-07-18)

- [x] **Phase 1 v2 (pretrain, real ~491M-token budget) — COMPLETE**
      (finished 2026-07-18 02:03, 7,492 steps): **val PPL 54.0** (best,
      loss 3.9883) — a massive improvement over the first attempt's
      PPL 490.0, confirming the token-budget cut was the real cause of
      that gap, not the architecture. Checkpoints:
      `uchi/flux/checkpoints/v050_phase1_v2/{ckpt_best.pt,ckpt_final.pt}`.
- [x] **Phase 2 v2 (SFT) — COMPLETE**, unmodified recipe, `--base
      v050_phase1_v2/ckpt_best.pt --pruned-vocab pruned_vocab_0_5_0_32k.json`:
      743 steps, single epoch, **val loss 2.808 (PPL 16.6, best)** — a huge
      jump over the first chain's PPL 247.5, directly reflecting the
      stronger Phase 1 v2 base. Checkpoints:
      `uchi/flux/checkpoints/v050_phase2_v2/{sft_best.pt,sft_epoch1.pt}`.
      **Session-boundary gotcha, resolved**: this run's own stdout log
      (`train_0_5_0_phase2_v2.log`) only ever captured a multiprocessing
      resource-tracker warning — Python's default full-buffering meant the
      real progress lines were sitting in an unflushed buffer when an
      *unrelated* python3 process got OOM-killed by the kernel in the same
      terminal ~2h after SFT had already finished and exited cleanly (confirmed
      via the checkpoint's own saved `step: 743` matching a full single epoch,
      and via `journalctl -k` showing the OOM'd PID was a distinct, later
      process). Lesson: a checkpoint's own saved metadata (step/loss) is the
      ground truth for "did this finish," not the wall-clock proximity to an
      unrelated crash in the same shell.
- [x] **Real bug found and fixed**: `cot_distill.py`'s default `torch.compile(model)`
      hit `Not enough SMs to use max_autotune_gemm mode` on this RTX 5070 and
      then stalled for 54+ minutes at ~87% CPU / **0% GPU util**, RSS growing
      to 14GB, zero training steps logged — inductor's autotune kernel search
      exploding on a GPU with limited SMs, not a hang. Confirmed no partial
      checkpoint existed (empty `v050_phase3_v2/`) before killing it — no lost
      work. **Fixed by passing `--no-compile`** (a flag the script already
      exposed for exactly this) — relaunch immediately hit `Step 00010` with
      55% GPU util, 4.4GB VRAM, RSS stable at 3.3GB. Standing habit going
      forward: always launch Phases 2-4 with `--no-compile` on this GPU unless
      compile is specifically re-verified safe.
- [x] **Phase 3 v2 (CoT) — COMPLETE** (2026-07-18 19:08 → 22:25, 3h17m,
      420 steps, 4 epochs), `--base v050_phase2_v2/sft_best.pt --pruned-vocab
      pruned_vocab_0_5_0_32k.json --no-compile`: val PPL improved every
      epoch — 8.0 → 6.9 → 6.6 → **6.5 (best)**. A massive jump over the
      first chain's PPL 64.4, directly reflecting the stronger Phase 1/2 v2
      bases all the way down the chain. Checkpoints:
      `uchi/flux/checkpoints/v050_phase3_v2/cot_best.pt` (+ one per epoch).
      Log: `.uchi/corpus/train_0_5_0_phase3_v2.log`.
      **Auto-chain worked as designed**: a detached shell script
      (independent of any Claude session) polled the training PID and, on
      clean exit, launched Phase 4 itself — verified via
      `.uchi/corpus/chain_phase3_to_phase4.log`. No manual relaunch needed.
- [x] **Phase 4 v2 (QAT) — COMPLETE** (2026-07-18 22:25 → 2026-07-19 03:13,
      4h48m, 600/600 macro-steps, `cot_frac=0.5`): val PPL improved every
      100 steps — text 112.9 → 89.7 → 79.1 → 86.9 → 79.0 → 80.0, CoT
      8.1 → 7.2 → 7.0 → 6.9 → 6.8 → **6.8 (best, final)**. Massive
      improvement over the first chain's QAT result (CoT PPL 55.0) —
      consistent with every earlier phase in this chain. Checkpoints:
      `uchi/flux/checkpoints/v050_phase4_v2/{qat_best.pt,qat_00[1-6]00.pt}`.
      **User's explicit direction (2026-07-19): promote unconditionally,
      regardless of gate-check result, and proceed straight into Items
      6/8/9** — no more gating on this check before promoting or starting
      those items (superseding the "Decision gate before Item 6/8/9"
      process below, which stays as a historical record of why the first
      chain was rejected).
      **Informational gate-check run anyway** (not a blocker, just a
      record): direct generation test — "What is 2+2?" → "The answer is
      1.The answer is 2.The answer is 2." (incoherent); code-writing prompt
      → non-functional garbled code. `swebench_real_eval.py --sample 3`:
      **0/3 resolved**, all failed fast (8.6s/instance). **Honest, not
      spun**: despite PPL improving by ~1-2 orders of magnitude over the
      first chain at every phase, real generation/task-execution quality is
      still poor — expected, not a bug. PPL measures next-token prediction
      quality on held-out CoT/text data; it does not measure instruction-
      following or code-repair competence, which is exactly what Item 6
      (GRPO self-play against real execution reward) exists to teach that
      pretraining/SFT/CoT/QAT alone don't. This result is consistent with
      that gap, not a contradiction of the strong PPL numbers.
      **Promoted unconditionally** (2026-07-19 03:13): backed up prior
      production `flux_best.pt` → `flux_best_pre_0_5_0.pt` first (reversible
      safety net), then copied `v050_phase4_v2/qat_best.pt` →
      `flux_best.pt`. `flux_best.pt` is now the 0.5.0 v2-chain checkpoint,
      not 0.4.0's. Full chain log: `.uchi/corpus/chain_phase4_to_promotion.log`.
