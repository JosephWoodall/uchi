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
      **Remaining, not code**: no full-scale corpus has actually been
      pulled and written to disk yet — that's a compute/storage/time-bound
      execution step (get the HF token authenticated, then run the CLIs
      at real scale), not a design or implementation gap.

## 2. Model Compaction + Context Extension (Item 2) — depends on 1

- [ ] Train from scratch on Item 1's decontaminated corpus.
- [ ] Micro-benchmark attention FLOPs/memory at 512/1024/2048 tokens vs.
      current 256 — pick max_seq_len where attention cost stays
      subdominant to checkpointed SSM scan cost. Arithmetic, not a
      research phase.
- [ ] Re-initialize HiPPO A_fast/A_slow timescales proportional to the new
      max_seq_len (current 2–10 / 10–34 horizons are tuned for 256 tokens,
      do not just reuse them).
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
- [ ] Stage 2 (before self-play scales to full-repo suites, not before
      Item 6 starts): container/VM-level isolation (Docker/gVisor-class).
- [ ] Stage 2: test-impact-analysis subsetting for the broader training
      corpus, where no FAIL_TO_PASS/PASS_TO_PASS annotation exists.

## 6. Execution-Verified Code Self-Play (Item 6) — depends on 2, 5

- [ ] GRPO (`grpo.py`) using Item 5's sandbox output as reward.
- [ ] **Curriculum**: bucket curated corpus by patch size (lines changed),
      train easy-to-hard — required to avoid degenerate zero-variance
      advantage on sparse reward.
- [ ] Log every trajectory in a shared Thought→Action→Observation schema
      (defined here, reused by Item 7 and Item 8 — don't wait for Item 7's
      full harness to define the schema).
- [ ] **Non-regression checkpoint (diagnostic, not a gate)**: re-run fast
      MMLU/ARC/HumanEval subset; separately record whether held-out
      pass-rate curve is actually rising. Item 8 proceeds regardless of
      the result — this just tells you what Item 8 is working with.

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

- [ ] Port `world_model.py` (DynamicsHead, ValueHead) + `mcts.py`
      (PUCT search, latent-only rollout, no intermediate decoding).
- [ ] Redefine "action" as hypothesis/reasoning-step level, using Item 6's
      Thought-tagged trajectory segments as the discrete units (mean-pooled
      embeddings, same approach as `proprioception.py`).
- [ ] Hook ValueHead to Item 5's real pass/fail as reward, trained via
      GRPO same as Item 6.
- [ ] **Operational "no precedent" check**: on failed MCTS search, measure
      distance to nearest known hypothesis embedding. Close = tune rollout
      budget. Far = widen Item 1's corpus / Item 6's curriculum — don't
      just throw more search compute at a coverage gap.
- [ ] **Non-regression checkpoint**: re-run fast MMLU/ARC/HumanEval subset
      again — second sequential fine-tune on top of Item 6's, forgetting
      compounds across passes if unchecked.

## 9. Full Benchmark Re-Validation (Item 9) — depends on 2, 4, 6, 7, 8

- [ ] Re-run MMLU, SWE-bench, ARC against the finished model.
- [ ] Establish first baselines on TruthfulQA and HumanEval.
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
