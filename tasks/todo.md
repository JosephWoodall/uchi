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

- [ ] Source (a)/(b)/(c) general-code portion — pick and document which.
- [ ] New required source: real GitHub issue→diff pairs at scale (issue
      description + diff, not diff-only).
- [ ] **Decontamination filter, hard gate before Item 2**: exclude every
      repo in SWE-bench's fixed eval set (astropy, django, flask,
      matplotlib, pytest, requests, scikit-learn, sympy, + full
      full/verified/lite list) at the **repo level**, not just
      instance-level dedup.
- [ ] Delegate: standalone decontamination script, reusable by any future
      corpus addition, not a one-off inline filter.

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

- [ ] Unchanged from prior releases' retrieval work.

## 3.5. Repo-Scale Code Retrieval (Item 3.5) — parallel, no dependency

- [ ] Delegate: port `rag.py`'s `LocalKnowledgeAnchor` (FAISS +
      sentence-transformer + cross-encoder rerank + graph edges).
- [ ] Per-repo index (file/function chunks) built at the start of each
      SWE-bench task; graph edges track import/call-graph for cross-file
      navigation.
- [ ] Confirm this index is exempt from Item 1's decontamination list —
      it's the actual task repo at eval time, not training data.

## 4. Retrieval Scaling Validation (Item 4) — depends on 3, 3.5

- [ ] Benchmark Item 3's flat fact retrieval and Item 3.5's code retrieval
      **separately** — cross-encoder rerank cost doesn't transfer from
      Item 3's numbers.

## 5. Execution Sandbox (Item 5) — starts immediately, no dependency

- [ ] Delegate: port `rl_env.py`'s `ExecutionSandbox`, extend to apply a
      patch to a sandboxed repo copy and run its real test suite
      (pass/fail + partial credit).
- [ ] Stage 1 (unblocks Item 6 now): subprocess + timeout + small held-out
      test subset.
- [ ] Stage 2 (before self-play scales to full-repo suites, not before
      Item 6 starts): container/VM-level isolation (Docker/gVisor-class).
- [ ] Stage 2: test-impact-analysis subsetting — map changed files/funcs
      to touching tests via coverage/import graph, run only those.

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

- [ ] Wire Item 5's sandbox into the oracle as a new cascade layer,
      additive-only, same contract as every existing layer.
- [ ] Port `react_agent.py` + `agentic_harness.py`; register `run_tests`,
      `apply_patch`, `read_file`, `grep_repo` in `tools.py`'s pattern.
- [ ] **Tri-state outcome policy**: pass / fail / timeout-or-flaky.
      Timeout-or-flaky = abstain, full stop — does NOT fall back to
      word-overlap/entailment/numeric/relational (they have no opinion
      about code execution).
- [ ] **Retry cap: 3–5 attempts**, hard number, no unbounded iteration.
- [ ] Port `parallel_thought.py`'s vectorized batch generation into
      `task_config_cache.py` so dynamic-N doesn't cost N× latency.

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
- [ ] Report the real SWE-bench number plainly against the 85% target.
- [ ] Confirm non-regression contract held end-to-end vs. 0.4.0 baseline
      (ODUSP recall, cascade precision, retrieval accuracy).

## Addendum, done alongside this file

- [x] `tasks/core_principle.md` — added clarification that Item 6's GRPO
      reward (real test execution) is not the previously-rejected
      "RL-tune the proposer against the verifier's reward" failure mode.
