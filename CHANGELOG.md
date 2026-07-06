# Changelog

All notable changes to the Uchi project will be documented in this file.

## [0.4.0] - In Progress - MetaUchi: giving Uchi hands

Wraps the v0.3.0 grounded Q&A engine in **`MetaUchi`**, the new default
orchestrator (`from uchi import Uchi`) — tool calling, a sandboxed Python
scratchpad, web search, multi-step goal tracking with pause/resume, and macro
distillation, on top of the same verified engine. The single-instance engine
moves to `uchi.Core` (unchanged internals) for anyone who wants the raw,
un-orchestrated node. **The grounded-or-abstain contract from 0.3.0 is
unchanged** — autonomy never gets to assert something ungrounded.

FLUX itself is mid-retraining as of this entry (Phase 1 of 4 complete, Phase 2
running) — see "FLUX scale-up" below and `docs/training.md`.

### Added
- **Verifier upgrade — layered veto architecture (Item 17), code and data
  pipeline built, training not yet run**: `EntailmentClassifier`
  (`uchi/flux/verifier_model.py`) — a from-scratch entailment classifier on
  its own embedding table, trained on real MNLI+SNLI, plugs into
  `FactCheckOracle` as a strictly additive veto — it can reject a claim the
  deterministic check already passed, never accept one it rejected.
  `uchi/numeric_plausibility.py` adds a second additive layer flagging
  implausible numbers via median+MAD outlier detection on real numeric
  facts pulled from the semantic index — the original plan to reuse
  `UniversalPredictor`/`AnomalyDetector` was tested and found unsuitable
  (that predictor needs sequential/ordered data; scattered numeric facts
  have no such order, and it scored 55 and 50,000 nearly identically in
  testing). Both default to `None` in `FactCheckOracle`'s constructor —
  zero behavior change until something is actually trained/fitted.
  `OODDetector` (Mahalanobis distance over the classifier's own pooled
  latent space) gates the entailment veto rather than acting as an
  independent one — out-of-distribution input suppresses the classifier's
  judgment to "no opinion" instead of trusting a confident-but-baseless
  verdict. `uchi/verifier_flywheel.py` closes the self-improvement loop:
  reuses the same entailment checker to detect when a user's next turn
  contradicts Uchi's prior answer, wired into `Core.ask()` purely
  observationally, exporting confirmed corrections as real training data
  for the next verifier run — no fabricated examples, empty until real
  conversations happen.
- **`MetaUchi`** (`uchi/meta.py`): the default facade, wraps `Core` and adds
  everything below. Forwards `ask()`/`learn()`/`ingest()` unchanged.
- **Tool Calling Interface** (`uchi/tool_calling.py`): `<|tool_call|> name(args)
  <|end_tool|>` grammar halts generation, dispatches to a registered Python
  function, splices the result back in. `<|tool_call_async|>` fires several
  calls concurrently. Every dispatch is logged (name, args, result/error,
  duration) from day one.
- **Filesystem sandbox** (`uchi/workspace.py`): tool-driven file reads/writes
  are resolved against a fixed `.uchi/workspace` root; realpath-based checks
  block both path-traversal and symlink escapes.
- **Python scratchpad** (`uchi/scratchpad.py`): arbitrary multi-line code
  execution in the sandbox, returns stdout/stderr/return code as text. Also
  `lint_python` — a `ruff`-based static-analysis pass before execution.
- **Self-Healing Loop Prevention** (`uchi/loop_guard.py`): an exact repeat of
  a previously-failed tool call, MCTS code candidate, or (closing a gap
  identified after initial release) swarm delegation attempt is blocked
  outright rather than retried, forcing a structurally different path.
- **Goal State & Memory Compaction** (`uchi/goal_state.py`): tracks a
  multi-step task's intent and running notes; compacts raw tool logs
  (extractive only, never a paraphrase) once they grow large.
- **Agentic Checkpointing** (`uchi/checkpoint.py`): `checkpoint()`/`resume()`
  serialize goal state, tool history, and pending HitL questions to disk —
  survives across processes.
- **Macro Skill Distillation** (`uchi/macro.py`): a successful multi-step
  task's tool-call sequence (errors already excluded) distills into a
  reusable fast-path tool, ingested into the knowledge index.
- **Human-in-the-Loop Yielding** (`uchi/hitl.py`): an explicit
  `<|yield_to_user|>`, or a tool call auto-escalated after a repeated
  failure, pauses and asks the human instead of guessing or looping.
- **IQ Task Router** (`uchi/iq_router.py`): a cheap heuristic gates
  `SwarmSynthesizer`'s FLUX-round-trip decomposition — most questions are
  atomic and shouldn't pay for it.
- **Dynamic Tool Registration** (`uchi/tool_learning.py`): `learn_tools(path)`
  parses a plain Python file's top-level functions into callable tools, no
  source changes required.
- **Web search as a tool** (opt-in via `Core(web_search=True)`), plus a
  default fallback: `GenerateAndGround` silently retries against live web
  search when local retrieval comes up empty, before abstaining.
- **OpenAI-compatible endpoint** (`POST /v1/chat/completions`) and **SSE
  streaming** (`POST /ask/stream`) — the latter streams real "thought" events
  (swarm decomposition, candidate generation, oracle pruning) as they occur,
  then a final "speech" event.
- **Skill sharing**: `export_skill()`/`import_skill()` package a verified
  skill as a `.uchi_skill` file (same markdown+frontmatter format skills
  already use) for sharing between instances.
- **Persistent user preferences** (`uchi/user_profile.py`): `remember_
  preference()` writes to `.uchi/user_profile.md`, auto-ingested on every
  `Core()` construction — cross-session memory with no vector database.
- **Enterprise data silos** (`uchi/data_silo.py`): `Core(allowed_paths=...,
  denied_paths=...)` restricts which directories `ingest()` may touch.
- **Front-Desk tone pass**: `ask_friendly()` rewrites a dry grounded answer
  in a warmer tone, verified against the original fact with the same
  `FactCheckOracle` before being trusted — falls back to the dry answer if
  the rewrite isn't grounded.
- **Observability exporter** (`uchi/observability.py`): tool-call log as
  OpenTelemetry-shaped spans (JSON Lines), no SDK dependency required.
- **Glass Brain** (`uchi/tui/glass_brain.py`): live tool-call trace panel in
  the TUI.
- **OPS Benchmark Harness** (`benchmarks/ops_benchmark.py`): Operations Per
  Second for the autonomous tool-calling loop — baseline 41.66 ops/sec,
  10-step task, 25-step context coherence confirmed.
- **`tests/test_edge_cases.py`**: black-box adversarial testing of every
  public `Core` method, existing and new — self-checking (fails if a new
  public method ships without corresponding coverage), so it can't quietly
  go stale as the API grows.

### Fixed
- `build_generate_fn` (`uchi/flux/inference_engine.py`) unconditionally
  loaded the full ~100K-token tokenizer regardless of the checkpoint's
  actual vocab size — the same mismatch already fixed on the *training*
  side (`sft_train.py --pruned-vocab`) was never fixed on the *inference*
  side. A pruned-vocab checkpoint (0.4.0 Item 0 — Phase 1 already trained
  with one) would have produced out-of-range token IDs the moment
  `Core()` tried to actually use it, once promoted from its isolated
  training directory to the default checkpoint path. Now auto-detects the
  mismatch from the checkpoint's own inferred `vocab_size` (the same
  "infer from tensor shapes" approach already used for `d_model`/
  `n_layers`/`d_state`) and loads the matching pruned tokenizer, with an
  explicit `pruned_vocab` override on `build_generate_fn`/
  `FluxProposer.load` for anything other than the default path. Verified
  against both the real, completed Phase 1 checkpoint (pruned, loads and
  generates correctly) and the current production `flux_best.pt` (full
  vocab, unchanged behavior).
- `FactCheckOracle.is_grounded()` rejected every claim when no relevant
  evidence was retrieved — including purely conversational replies that
  never asserted anything checkable ("Hello! How can I help you today?"),
  which got vetoed the same way a fabricated fact would, causing `ask()` to
  abstain on a bare greeting. Now, when evidence is empty, only a claim's
  *specific* content (proper nouns, numbers) needs support — a claim built
  entirely from generic vocabulary has nothing checkable in it and is
  emitted; a claim naming something specific with zero evidence is still
  rejected exactly as before. The pronoun "I" is explicitly excluded from
  the proper-noun check (always capitalized regardless of position, a
  spelling convention rather than a specificity signal). Every relaxed-path
  decision is logged (`oracle.relaxed_pass_log`) for future refinement.
  Two upstream honesty gates in `generate_and_ground.py` (`_known_fraction`
  and the retrieval-similarity check) had the same problem and are now
  gated on whether the *question itself* asserts anything specific — a
  real factual question with weak evidence still abstains exactly as
  before; a generic conversational one proceeds to candidate generation,
  where the oracle's relaxation makes the final call. `ev_texts` passed to
  the proposer/oracle now excludes evidence below the similarity
  threshold, so a topically-irrelevant-but-real match doesn't force the
  strict check on a candidate that never asserted anything the retrieved
  text could support in the first place. Covered by
  `tests/test_oracle_no_evidence_relaxation.py`.
- `/v1/chat/completions` accepted a full OpenAI-shaped `messages` array but
  silently discarded everything except the last message. Fixed by threading
  the prior turns through as conversation context. Separately, the REST
  server's `_router` is one global `Core` instance shared by every caller
  (constructed once in `lifespan()`) — reading/writing `self.episodic_memory`
  for that per-request context would have mixed different clients'
  conversations together. Fixed via a new `Core.ask(...,
  conversation_context: str | None = None)` parameter that overrides
  `self.episodic_memory` for that call only, without reading from or writing
  to it — REST callers get correct per-request history with no
  cross-contamination, SDK/TUI's single-instance session behavior unchanged.
- `ask("/")` (bare slash, no command) raised an unhandled `IndexError` from
  slash-command parsing instead of a clean "unknown skill" message.
- `_parse_kwargs` (tool-call argument parsing) silently dropped any
  multi-line `code=` argument — a raw newline inside a quoted string broke
  `ast.parse`, caught by a bare `except`. Broke every real multi-line
  `run_python`/`lint_python` call.
- `run_python`'s registry-facing wrapper never raised on script failure, so
  the loop guard and HitL auto-escalation couldn't see scratchpad errors as
  failures.
- Two duplicate `REPLOracle` classes (`code_engine.py`, `procedural_
  memory.py`) consolidated to one — the duplicate was live and load-bearing
  in `generate_and_ground.py`'s empirical-synthesis fallback.
- `uchi/simple.py`'s docstring referenced a `u.router`/`OmniRouter` that was
  never actually assigned — the real object is `self.pipeline`
  (`GenerateAndGround`).
- SFT/CoT data-source loading (`sft_train.py`, `cot_distill.py`) shared one
  cumulative example-count cap across all sources — the first source to
  load (SQuAD / GSM8K) reached the cap before exhausting its pool, silently
  starving every source loaded after it to ~1 example. Each source now gets
  an independent budget.
- `pyproject.toml`'s `package-data` never included `uchi/skills/*.md` — every
  documented slash command (`/classify`, `/forecast`, etc.) silently failed
  on every real `pip install` with zero warning.

### FLUX scale-up (in progress)
- **Vocab pruning** (`uchi/flux/vocab_prune.py`, `scripts/build_pruned_
  vocab.py`): the full cl100k_base vocab is ~66% of the model's params at
  `d_model=768`; a compact 32K vocab built from the real training corpus
  mix measures 97–99% token coverage, freeing ~52M params. `PrunedTikToken
  Tokenizer` is a drop-in wrapper; `--pruned-vocab` on `train_v2.py`/
  `sft_train.py` opts in.
- **Fused SSM scan**: `UCHI_FUSE_SSM_SCAN=1` kernel-fuses the existing
  sequential scan (same algorithm — a parallel scan was already tried and
  reverted for measured memory-bandwidth reasons) via `torch.compile` over
  the whole loop. Measured 3.08× forward speedup on an RTX 5070.
- **New training data**: UltraChat-200k (Phase 2, conversational tone),
  OpenOrca, Magicoder-OSS-Instruct, and CommitPackFT (Phase 3, general,
  code, and repo-level code-change reasoning CoT) — closing gaps where the
  prior data mix had zero natural dialogue, math-only CoT, and no
  understanding of *existing* code changes (only from-scratch generation).
  CommitPackFT (originally scoped for 0.5.0, pulled forward since Phase 3
  hadn't started training yet) pairs a real diff with a real, human-written
  commit message; the `<|think|>` content is a real, computed fact about
  the diff (lines added/removed), not a fabricated reasoning narrative —
  the dataset has no separate reasoning-steps field the way GSM8K does.
  Loaded via the raw per-language JSONL file directly, since
  `bigcode/commitpackft`'s HF dataset-script loader was removed by a
  `datasets` library version bump; the underlying data is unaffected.
- Phase 1 (pretrain) complete: 64.0M params (32K vocab), val loss 5.82 →
  4.11 over 5,000 steps, no NaN/crash. Phase 2 (SFT) proof run validated
  the new tokenizer + data sources end-to-end; full run in progress.

## [0.3.0] - 2026-07-04 - FLUX + Uchi: a trained proposer behind the verifier

Adds **FLUX**, a small (~116M) from-scratch SSM/attention model, as the swappable
**Proposer** behind Uchi's Generate-and-Ground **Verifier**. FLUX proposes; Uchi
grounds or abstains. Trained end-to-end (pretrain → SFT → CoT → ternary QAT) and
shipped with a premade general-knowledge brain, so `Uchi()` works out of the box.

### Added
- **FLUX training pipeline** (`scripts/train_all.sh`): pre-tokenize → pre-train →
  SFT → CoT distillation → ternary QAT, producing the canonical
  `uchi/flux/checkpoints/flux_best.pt`. Documented in `docs/training.md`.
- **`scripts/pretokenize.py`**: FineWeb-Edu → uint32 memmap `.bin`; makes training
  GPU-bound (~10× faster than the streaming/tokenizing loop).
- **`FluxProposer`** (`uchi/proposer.py`): loads `flux_best.pt` and drives the
  proposer; degrades to `None` (extractive/abstain) when no checkpoint is present.
  `propose(..., think=True)` primes CoT's trained reasoning-before-answering
  (`<|think|>`) instead of a bare answer — wired into the main answering path.
- **Premade general-knowledge brain** (`uchi/data/embeddings.pt`, `scripts/build_brain.py`):
  46K-word skip-gram vocabulary + 130K+ pre-embedded Wikipedia passages, so a
  fresh `Uchi()` can retrieve and answer with zero `learn()` calls. Fixes a
  root-cause gap: `SemanticIndex._vec()` only embeds words already in its
  vocabulary — without a shipped vocabulary, `learn()` was a silent no-op.
- REST API `POST /ask` (`{"query": ...}` → `{"answer": ...}`) and `GET /health`,
  mirroring the SDK. `uchi tui` / `uchi serve` subcommands.
- TUI: shows FLUX's live internal dialogue (reasoning trace, Devil's Advocate
  critique, verification outcomes) with clear labels, not just status text;
  friendlier boot/help copy.
- Trained weights (`flux_best.pt`, optimizer state stripped for inference) and
  the premade brain ship via Git LFS — `git clone` + `git lfs pull` gets a
  working system, not just source.
- `tests/test_flux_uchi.py`: architecture coverage (compounding, abstention,
  proposer seam, skills, CLI helpers, REST API).

### Fixed
- SFT training had no real gradient accumulation despite `--grad-accum` being
  passed — the LR schedule hit its floor ~3% into the epoch and trained at
  near-zero LR for the rest. Fixed to match `cot_distill.py`'s correct pattern.
- QAT's recovery data was 100% general text — zero exposure to CoT's format
  silently erased the reasoning behavior CoT had just installed. Now mixes CoT-
  formatted batches (masked loss) with general text, both under quantization;
  "best" checkpoint selection now tracks CoT val loss.
- Production inference (`build_generate_fn`) unconditionally ran with
  quantization off regardless of checkpoint, and built prompts as literal
  `"<|user|>..."` text through the general tokenizer instead of the special-
  token ids training actually used — both silently degraded every real
  `Uchi.ask()` → FLUX call. Checkpoints now self-report whether they were
  QAT-trained; prompts now match training exactly.
- `benchmarks/{mmlu,swebench,arc}_benchmark.py` imported the deleted
  `omni_router.OmniRouter` — rewritten against `Uchi()` directly.
- `pyproject.toml` package-data was missing `data/*.pt` — `pip install` shipped
  no trained models at all (decoder/answerability/chat_decoder/embeddings).
- Restored `uchi/predictor.py` → `/classify`, `/regress`, `/anomaly`, `/forecast`,
  `/tsclassify` work again; added the lazy `Uchi.predictor` SDK API.
- Rewrote `uchi/cli.py` off the deleted `omni_router` (TUI + serve now run).
- Rerouted the `code` and `overview` skills onto the FLUX + Uchi components.
- Fixed the test suite (was fully broken on the deleted `omni_router`): removed
  tests for deleted subsystems, repaired `conftest`.

### Docs
- New `docs/training.md`; README documents training from scratch and using the
  shipped Git LFS weights. Corrected capability overclaims across the docs
  (benchmarks/reasoning/architecture) to reflect a small proposer + honest
  verifier; `docs/reasoning.md` now describes the real verification mechanisms
  (CoT think-trace, REPL empirical synthesis, cross-examination) instead of a
  `ReasoningEngine`/`sympy` claim that was never implemented; fixed the mkdocs nav.
- `.agents/skills/release_readiness/SKILL.md` updated: capability benchmarks are
  a dashboard (not a pass/fail gate), premade-brain sanity check added, legacy
  `run_benchmarks.py` (v0.2.0 OmniRouter internals) removed from the checklist.

### Generate-and-Ground: the trustworthy, no-LLM foundation

Everything above builds on the ground-up rearchitecture that shifted Uchi's
identity from "universal sequence predictor" to **a grounded assistant that
verifies factual answers and abstains rather than confabulate**.

### New architecture
- **Generate-and-Ground** (`uchi/generate_and_ground.py`): the primary `ask()`
  path — retrieve evidence → generate a candidate → fact-check → emit or abstain.
- **Semantic retrieval index** (`uchi/retrieval.py`): skip-gram-embedding passage
  index over the brain; hybrid lexical+semantic ranking. Built into the brain by
  `incremental_builder`; fed live by `learn()`.
- **Fact-check oracle** (`uchi/oracle.py`) + **answerability gate**
  (`uchi/answerability.py`, a from-scratch classifier trained on SQuAD 2.0): the
  honesty gates. The system abstains when the evidence doesn't answer the question.
- **Neural answer decoder** (`uchi/decoder.py`): small from-scratch BiGRU seq2seq,
  retrieval-conditioned. Rough but grounded; the oracle catches its fabrications.
- **Three-lane router** (`uchi/intent_router.py`): `ask()` routes to skills,
  free-generated social chit-chat (`uchi/conversation.py` — no oracle, no facts to
  verify), or grounded factual answering.
- **Trustworthiness benchmark** (`benchmarks/trustworthiness.py`): SQuAD 2.0
  coverage / precision@answered / honest-abstention / hallucination-rate. Replaces
  MMLU/ARC/SWE accuracy as the primary scorecard.

### Removed (superseded "Family C" machinery, ~9k lines)
- `convergent_engine`, `tree_search_engine`, `grpo`, `grpo_offline_trainer`,
  `calibration`, `grammar_mask`, `omni_evaluator`, and the SSM QA-discrimination /
  GRPO training path from `omni_router`. Trie-based text generation (garbled) is
  retired in favour of the neural decoder. The trie is kept as recall + grounding.

### Honest status
- Trustworthy on clearly-unknown queries (abstains) and social turns. **Not yet
  trustworthy on hard open-domain QA** — retrieval+generation precision (~57%) is
  the current ceiling and the primary roadmap item.

## [0.2.0] - The Omni-modal Deterministic Universal Sequence Predictor (ODUSP)

### Routing & Pipeline
- **Routing Layer**: `ProceduralMemory` classifies intent via keyword matching before tokenization and prepends a routing hint — eliminates blind single-pipeline routing.
- **GRPO Value Head**: SSM value head trains online from user sentiment and code evaluation signals via Group Relative Policy Optimization (DeepSeek-R1 method). Replaces random-weight hallucination gate.
- **CPUVectorMemory**: Persistent numpy/JSON vector store replaces in-memory AssociativeMemory buffer. SSM state vectors survive restarts.
- **Cold-Start Bootstrap**: First launch automatically ingests Python stdlib function patterns and Wikipedia fact triples via `_bootstrap_knowledge()`.
- **Cosine Similarity Retrieval**: AssociativeMemory query now uses SSM-encoded cosine similarity instead of token overlap scoring.
- **Root Problem Fix — Stream After Answer**: Trie now trains on complete `<|user|> query <|assistant|> response` sequences only after generation, preventing partial-sequence corruption.

### Interfaces
- **TUI**: Full terminal UI with real-time MCTS telemetry, MoE routing histograms, `/load`, `/save`, and feedback commands.
- **REST API**: FastAPI server with `/chat`, `/metrics`, `/debug/walk`, `/bootstrap` endpoints.
- **Python SDK**: `OmniRouter`, `load_brain`, `save_brain` for programmatic access.

### Benchmarks & Correctness
- **Pre-flight Classify**: Dynamic MCTS budget per query type (factual=5 sims, generative=20); O(1) greedy bypass for peaked-trie factual answers.
- **SSM Gate Bypass**: Untrained SSM no longer rejects peaked-trie factual answers; gate is bypassed when preflight classifies a query as factual.
- **CoherenceOracle Min-Length**: Lowered 5→1 so single-token factual answers (e.g., "paris") pass the coherence check.
- **Pre-load Recall: 80.0%** (40/50 factual Q&A pairs across geography, science, history, Python/CS).
- **Zero Catastrophic Forgetting: 100.0%** (10/10 anchor facts recalled after 1 000 noise facts).
- **Inference Latency: 2 333 ms** — 7.6× faster than pre-optimization baseline via dynamic MCTS budget scaling.
- **Hallucination Rate: 0%** — structural guarantee from trie boundary enforcement.

### RAG & Retrieval
- **Web-Content Direct Return**: When MCTS fails on a sparsely-trained trie, retrieved web content is returned directly via a sentinel tag, preventing silent hallucination.
- **Memory False-Positive Filter**: Keyword-overlap guard on cosine memory matches rejects SSM false positives with zero semantic overlap.
- **Word-Root Bias Matching**: MCTS bias scoring strips synset suffixes (`energy.n.01` → `energy`) before matching plain-text web content.
- **Hallucination Gate Bypass for Grounded Replies**: SSM gate skipped when retrieval context is present.
- **Universal Builder Pipeline**: Consolidated 5-stage ingest pipeline (Dolly, Hermes, Wikipedia, MMLU, GSM8K, SWE-Bench).
- **N-Gram Backoff Smoothing**: MCTS tree search falls back gracefully from N=8→2 grams during cold-start traversal.
- **InfoNCE Geometry**: SSM uses InfoNCE loss + L2 normalization; Holographic Reduced Representations via FFT.
- **Grammar-Constrained Sampling**: `GrammarMask` filters invalid tokens inside Python blocks during MCTS expansion.
- **`--wipe` Flag for `benchmarks/run_benchmarks.py`**: Deletes all brain files pre-benchmark to trigger Universal Builder rebuild.
- **Web Search Coverage**: `perform_web_search` default `max_results` raised 3→5.

---

## [0.1.0] - ODUSP Foundation

- **Fractal Attention**: Replaced the fixed sliding window in `AssociativeMemory` with a dynamic, global co-occurrence graph that natively mimics multi-headed self-attention at $O(1)$ speed.
- **AST Coding Superpowers**: `OmniTokenizer` now natively parses Python code into an Abstract Syntax Tree (AST), allowing Uchi to deterministically learn the structural geometry of code.
- **Natural Autocomplete CLI**: Re-engineered the CLI to naturally autocomplete `<|assistant|>` boundaries without forcing them, ensuring strict geometric coherence. Bootstrapped with over 100 isolated conversation turns via `persona.txt`.
- **Levenshtein Subword Fallback**: `OmniTokenizer` dynamically clusters Out-Of-Vocabulary slang and domain terminology using `difflib` subword distances.
- **Fluid Dual-Pass CLI**: Removed all clunky subcommands (`serve`, `chat`, etc) in favor of a unified REPL that automatically routes user inputs through the `AssociativeMemory` graph before seeding the generative sequence (Zero-Shot RAG).
- **Structured Context Injection**: `uchi --preload` natively wraps source code in mathematical boundaries (`<|file: filename|>`) to eliminate generative context bleeding.
- **Persistent Brain States**: `uchi` now implicitly saves and loads the `OmniRouter` state to a `.uchi` binary file.
- **Pillar 1: The OmniRouter (Multi-Modal Frontend)**: A master controller that seamlessly ingests Text, Audio `.wav`, Image `.jpg`, Math telemetry, and `OntologicalState` objects simultaneously via a universal geometric concept space.
- **Pillar 2: Zero-Shot Associative Memory**: An $O(1)$ non-parametric query buffer. Natively passed Facebook bAbI Reasoning Tasks 1 & 2 at 100%.
- **Pillar 3: Infinite Compression (Phase 4)**: `OnlineTokenizer` compresses streams via continuous BPE, preventing $O(N^2)$ RAM explosion.
- **Pillar 4: Predictive Subconscious (Phase 3)**: `SequenceGenerator` builds plural future simulations over compressed BPE concepts.
- **Pillar 5: Node Compressor**: `NodeCompressor` freezes and compresses stabilized nodes into binary encodings, saving 70% long-term RAM.
- **Comprehensive Regression Suite**: Rewritten `tests/` utilizing Pytest for seamless regression validation including bAbI Reasoning Benchmarks.
