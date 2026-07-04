# Changelog

All notable changes to the Uchi project will be documented in this file.

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
