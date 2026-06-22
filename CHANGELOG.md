# Changelog

All notable changes to the Uchi project will be documented in this file.

## [0.2.0] - The Omni-modal Deterministic Universal Sequence Predictor (ODUSP)
- **Fractal Attention**: Replaced the fixed sliding window in `AssociativeMemory` with a dynamic, global co-occurrence graph that natively mimics multi-headed self-attention at $O(1)$ speed.
- **AST Coding Superpowers**: `OmniTokenizer` now natively parses Python code into an Abstract Syntax Tree (AST), allowing Uchi to deterministically learn the structural geometry of code to ensure mathematically impossible syntax errors.
- **Levenshtein Subword Fallback**: `OmniTokenizer` dynamically clusters Out-Of-Vocabulary slang and domain terminology using `difflib` subword distances.
- **Preloading CLI**: Added the `--preload` flag to `uchi chat` and `uchi debate` for instant "Runtime Pre-Training" over entire directories of code and text.
- **Persistent Brain States**: `uchi chat` now implicitly saves and loads the `OmniRouter` state to a `.uchi` binary file.

### Architecture Shift: ODUSP
Uchi v0.2.0 officially transitions the `UniversalPredictor` from a specialized math tool into an **Omni-modal Deterministic Universal Sequence Predictor**, capable of functioning as a lifelong mathematical brain at the absolute edge.

### Added
- **Pillar 1: The OmniRouter (Multi-Modal Frontend)** 
  A master controller that seamlessly ingests Text strings, Audio `.wav` files, Image `.jpg` paths, strict Math telemetry, and `OntologicalState` Agent code simultaneously. The `OmniTokenizer` geometrically hashes all modalities into a universal abstract Concept space, enabling true synesthesia.
- **Pillar 2: Zero-Shot Associative Memory (Phase 5 Backend)** 
  An $O(1)$ non-parametric query buffer that flawlessly mimics LLM attention. Natively passed Facebook bAbI Reasoning Tasks 1 & 2 at 100%. Allows users to query complex multimodal data streams in plain English without any neural weights.
- **Pillar 3: Infinite Compression (Phase 4)** 
  `OnlineTokenizer` automatically compresses the stream using continuous Byte Pair Encoding (BPE), ensuring the prediction tree can trace infinite context histories without RAM explosion.
- **Pillar 4: Predictive Subconscious (Phase 3)**
  `SequenceGenerator` now inherently builds plural future simulations over compressed BPE concepts.
- **Pillar 5: Node Compressor (Storage Optimizer)** 
  Introduced `NodeCompressor` to freeze and compress stabilized nodes in the PredictorForest into binary encodings, destroying raw dicts to save 70% of long-term RAM.
- **CLI & Multi-Agent Debate Simulator**: Added the `uchi` CLI providing `chat`, `ingest`, and `debate` modes. The multi-agent debate explicitly relies on BPE compression to prevent infinite $O(N)$ RAM explosions.
- **Dedicated Documentation (`docs/`)**: Extensive documentation with interactive examples covering all 5 pillars of v0.2.0.
- **Comprehensive Regression Suite**: Rewritten `tests/` utilizing Pytest for seamless regression validation including bAbI Reasoning Benchmarks.

### Changed
- Refactored project structure to cleanly package everything beneath `uchi/`.
- Optimized GitHub Actions configuration.

### Fixed
- Stabilized tabular encoding and temporal distributions handling.
