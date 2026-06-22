# Changelog

All notable changes to the Uchi project will be documented in this file.

## [0.2.0] - 2026-06-22

### Added
- **Mathematical Simulation Stream (`datasets.load_math_corpus`)**: Real-time extraction of logical structure from math-equation streams.
- **Ontological Process Predictor (`ProcessPredictor`)**: Models workflows using strict typing (`OntologicalState`, `OntologicalAction`).
- **Plural Simulation Engine (`LifelongSimulationEngine`)**: Parallelizes multiple independent predictor forests ("lives") and provides aggregated "wisdom of the crowd" voting.
- **Infinite Context Engine (`OnlineTokenizer`, `NodeCompressor`)**: Radically compresses real-time streams and prunes dead nodes, allowing infinite $O(1)$ RAM scaling without catastrophic forgetting.
- **Semantic Abstraction & Associative Memory (`SemanticTokenizer`, `AssociativeMemory`)**: Replaces vector retrieval with non-parametric zero-shot geometric attention, granting the deterministic engine LLM-level creative generalization and Ad-Hoc Question Answering without neural weights.
- **Dedicated Documentation (`docs/`)**: Extensive documentation with interactive examples covering all 5 pillars of v0.2.0.
- **Comprehensive Regression Suite**: Rewritten `tests/` utilizing Pytest for seamless regression validation including bAbI Reasoning Benchmarks.

### Changed
- Refactored project structure to cleanly package everything beneath `uchi/`.
- Optimized GitHub Actions configuration.

### Fixed
- Stabilized tabular encoding and temporal distributions handling.
