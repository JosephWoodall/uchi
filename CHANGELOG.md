# Changelog

All notable changes to the Uchi project will be documented in this file.

## [0.2.0] - 2026-06-22

### Added
- **Mathematical Simulation Stream (`datasets.load_math_corpus`)**: Real-time extraction of logical structure from math-equation streams.
- **Optimal Vector Retrieval (`SequenceGenerator(use_vector_retrieval=True)`)**: Fallback retrieval mechanism allowing the sequence generator to handle entirely novel prefixes via geometric lookup.
- **Ontological Process Predictor (`ProcessPredictor`)**: Models workflows using strict typing (`OntologicalState`, `OntologicalAction`).
- **Plural Simulation Engine (`LifelongSimulationEngine`)**: Parallelizes multiple independent predictor forests ("lives") and provides aggregated "wisdom of the crowd" voting.
- **Dedicated Documentation (`docs/`)**: Extensive documentation with interactive examples covering all features of v0.2.0.
- **Comprehensive Regression Suite**: Rewritten `tests/` utilizing Pytest for seamless regression validation.

### Changed
- Refactored project structure to cleanly package everything beneath `uchi/`.
- Optimized GitHub Actions configuration.

### Fixed
- Stabilized tabular encoding and temporal distributions handling.
