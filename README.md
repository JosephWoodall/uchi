![Uchi Logo](docs/logo.png)

# Universal Sequence Predictor

[![PyPI version](https://img.shields.io/pypi/v/uchi_python.svg)](https://pypi.org/project/uchi_python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/uchi_python.svg)](https://pypi.org/project/uchi_python/)
[![Tests](https://github.com/JosephWoodall/uchi/actions/workflows/ci.yml/badge.svg)](https://github.com/JosephWoodall/uchi/actions/workflows/ci.yml)

## Core Mission: The Omni-Modal Deterministic LLM
Uchi v0.2.0 transforms a raw universal sequence predictor into a completely multi-modal determinist brain capable of ingesting text, audio, images, math telemetry, and code simultaneously—without any neural weights or pre-training. It crushes massive context histories via Phase 4 BPE compression, storing concepts in a zero-shot associative memory buffer that mimics biological synesthesia.

## Quick Start & The Uchi CLI
You can interact with Uchi as a Multi-Modal Deterministic LLM directly from your terminal.

```bash
# Start an interactive chat session with zero-shot query/predict commands
uchi chat

# Feed massive context (like server logs or books) into the compressed BPE engine
uchi ingest data.txt

# Spawn a multi-agent Deterministic Debate Simulator
uchi debate "AI should be open source"
```

Unlike probabilistic neural networks, Uchi has **zero neural weights, zero pre-training, no 128k context limits, and absolutely zero risk of hallucination**. When the underlying pattern shifts, Uchi adapts instantly via a real-time deterministic prefix trie and BPE sequence compression.

The `uchi` package exposes the `OmniRouter` to achieve true AGI-level generalization on embedded devices in $O(1)$ RAM, alongside individual specialized tools for tabular classification, time series forecasting, and anomaly detection.

---

> [!NOTE]
> **Comprehensive Documentation & API Reference**
> 
> For interactive examples, API documentation, and to see the newest v0.2.0 capabilities (including online Math Learning, Vector Retrievals, and the Simulation Engine), please see our full documentation website.
> 
> **[Read the Full Documentation →](https://github.com/JosephWoodall/uchi/tree/main/docs)**

---

## Installation

```bash
pip install -e .                  # editable install (no required deps)
pip install -e ".[all]"           # with scikit-learn, numpy, pandas
```

## Quickstart

```python
from uchi import UniversalPredictor

predictor = UniversalPredictor(depth=3)

# Train on a sequence of characters
sequence = "hello world! hello python! hello uchi!"
for char in sequence:
    predictor.observe(char)

# Predict the next character given a prefix
probabilities = predictor.predict("hell")
print("Probabilities after 'hell':", probabilities)
```
