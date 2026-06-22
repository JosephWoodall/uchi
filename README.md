![Uchi Logo](docs/logo.png)

# Universal Sequence Predictor

[![PyPI version](https://img.shields.io/pypi/v/uchi_python.svg)](https://pypi.org/project/uchi_python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/uchi_python.svg)](https://pypi.org/project/uchi_python/)
[![Tests](https://github.com/JosephWoodall/uchi/actions/workflows/ci.yml/badge.svg)](https://github.com/JosephWoodall/uchi/actions/workflows/ci.yml)

**Uchi** is a Multi-Modal Deterministic LLM that functions as an infinite lifelong simulation engine. 

By routing Text, Audio, Images, Math Telemetry, and Python Agent code through a singular `OmniRouter`, Uchi natively translates messy multi-modal data into clean, abstract geometry. It learns to perfectly predict the future of the stream and achieves zero-shot cross-modal ad-hoc question answering.

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
