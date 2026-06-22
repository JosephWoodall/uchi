![Uchi Logo](docs/logo.png)

# Universal Sequence Predictor

[![PyPI version](https://img.shields.io/pypi/v/uchi_python.svg)](https://pypi.org/project/uchi_python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/uchi_python.svg)](https://pypi.org/project/uchi_python/)
[![Tests](https://github.com/JosephWoodall/uchi/actions/workflows/ci.yml/badge.svg)](https://github.com/JosephWoodall/uchi/actions/workflows/ci.yml)

Online, instance-based sequence predictor. Given any stream of discrete observations, it learns to predict what comes next — for any symbol type, in any domain — without assuming a fixed distribution, a known alphabet, or a stationary process.

The `uchi` package extends this core engine to tabular classification, regression, multivariate time series forecasting, anomaly detection, and generative modeling. All classes are sklearn-compatible.

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
