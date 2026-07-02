![Uchi Logo](docs/logo.png)

[![PyPI version](https://img.shields.io/pypi/v/uchi_python.svg)](https://pypi.org/project/uchi_python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/uchi_python.svg)](https://pypi.org/project/uchi_python/)
[![Tests](https://github.com/JosephWoodall/uchi/actions/workflows/ci.yml/badge.svg)](https://github.com/JosephWoodall/uchi/actions/workflows/ci.yml)

## Uchi — The Reality-Anchored Verifier (v0.3.0)

Uchi introduces a breakthrough architecture: **FLUX as the Proposer, Uchi as the Verifier.** 
By pairing the Out-Of-Distribution (OOD) generalization and reasoning chains of FLUX with the strict mathematical grounding of Uchi, we achieve the ultimate balance of capability and trustworthiness.

### 5 Non-Negotiables for v0.3.0
1. **Compounding Effect:** Knowledge is persistently stored and compounding across instances.
2. **Simplified Public API:** Accessible universally via our SDK, TUI, and REST API.
3. **General Reasoning & Reasoning Chains:** FLUX proposes complex chains of thought; Uchi verifies every link.
4. **Human-Readable I/O:** Clear, transparent, and interpretable input and output.
5. **OOD Generalization:** FLUX provides the raw LLM capability to tackle Out-Of-Distribution tasks.

```python
from uchi import Uchi

u = Uchi()
u.learn("The Eiffel Tower is a wrought-iron lattice tower in Paris, France.")

u.ask("What is the Eiffel Tower?")
# FLUX proposes answer -> Uchi verifies -> Output: "The Eiffel Tower is a wrought-iron lattice tower in Paris, France."
```

### Trustworthiness Meets Capability

Uchi verifies factual claims and chains of logic against its semantic memory. If FLUX proposes an answer that cannot be grounded, Uchi intercepts it and honestly abstains. Because of this powerful pairing, **accuracy benchmarks (MMLU, SWE-bench, and ARC-Challenge) are back.** We rely on FLUX to propose the right answer, and Uchi to prove it.

## Simplified Public API (SDK, TUI, & REST)

Uchi v0.3.0 standardizes all interactions across three human-readable interfaces. Whether you are scripting, using the terminal, or building a web app, the commands are identical.

### 1. Python SDK
```python
from uchi import Uchi
u = Uchi()

# Ingest directories or files
u.ingest("docs/").ingest("data.csv")

# Analytical skills
u.ask("/classify", X=X_train, y=y_train)
```

### 2. Terminal UI (TUI)
Run Uchi directly from your terminal with a beautiful interface:
```bash
uchi tui

# Inside the TUI, use the exact same commands:
> /classify data.csv --label target_col
```

### 3. REST API
Host Uchi as a backend service:
```bash
uchi serve --port 8000
```
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "/classify data.csv"}'
```

## Install

```bash
pip install uchi_python
```
See `docs/` for architecture details and the full API reference.
