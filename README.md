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

## Skills

Analytical capabilities and reasoning operators are exposed natively:

```python
u.ask("/classify",  X=X_train, y=y_train)
u.ask("/regress",   X=X_train, y=y_train)
```

## Install

```bash
pip install uchi_python
```
See `docs/` for architecture details and the full API reference.
