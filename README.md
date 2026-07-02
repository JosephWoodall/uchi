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

Uchi verifies factual claims and chains of logic against its semantic memory. If FLUX proposes an answer that cannot be grounded, Uchi intercepts it and honestly abstains. We rely on FLUX to propose, and Uchi to prove.

> **On benchmarks, honestly:** FLUX is a small (~116M) from-scratch model. MMLU,
> SWE-bench, and ARC-Challenge are tracked as a **dashboard** to watch the proposer
> improve — at this scale they stay near baseline, and that is expected. The point
> of the pairing is *trustworthiness* (grounded answers or honest abstention), not
> a leaderboard score. See [`docs/training.md`](docs/training.md) for how FLUX is
> trained and what the final `flux_best.pt` artifact is.

## How It Connects

Training produces **one artifact**, and every interface loads it through **one place** — `Uchi.__init__`. The SDK, TUI, and REST server all construct the same `Uchi` object, so training FLUX once makes it live everywhere automatically. If no checkpoint exists yet, the proposer degrades gracefully and the verifier falls back to grounded extraction / honest abstention — Uchi still runs.

```
scripts/train_all.sh ─► uchi/flux/checkpoints/flux_best.pt   ◄── the artifact
                                     │
        Uchi.__init__ picks first that exists:
        flux_best → qat_best → cot_best → sft_best   (else None)
                                     │
        FluxProposer.load(ckpt) → build_generate_fn(ckpt)
           loads the HybridTSSM model, returns generate_fn(prompt) -> str
                                     │
        self.proposer ──► GenerateAndGround(index, oracle, proposer)
                                     │              (retrieve → propose → verify → emit/abstain)
                               Uchi.ask(q)
             ┌───────────────────────┼───────────────────────┐
            SDK                      TUI                      REST
   from uchi import Uchi     `uchi tui` → UchiApp     `uchi serve` → api_server
   u = Uchi(); u.ask(q)      → router = Uchi()        → _router = Uchi(); POST /ask
```

See [`docs/training.md`](docs/training.md) for the training pipeline and the `flux_best.pt` artifact.

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
