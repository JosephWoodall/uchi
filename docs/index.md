# Uchi Documentation

> A powerful pairing: **FLUX as the Proposer, Uchi as the Verifier.**
> Uchi grounds factual answers, reasons in verified chains, offers a simplified API (SDK, TUI, REST), and delivers compounding knowledge with OOD generalization.

```python
from uchi import Uchi

u = Uchi()
u.learn("The Eiffel Tower is a wrought-iron lattice tower in Paris, France.")
u.ask("What is the Eiffel Tower?")     # FLUX proposes, Uchi verifies.
```

## Start here

- **[Python API](python-api.md)** — the `Uchi` class: `learn()`, `ask()`, `ingest()`, `save()`.
- **[Architecture](architecture.md)** — how FLUX and Uchi work together.
- **[Generate-and-Ground](generate-and-ground.md)** — the factual answering pipeline.
- **[Reasoning](reasoning.md)** — verified multi-step reasoning chains.
- **[Benchmarks](benchmarks.md)** — MMLU, SWE-bench, and ARC-Challenge.

## 0.3.0 Non-Negotiables

1. **Compounding Effect**
2. **Simplified Public API (SDK, TUI, REST API)**
3. **General Reasoning & Reasoning Chains**
4. **Human-Readable Input & Output**
5. **OOD Generalization**
