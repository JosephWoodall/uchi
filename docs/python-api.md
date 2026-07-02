# Python API — `Uchi`

`Uchi` provides a **simplified public API** accessible universally via this Python SDK, our TUI, and the REST API server.

```python
from uchi import Uchi
```

---

## 1. The Compounding Effect & Human-Readable I/O

The most important concept in the API is the compounding effect driven by **human-readable input and output**.

`ask()` **always returns a human-readable string.** `learn()` **always accepts a human-readable string.** This single design choice means the output of any analysis is immediately learnable knowledge for any other instance.

```python
# Three instances. Each learns from the previous one's output.
u1 = Uchi()
u1.learn(open("quarterly_report.txt").read())
forecast = u1.ask("/forecast", X=revenue_series, steps=4)

u2 = Uchi()
u2.learn(forecast)
strategy = u2.ask("What do these results imply for Q4 planning?")

u3 = Uchi()
u3.learn(strategy)
u3.ask("What should the board prioritise this quarter?")
```

## 2. FLUX Proposer & Uchi Verifier

When using `ask()`, the underlying pipeline automatically leverages **FLUX as the Proposer** to provide **OOD Generalization** and **General Reasoning Chains**, while Uchi acts as the reality-anchored verifier against the compounding brain.

```python
u = Uchi()
# FLUX proposes the factual answer; Uchi grounds it, or honestly abstains.
u.ask("What is the capital of France?")
```

## 3. Simplified Public API

Whether you are using this Python SDK, the Terminal UI (TUI), or the REST API, the interface remains identical.

```python
u.ingest("knowledge_base/")            # walk directory — txt/md/py/json/csv
u.ask("/classify",  X=X_train, y=y_train)
u.save("my_brain.uchi")
```

The exact same commands map 1:1 in the TUI (`/classify data.csv`) and REST endpoints.
