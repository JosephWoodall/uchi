# Uchi Documentation

> A from-scratch, **no-LLM** assistant that grounds factual answers, reasons in
> verified steps, converses, and **abstains instead of confabulating**. One import.
> Knowledge compounds across instances.

```python
from uchi import Uchi

u = Uchi()
u.learn("The Eiffel Tower is a wrought-iron lattice tower in Paris, France.")
u.ask("What is the Eiffel Tower?")     # → grounded answer
u.ask("Who was the 14th president of Mars?")   # → "I don't have grounded knowledge to answer that."
```

## Start here

- **[Python API](python-api.md)** — the `Uchi` class: `learn()`, `ask()`, `ingest()`, `save()`.
- **[Architecture](architecture.md)** — how `ask()` routes and how answers get grounded.
- **[Generate-and-Ground](generate-and-ground.md)** — the factual answering pipeline.
- **[Reasoning](reasoning.md)** — verified multi-step reasoning.
- **[Benchmarks](benchmarks.md)** — trustworthiness KPIs (measured, not claimed).

## What Uchi is

Every natural-language message routes into one of three lanes:

| Lane | Behaviour |
|------|-----------|
| **factual** | retrieve evidence → generate → **fact-check** → answer or **abstain** |
| **social** | free-generated chit-chat (asserts no facts → nothing to verify) |
| **skill** | analytical commands (`/classify`, `/forecast`, …) and code |

The single design principle: **a fallible proposer plus a reality-anchored
verifier.** Uchi generalises by generating over retrieved knowledge, and stays
honest by verifying before it speaks — on factual answers and on each reasoning
step alike.

## The compounding contract

`ask()` always returns a string; `learn()` always accepts one. The output of one
instance is directly learnable by another — no schema, no glue:

```python
report = u.ask("/classify", X=X_train, y=labels)   # analytical skill → string
u2 = Uchi(); u2.learn(report)                       # analysis becomes knowledge
u2.ask("What accuracy did we get?")                 # grounded in that report
```

## Honest status

Uchi is reliably honest on clearly-unknown queries (it abstains) and on social
turns. It is **not yet trustworthy on hard open-domain QA** — retrieval and
generation precision (~57%) is the current ceiling and the primary roadmap item.
It is smaller, deterministic where possible, auditable, and needs no GPU — its bet
is **trustworthiness over raw capability**. See [Benchmarks](benchmarks.md) for the
real numbers.
