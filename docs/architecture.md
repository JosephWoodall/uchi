# Architecture

> **Python users:** you do not interact with the architecture directly. Use
> `from uchi import Uchi` and call `learn()` / `ask()`. See [Python API →](python-api.md).

---

Uchi is built on one core mathematical principle: **FLUX as the Proposer, Uchi as the Verifier.**

## The Two-Engine System

1. **The Proposer (FLUX):** A small (~116M) from-scratch SSM/attention model that
   generates candidate answers and human-readable reasoning chains. It supplies
   out-of-distribution *generalization attempts* — bounded by its scale — that a
   pure retrieval system cannot. See [Training FLUX →](training.md).
2. **The Verifier (Uchi):** A reality-anchored verifier that intercepts FLUX's output. It grounds the claims against Uchi's compounding brain, checks semantic validity, and abstains when FLUX hallucinates.

The Proposer is deliberately **swappable** (`uchi/proposer.py`): a better model
raises the ceiling on capability, while the verifier keeps output honest
regardless of how good or bad the proposer is. When no trained FLUX checkpoint is
present, the proposer degrades gracefully and the verifier falls back to grounded
extraction / abstention.

## The Three Lanes behind `ask()`

Every natural-language message is classified by `intent_router` and routed:

```
ask(str) ─► router ── social ──────────────────►  ConversationEngine  (free chit-chat)
                 ├─────────── factual ──────────►  FLUX Proposer ─► Uchi Verifier
                 └─────────── skill ───────────►  SkillRegistry  (code, classification)
```

## Reasoning — Verified Steps
For complex tasks, FLUX proposes a multi-step reasoning chain. The `ReasoningEngine` executes each step using a self-verifying operator. Every emitted conclusion is a chain of mathematically verified steps.

## Persistence & Compounding
`ask()` returns a human-readable string; `learn()` accepts one. `learn()` feeds the retrieval index live, so knowledge added to one instance grounds another, allowing a continuous compounding effect.
