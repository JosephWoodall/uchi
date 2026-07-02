# Architecture

> **Python users:** you do not interact with the architecture directly. Use
> `from uchi import Uchi` and call `learn()` / `ask()`. See [Python API →](python-api.md).

---

Uchi is built on one core mathematical principle: **FLUX as the Proposer, Uchi as the Verifier.**

## The Two-Engine System

1. **The Proposer (FLUX):** Provides state-of-the-art Out-Of-Distribution (OOD) generalization, world knowledge, and generates complex, human-readable reasoning chains. 
2. **The Verifier (Uchi):** A reality-anchored verifier that intercepts FLUX's output. It grounds the claims against Uchi's compounding brain, checks semantic validity, and abstains when FLUX hallucinates.

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
