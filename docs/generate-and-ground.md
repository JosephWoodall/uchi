# Generate-and-Ground

The factual answering pipeline — Uchi's primary `ask()` path. It exists to enforce the core v0.3.0 guarantee: **FLUX Proposer + Uchi Verifier.**

```
question
   ├─ FLUX proposes answer and reasoning chain  (OOD Generalization)
   ├─ retrieve evidence from the brain          (retrieval.SemanticIndex)
   ├─ answerability gate: does the evidence      (answerability.AnswerabilityChecker)
   │   actually answer this question?  ── no ──► ABSTAIN
   ├─ fact-check: is FLUX's proposal supported? (oracle.FactCheckOracle) ── no ──► ABSTAIN
   └─ emit the grounded answer in human-readable output
```

## Why generate *and* verify
Retrieval alone can't phrase an answer; an LLM alone hallucinates. By using FLUX to propose the answer, we achieve massive **Out-Of-Distribution (OOD) generalization**. By using Uchi to verify it against the **compounding** semantic index, we guarantee trustworthiness. If FLUX fabricates freely, the oracle intercepts it, and the fabrication becomes an abstention instead of a lie.

## Tuning
`GenerateAndGround(min_sim=…, min_answerable=…)` trades coverage for caution. A higher `min_answerable` abstains more.
