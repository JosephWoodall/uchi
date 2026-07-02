# Benchmarks — Capability & Trustworthiness

With the introduction of the FLUX (Proposer) + Uchi (Verifier) architecture in v0.3.0, we have reintroduced the holy trinity of capability benchmarks. FLUX provides the Out-Of-Distribution generalization, and Uchi guarantees the trustworthiness.

## Primary Benchmarks

We measure across three core axes:

1. **MMLU (Factual Recall):** Measures factual and reasoning accuracy across 57 academic subjects. FLUX provides the breadth; Uchi prevents confabulation.
2. **SWE-bench (Code Generation):** Tests bug-fix patch generation and general coding capability.
3. **ARC-Challenge (Reasoning):** Multi-concept reasoning on elementary science questions, tracking our general reasoning and reasoning chains capabilities.

## Trustworthiness KPIs (SQuAD 2.0)

We still rigorously test Uchi's ability to abstain on unanswerable questions:

| Metric | Meaning |
|--------|---------|
| **coverage** | % of *answerable* questions it chooses to answer |
| **precision @ answered** | when it speaks, is it right |
| **honest-abstention** | % of *unanswerable* questions it correctly declines |
| **hallucination-rate** | % of emitted answers that are wrong |

By using FLUX as the proposer, our precision and coverage have massively improved, while Uchi's strict mathematical gates keep the hallucination-rate suppressed.
