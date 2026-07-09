# Benchmarks — Capability & Trustworthiness

The FLUX (Proposer) + Uchi (Verifier) architecture is measured on two very
different things, and it is important not to conflate them.

!!! warning "Read this first"
    FLUX is a **small (~64M-parameter) from-scratch model**. On raw capability
    benchmarks it stays **near baseline** — MMLU near random (~25%), SWE-bench and
    ARC-Challenge low — and no amount of engineering changes that at this scale;
    it is a property of model size and training budget. We track these numbers as
    a **dashboard to watch the proposer improve**, not as a target it can climb.
    Uchi's actual value is *trustworthiness*: it grounds what FLUX proposes or
    honestly abstains. See [Training FLUX](training.md) for why.

## Capability dashboard (tracking, not targets)

1. **MMLU:** factual/reasoning accuracy across 57 subjects — tracks whether the
   proposer's knowledge is moving off the random baseline.
2. **SWE-bench:** bug-fix patch generation — tracks code capability.
3. **ARC-Challenge:** multi-concept elementary-science reasoning.

## Trustworthiness KPIs (SQuAD 2.0)

We still rigorously test Uchi's ability to abstain on unanswerable questions:

| Metric | Meaning |
|--------|---------|
| **coverage** | % of *answerable* questions it chooses to answer |
| **precision @ answered** | when it speaks, is it right |
| **honest-abstention** | % of *unanswerable* questions it correctly declines |
| **hallucination-rate** | % of emitted answers that are wrong |

These trustworthiness KPIs — not the capability dashboard — are the real measure
of the system. Uchi's gates are designed to keep the hallucination-rate low by
abstaining whenever FLUX's proposal cannot be grounded, trading coverage for
honesty. A better-trained FLUX raises coverage; the verifier keeps it honest
regardless.
