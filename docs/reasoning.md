# Reasoning — Verified Chains

In v0.3.0, Uchi achieves **general reasoning and reasoning chains** through a dual architecture: FLUX as the Proposer and Uchi as the Verifier.

## The Loop

1. **Decompose & Propose:** FLUX takes a complex goal and proposes a human-readable chain of thought, breaking the problem into sub-steps. This leverages its unparalleled OOD generalization.
2. **Execute & Verify:** Uchi executes each step in the chain with a self-verifying operator:
   - **math** — a symbolic evaluator (`sympy`); verifiable by construction.
   - **code** — a REPL-checked code operator.
   - **factual** — Generate-and-Ground (grounded against Uchi's compounding brain, or abstain).
3. **Compose:** Combine the verified results.
4. **Abstain with Provenance:** The moment a step in FLUX's proposed chain cannot be verified, Uchi aborts and names the exact failing step in human-readable output instead of guessing.

## The Honest Claim

The guarantee is about **trust, not raw power**. FLUX (a small ~116M model)
proposes chains of thought; Uchi verifies each step and abstains — naming the
failing step — the moment one cannot be verified. So the system never asserts an
unverified conclusion. It does **not** mean FLUX solves hard reasoning
benchmarks: at this scale its proposals are often wrong, and the honest outcome
is then an abstention rather than a confident error. A stronger proposer raises
how often a chain completes; the verifier keeps every completed chain grounded.
