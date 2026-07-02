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

Because FLUX proposes the chain, Uchi can now solve extremely complex reasoning tasks (like ARC-Challenge) that were previously impossible for a purely semantic system, while maintaining 100% verifiability at every node.
