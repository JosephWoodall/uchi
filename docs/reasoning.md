# Reasoning — Verified Chains

In v0.3.0, Uchi achieves **general reasoning and reasoning chains** through a dual architecture: FLUX as the Proposer and Uchi as the Verifier. Three real, distinct mechanisms compose to make this work — not a single "reasoning engine," but three verifiers each suited to a different kind of claim.

## The Three Verification Mechanisms

1. **CoT think-trace (`uchi/proposer.py`, `FluxProposer.propose(..., think=True)`):**
   FLUX was distilled on real GSM8K reasoning traces. Priming generation with
   `<|think|>` instead of jumping straight to an answer elicits genuine
   multi-step narration ("First find X... Then Y... The answer is Z") instead
   of a bare answer. This is the main path in
   [Generate-and-Ground](generate-and-ground.md)'s candidate generation.
2. **Empirical Synthesis Loop (`uchi/generate_and_ground.py`, `answer()`):**
   when text grounding fails, FLUX writes a standalone Python function with
   `assert` statements proving its own logic, and a `REPLOracle` actually
   *executes* it. The result only reaches the user if the code runs and the
   asserts pass — proof by execution, not by claim. On failure, the traceback
   is fed back to FLUX for a bounded number of reflection attempts.
3. **Cross-Examination / Devil's Advocate (`uchi/generate_and_ground.py`,
   `_candidates()`):** after a candidate passes the FactCheck Oracle, a second
   FLUX pass is prompted to attack its own logic. If a flaw is found, the
   critique is fed back and the candidate is regenerated before it reaches the
   user.

For genuinely decomposable questions, the **Swarm Synthesizer**
(`uchi/swarm.py`) breaks a complex question into independent sub-questions,
solves each through the full loop above in parallel, and synthesizes a final
answer only from the verified sub-answers.

## The Loop

1. **Decompose & Propose:** FLUX takes a complex goal and proposes a human-readable chain of thought, breaking the problem into sub-steps. This leverages its OOD generalization — trained reasoning applied to questions it has never seen verbatim.
2. **Execute & Verify:** each step is checked by the mechanism that fits its shape — code by REPL execution, factual claims by the FactCheck Oracle against the brain, logical soundness by cross-examination.
3. **Compose:** combine the verified results.
4. **Abstain with Provenance:** the moment a step cannot be verified, Uchi aborts and names the failing step in human-readable output instead of guessing.

## The Honest Claim

The guarantee is about **trust, not raw power**. FLUX (a small ~64M model)
proposes chains of thought; Uchi verifies each step and abstains — naming the
failing step — the moment one cannot be verified. So the system never asserts an
unverified conclusion. It does **not** mean FLUX solves hard reasoning
benchmarks: at this scale its proposals are often wrong (numbers in a GSM8K-style
trace are frequently garbled), and the honest outcome is then an abstention
rather than a confident error. A stronger proposer raises how often a chain
completes; the verifier keeps every completed chain grounded.
