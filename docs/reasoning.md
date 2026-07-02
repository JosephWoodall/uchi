# Reasoning — Verified Steps

Uchi does not reason like an LLM (implicitly, in weights). It reasons like a
scientist: **decompose a goal into steps, execute each step with an operator that
can VERIFY its own output, compose the survivors, and abstain — naming the exact
step — when it can't.**

Every emitted conclusion is a chain of *verified* steps. That is a claim no LLM can
make: LLMs reason fluently but cannot check themselves. Uchi reasons *by
verification*.

```python
u.ask("What is 12 times 15; then add 8")
# → 188   (1. 12 × 15 → 180 [math]   2. 180 + 8 → 188 [math])

u.ask("What temperature does water boil at; then subtract 20")
# → 80    (1. …boil at → 100 [factual, grounded]   2. 100 − 20 → 80 [math])

u.ask("What is the population of Atlantis; then double it")
# → "I can establish the first 0 step(s) but cannot verify this one, so I won't
#    guess:  ✗ What is the population of Atlantis"
```

Multi-step questions (connected by `;`, `then`, or numbered) route to
`reasoning.ReasoningEngine`; single-step questions go straight to
[Generate-and-Ground](generate-and-ground.md), so reasoning never makes simple
queries worse.

## The loop

1. **Decompose** the goal into sub-steps.
2. **Execute** each step with a self-verifying operator:
   - **math** — a symbolic evaluator (`sympy`); verifiable by construction.
   - **code** — a REPL-checked code operator.
   - **factual** — Generate-and-Ground (grounded, or abstain).
3. **Compose** verified results (a step can reference the previous one).
4. **Abstain with provenance** the moment a step can't be verified — it names the
   failing step instead of guessing a conclusion.

## What's solid vs. the open problem

- **Solid — the verified-step spine.** Route each step to an operator that checks
  itself, keep only what verifies, abstain otherwise. This is the differentiator.
- **Extensible operators.** Add algebra/units oracles, unit-test oracles for code,
  or the ARC-AGI DSL solver — each a self-verifying reasoning primitive.
- **The hard open problem — the planner.** Turning arbitrary language into a
  verifiable plan without an LLM is genuinely hard; the current decomposer is a
  heuristic (connectives, numbered steps, follow-up operations). A small trained
  planner and MCTS-over-plans (using the verify/fail signal as reward) are the next
  steps.

## The honest claim

Not "Uchi reasons like a brain," but: **"Uchi reasons in verified steps — it
decomposes a problem, checks each step against reality, composes the survivors, and
tells you exactly which step it couldn't establish instead of guessing."** That is
true, demonstrable via the trace, and genuinely differentiated.
