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
2. **The Verifier (Uchi):** A layered, additive-only cascade that intercepts
   FLUX's output before it ever reaches the user. Each layer can only turn a
   pass into a reject — none can override a rejection into an acceptance, and
   none can accept a claim on its own authority. See "The Verification
   Cascade" below for what each layer actually checks.

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

## The Verification Cascade

A candidate claim from FLUX passes through these layers, in order. Any layer
can veto; none can grant an acceptance the layers before it didn't already
give. This asymmetry is deliberate and load-bearing: a false positive in a
later layer costs an unnecessary abstention, not a wrong answer reaching the
user.

1. **Deterministic word-overlap (`uchi/oracle.py`, `FactCheckOracle`)** — the
   floor. Checks whether the claim's salient terms are actually present in
   retrieved evidence. Cheap, fully auditable, catches claims fabricated out
   of thin air.
2. **Entailment classifier + OOD gate (`uchi/flux/verifier_model.py`)** — a
   separately-trained-from-scratch classifier (its own embedding table, never
   shares weights or gradients with FLUX) judging whether the claim
   contradicts the evidence. Catches semantic contradictions word-overlap
   can't see — a claim can share every salient word with the evidence and
   still assert the wrong number or the opposite fact. Gated by a Mahalanobis
   out-of-distribution check on the classifier's own latent space: if the
   input is far from anything the classifier was trained on, its verdict is
   discarded as "no opinion" rather than trusted, and the earlier layer's
   decision stands instead.
3. **Numeric plausibility (`uchi/numeric_plausibility.py`)** — a
   median/MAD statistical outlier check on claimed numbers against real
   values seen in ingested passages. Opt-in (`fit_numeric_plausibility_checker()`),
   since fitting scans the whole index.
4. **Relational transitivity (`uchi/relational_reasoning.py`)** — a
   deterministic check for simple comparative relations (taller/shorter,
   older/younger, before/after, etc.), building a transitive closure from
   evidence and vetoing conclusions that contradict it — including chains
   spanning more than one stated fact. Explicitly narrow in scope: this is
   not general logical-inference verification.

If nothing survives this cascade, Uchi abstains rather than guess.

## Dynamic Compute Allocation

Not every question deserves the same amount of work. Two mechanisms decide
how much effort a given question gets, both additive-only — they can only
increase scrutiny, never decrease it below a safe baseline:

- **Dynamic-N self-consistency voting (`uchi/task_config_cache.py`)** — the
  number of candidate answers FLUX generates per question isn't fixed. A
  baseline comes from a cheap structural-complexity heuristic
  (`iq_router.estimate_complexity`); `TaskConfigCache`, backed by the same
  credibility-weighted trie used for exact recall, refines this by recalling
  how many votes a *structurally similar* question needed last time —
  keyed on shape (comparison, multi-part, enumeration, length), not exact
  text, so it generalizes across topics.
- **Proprioception (`uchi/proprioception.py`, experimental)** — FLUX's own
  sense of whether a question's topic/shape is familiar, checked *before*
  generation via a Mahalanobis distance over FLUX's own hidden state for the
  raw question. An "unfamiliar" verdict can only raise the vote count, never
  lower it. This is **not** a substitute for the verification cascade above —
  it never sees a generated claim, only the question beforehand, so it
  structurally cannot check factual correctness. It answers "is this topic
  familiar," not "is this specific claim true."

## Multi-Step Reasoning

For questions that bundle multiple independent sub-problems, `SwarmSynthesizer`
decomposes the question (gated by the same complexity heuristic above, so
atomic questions skip decomposition entirely) into sub-questions, answers each
one independently through the full pipeline described above, and aggregates
the results. For reasoning that benefits from working through steps before
answering, FLUX's CoT-trained generation path (`think=True`) produces a
reasoning trace ahead of the final answer — the trace itself isn't
independently re-verified step-by-step; the same verification cascade checks
the final claim against evidence, same as any other answer.

## Persistence & Compounding
`ask()` returns a human-readable string; `learn()` accepts one. `learn()` feeds the retrieval index live, so knowledge added to one instance grounds another, allowing a continuous compounding effect.
