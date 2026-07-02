# Generate-and-Ground

The factual answering pipeline — Uchi's primary `ask()` path. It exists to make one
guarantee tractable: **answer from grounded evidence, or abstain — never
confabulate.**

```
question
   ├─ retrieve evidence from the brain          (retrieval.SemanticIndex)
   ├─ answerability gate: does the evidence      (answerability.AnswerabilityChecker)
   │   actually answer this question?  ── no ──► ABSTAIN
   ├─ generate a candidate answer               (decoder.NeuralDecoder, or extractive)
   ├─ fact-check: is it supported by evidence?  (oracle.FactCheckOracle) ── no ──► ABSTAIN
   └─ emit the grounded answer
```

## Components

- **`retrieval.SemanticIndex`** — passages embedded with skip-gram word vectors,
  retrieved by hybrid lexical+semantic ranking. Built into the brain during
  ingestion and fed live by `learn()`.
- **`answerability.AnswerabilityChecker`** — a small from-scratch classifier
  (trained on SQuAD 2.0) that estimates whether the retrieved evidence answers the
  question. Catches *subtle* unanswerability where the topic is present but the
  answer is not.
- **`decoder.NeuralDecoder`** — a small BiGRU seq2seq trained from scratch,
  conditioned on question + evidence. Rough but grounded; on low confidence the loop
  falls back to an extractive answer (a real evidence sentence).
- **`oracle.FactCheckOracle`** — verifies the candidate's salient terms are
  supported by the evidence. It is a verifier, not a value critic: it checks
  grounding, not quality.

## Why generate *and* verify

Retrieval alone can't phrase an answer; a generator alone hallucinates. Uchi
generates for reach and verifies for honesty. The generator may fabricate freely —
the oracle and answerability gate are the safety net, so a fabrication becomes an
abstention instead of a lie.

## Honest limitations

The oracle checks *token support*, not full entailment, and retrieval finds the
right topic more reliably than the exact answer-bearing passage. On hard
open-domain QA this yields ~57% precision on answered questions — see
[Benchmarks](benchmarks.md). The gates make Uchi *cautious*, not *omniscient*: it
abstains a lot, and when it answers a hard question it is right roughly half the
time. The fix is better retrieval + generation, not a stricter gate.

## Tuning

`GenerateAndGround(min_sim=…, min_answerable=…)` trades coverage for caution. A
higher `min_answerable` abstains more (fewer wrong answers reach the user, at the
cost of answering fewer questions).
