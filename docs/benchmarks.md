# Benchmarks — Trustworthiness

Uchi is evaluated on **what it is for**: being trustworthy. The right question is
not "how smart is it" but **"when it speaks, is it right; does it know what it
doesn't know; does it ever confabulate."** Accuracy benchmarks like MMLU / ARC /
SWE-bench are *retired* — they measure LLM-style reasoning, an axis a no-LLM system
is at random on and which is not the product's purpose.

## Headline: SQuAD 2.0 (answerable + unanswerable)

`python benchmarks/trustworthiness.py --sample 800`

| Metric | Meaning |
|--------|---------|
| **coverage** | % of *answerable* questions it chooses to answer |
| **precision @ answered** | when it speaks, is it right |
| **honest-abstention** | % of *unanswerable* questions it correctly declines |
| **hallucination-rate** | % of emitted answers that are wrong — the number that matters |

Raising the answerability threshold trades coverage for caution:

| answerability threshold | coverage | precision@answered | honest-abstention | hallucination |
|---|---|---|---|---|
| 0.0 (grounding only) | 99% | 56% | 2% | 73% |
| 0.6 (default) | ~85% | 58% | ~35% | ~69% |
| 0.95 (cautious) | 57% | 57% | 53% | 69% |

**Read these honestly:**

- **Precision-on-answered is ~57%** — even on a genuinely answerable question the
  system finds the right *topic* but not always the exact answer-bearing passage,
  and the from-scratch decoder is weak. This is the real ceiling: **Uchi is not yet
  trustworthy on hard open-domain QA** — it hallucinates on a meaningful fraction of
  what it answers.
- The abstention gates (fact-check oracle + answerability classifier) raise
  honest-abstention substantially but **cannot lower hallucination below the
  generation-precision floor.** Better retrieval (a trained dense retriever) and a
  stronger generator are the primary roadmap items.
- Uchi **is** reliably honest on *clearly-unknown* queries (it abstains) and on
  social turns (nothing to verify).

## Reasoning (separate skill)

The ARC-AGI program-synthesis reasoner (`experiments/arc_dsl.py`) demonstrates
*provable* multi-step reasoning: it searches for a program that reproduces the
demonstration examples and abstains if none is found. It reaches single-digit
solve-rate on ARC-AGI eval at ~100% precision (a found program is verified). Honest
scope: grid-transformation tasks, not general QA — it is a proof of the
verify-guided reasoning mechanism, not a general reasoner.

## Structural guarantee

- **0% catastrophic forgetting** — the recall trie is append-only. New knowledge
  never overwrites old anchors.

## What was retired

MMLU accuracy, ARC-Challenge, SWE-bench composite, and the sequence-prediction
corpora benches are no longer the primary scorecard. They measured the wrong axis
for a trustworthy, no-LLM assistant.
