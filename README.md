![Uchi Logo](docs/logo.png)

[![PyPI version](https://img.shields.io/pypi/v/uchi_python.svg)](https://pypi.org/project/uchi_python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/uchi_python.svg)](https://pypi.org/project/uchi_python/)
[![Tests](https://github.com/JosephWoodall/uchi/actions/workflows/ci.yml/badge.svg)](https://github.com/JosephWoodall/uchi/actions/workflows/ci.yml)

## Uchi — a grounded assistant that would rather say "I don't know" than make something up

Uchi is a from-scratch, **no-LLM** assistant built around a single principle:
**generate an answer, then verify it against what the brain actually knows — and
abstain when it can't.** It answers factual questions from grounded evidence,
talks to you conversationally, runs analytical skills on your data, and — by
design — never *tries* to confabulate.

```python
from uchi import Uchi

u = Uchi()
u.learn("The Eiffel Tower is a wrought-iron lattice tower in Paris, France.")

u.ask("What is the Eiffel Tower?")
# → "The Eiffel Tower is a wrought-iron lattice tower in Paris, France."

u.ask("Who was the 14th president of Mars?")
# → "I don't have grounded knowledge to answer that."   (it abstains — it doesn't know)

u.ask("hi there!")
# → a conversational reply (no facts asserted, so nothing to verify)
```

`ask()` always returns a string; `learn()` always accepts one. That's the
**compounding contract** — the output of one instance is directly learnable by
another, no schema or glue:

```python
report = u.ask("/classify", X=X_train, y=churn_labels)   # analytical skill → string
u2 = Uchi(); u2.learn(report)                             # analysis becomes knowledge
u2.ask("What accuracy did we get?")                       # grounded in that report
```

---

## How it works — three lanes behind `ask()`

Every natural-language message is routed into one of three lanes:

| Lane | Handler | Behaviour |
|------|---------|-----------|
| **factual** | **Generate-and-Ground** | retrieve evidence → generate a candidate → **fact-check it** → emit if grounded, else **abstain** |
| **social** | conversation engine | free-generated chit-chat — asserts no facts, so it needs no verification |
| **skill** | `SkillRegistry` | `/classify`, `/regress`, `/forecast`, `/anomaly`, code, … |

The key design decision: **only the factual lane is verified.** Social replies
have no ground truth to violate, so a small dialogue model generates them freely —
that's how Uchi has a personality without weakening its honesty on facts.

### The factual lane in detail

```
question → retrieve relevant knowledge (semantic index over the brain)
         → generate a candidate answer (small from-scratch decoder, or extractive)
         → FACT-CHECK: is the answer supported by the evidence, and does the
           evidence actually answer the question? (answerability gate)
         → emit if grounded, otherwise ABSTAIN — never confabulate
```

Generalisation comes from generating over retrieved knowledge; honesty comes from
verifying before speaking.

---

## Trustworthiness — measured, not claimed

Uchi is evaluated on **what it is for** — being trustworthy — not on LLM-style raw
reasoning. The headline benchmark is **SQuAD 2.0** (answerable + *unanswerable*
questions), measured with `benchmarks/trustworthiness.py`:

- **coverage** — % of answerable questions it chooses to answer
- **precision @ answered** — when it speaks, is it right
- **honest-abstention** — % of unanswerable questions it correctly declines
- **hallucination-rate** — % of emitted answers that are wrong (the number that matters)

The abstention threshold trades coverage for caution:

| answerability threshold | coverage | precision@answered | honest-abstention | hallucination |
|---|---|---|---|---|
| 0.0 (grounding only) | 99% | 56% | 2% | 73% |
| 0.6 (default) | ~85% | 58% | ~35% | ~69% |
| 0.95 (cautious) | 57% | 57% | 53% | 69% |

**Honest limitations (read these):**
- The current **retrieval + generation precision is ~57%** — even on a genuinely
  answerable question, the system finds the right *topic* but not always the exact
  answer-bearing passage, and the from-scratch decoder is weak. This is the real
  ceiling, and it means **Uchi is not yet trustworthy on hard open-domain QA** — it
  hallucinates on a meaningful fraction of what it answers.
- It **is** reliably honest on *clearly-unknown* queries (it abstains) and on
  *social* turns (nothing to verify).
- The conversational and answer decoders are small and trained from scratch (no
  LLM) — grammatical but rough, not fluent.

Improving retrieval (a trained dense retriever) and the generator is the primary
roadmap item; the architecture is built so those upgrades slot in behind a stable
interface.

---

## Skills

Analytical capabilities are exposed as skills, invokable directly:

```python
u.ask("/classify",  X=X_train, y=y_train)
u.ask("/regress",   X=X_train, y=y_train)
u.ask("/anomaly",   X=sensor_matrix)
u.ask("/forecast",  X=time_series, steps=20)
```

A separate, self-verifying **program-synthesis reasoner** (grid-transformation
tasks, ARC-AGI style) demonstrates provable multi-step reasoning: it searches for a
program that reproduces the demonstration examples, applies it, and abstains if
none is found.

## Install

```bash
pip install uchi_python
```

## Ingesting knowledge

```python
u = Uchi().ingest("docs/").ingest("data.csv").ingest("report.pdf")
```

Growing the brain expands what Uchi can *answer* (more retrievable knowledge → less
abstention). It does not, by itself, improve reasoning — that is a separate axis.

## What Uchi is (and isn't)

- **Is:** a no-LLM, from-scratch assistant that grounds factual answers, abstains
  honestly, converses, runs analytical skills, and compounds knowledge across
  instances.
- **Isn't:** an LLM. It is smaller, deterministic where possible, auditable, and
  needs no GPU at inference — and, today, is materially less capable than a large
  model on open-domain QA. Its bet is **trustworthiness over raw capability**.

See `docs/` for architecture details and the full API reference.
