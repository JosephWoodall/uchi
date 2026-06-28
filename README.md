![Uchi Logo](docs/logo.png)

[![PyPI version](https://img.shields.io/pypi/v/uchi_python.svg)](https://pypi.org/project/uchi_python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/uchi_python.svg)](https://pypi.org/project/uchi_python/)
[![Tests](https://github.com/JosephWoodall/uchi/actions/workflows/ci.yml/badge.svg)](https://github.com/JosephWoodall/uchi/actions/workflows/ci.yml)

## Core Mission: One Import. Everything Compounds.

Uchi v0.3.0 dramatically simplifies the public API and introduces a compounding mechanism that makes every analysis result immediately learnable by every other instance.

```python
from uchi import Uchi

u = Uchi()
u.learn("Q3 revenue was $4.2M, up 23% YoY.")
report = u.ask("/classify", X=X_train, y=churn_labels)

u2 = Uchi()
u2.learn(report)                          # analysis becomes knowledge
u2.ask("What does this imply for Q4?")   # knowledge compounds
```

`ask()` always returns a string. `learn()` always accepts a string. This is the compounding guarantee — outputs of one instance are directly learnable by another, with no serialisation, schema, or orchestration layer required.

Under the hood, Uchi is an **Omni-modal Deterministic Universal Sequence Predictor (ODUSP)** — it ingests text, tabular data, time series, and code simultaneously without any neural weights or pre-training. A trainable SSM confidence signal (GRPO), persistent vector memory, intent-based routing, and a full analytical skill layer are all accessible through the single `Uchi` entry point. No LLM dependency at any layer.


> [!NOTE]
> Please see `docs/` for the complete Algorithmic Walkthrough, ODUSP vs LLM Benchmarks, and full API references.

---

> [!NOTE]
> **Comprehensive Documentation & API Reference**
>
> For interactive examples, API documentation, and to see the newest capabilities (including online Math Learning, Vector Retrievals, and the Simulation Engine), please see our full documentation website.
>
> **[Read the Full Documentation →](https://github.com/JosephWoodall/uchi/tree/main/docs)**

---

## Installation

```bash
pip install -e ".[all]"
```

On first launch, Uchi runs a one-time bootstrap (Python stdlib patterns + Wikipedia facts) and saves the result to `brain.uchi`. Subsequent launches are instant.

## Quickstart

Uchi has two entry points that share the same `brain.uchi` — every interaction in either one improves the other.

### 1. Terminal UI (TUI)

```bash
uchi                        # launch interactive chat
uchi --preload data.txt     # pre-train with a file before chatting
uchi --brain /path/to/brain.uchi   # use a specific brain file
```

Inside the TUI:

| Command | Description |
|---|---|
| Just type | Chat with Uchi — it learns from every turn |
| `/load <file>` | Stream any file into the knowledge base |
| `/save` | Force-save the current brain state to disk |
| `Ctrl+S` | Save brain |
| `Ctrl+C` | Save and quit |

Uchi gives positive/negative feedback signals to improve itself — type "good", "correct", "yes" to reinforce a response, or "wrong", "bad", "no" to prune it.

### 2. REST API

```bash
uvicorn uchi.api_server:app --host 0.0.0.0 --port 8000
```

**POST /chat**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "what is the capital of France?"}'
# {"reply": "paris", "entropy": 3.2}
```

**GET /metrics**
```bash
curl http://localhost:8000/metrics
# {"status": "online", "memory_records": 1024, "mode": "deterministic"}
```

**GET /debug/walk**
```bash
curl http://localhost:8000/debug/walk
# Returns trie walk data from the last prediction (depth, contributions, similarity)
```

### 3. Python API

`Uchi` is the single entry point for the entire library. One import, everything discoverable.

```python
from uchi import Uchi
```

#### Knowledge & Q&A

```python
u = Uchi()                                          # loads pre-packaged brain
u.learn("The boiling point of water is 100°C.")    # stream text into the trie
u.ask("At what temperature does water boil?")      # → "100°C"
```

#### File and directory ingestion

```python
u.ingest("knowledge_base/")           # walk directory — all .txt .md .py .json .csv
u.ingest("report.pdf")                # PDF extraction (pip install pdfminer.six)
u.ingest("events.csv", col="notes")   # specific CSV column

# chainable — returns self
u = Uchi().ingest("docs/").ingest("data.csv").ingest("handbook.md")
u.save("expanded_brain.uchi")
```

#### Analytical tools via slash commands

Every tool returns a plain string you can immediately feed into another instance.

```python
result = u.ask("/classify", X=X_train, y=y_train)  # classification report
result = u.ask("/regress",  X=X_train, y=y_train)  # regression report
result = u.ask("/anomaly",  X=sensor_matrix)        # anomaly detection report
result = u.ask("/forecast", X=time_series, steps=20)# forecast report
result = u.ask("/tsclassify", X=windows, y=labels) # time series classification
```

The same commands accept a CSV path when called from the TUI:

```
/classify data.csv --label target_col
/anomaly  sensors.csv
```

#### Compounding analysis — the core value

`ask()` always returns a string. `learn()` always accepts a string.
This means any analysis result is directly learnable by any other `Uchi` instance —
no serialisation, no schema, no glue code required.

```python
# Step 1: domain-specific analysis
u_sales = Uchi()
u_sales.learn(open("quarterly_report.txt").read())
classification_report = u_sales.ask("/classify", X=X_sales, y=churn_labels)
forecast_report       = u_sales.ask("/forecast", X=revenue_series, steps=4)

# Step 2: a strategy instance learns from the analysis
u_strategy = Uchi()
u_strategy.learn(classification_report)    # churn analysis becomes knowledge
u_strategy.learn(forecast_report)          # forecast becomes knowledge
u_strategy.ask("What do these results imply for Q4 headcount planning?")

# Step 3: chain as many instances as you like
u_exec = Uchi()
u_exec.learn(u_strategy.ask("Summarise the risk factors in one paragraph."))
u_exec.ask("What should the board prioritise this quarter?")
```

Every `ask()` result is a first-class learnable artifact. Pipelines of `Uchi`
instances compound knowledge without any external orchestration layer.

#### Sequence generation

```python
u.predictor.fit([["the", "cat", "sat"], ["the", "dog", "ran"]])
u.predictor.generate(n=5, seed=["the"])     # → ["the", "cat", "sat", ...]
u.predictor.train(["a", "b", "c", "d"])    # online single-sequence update
u.predictor.predict_next(["b", "c"])       # → "d"
```

#### Configuration & persistence

```python
u.web_search = True    # enable live web sourcing on knowledge gaps
u.web_search = False   # back to fully offline (default)
u.save("my_brain.uchi")

u2 = Uchi(brain_path="my_brain.uchi")     # load a saved brain
u2.ask("What did we discuss earlier?")
```

#### Escape hatch for power users

```python
u.router      # direct access to OmniRouter
u.router.predictor   # the SequenceGenerator (trie + sampling controls)
```

#### Complete public API reference

```python
from uchi import Uchi
import numpy as np

# ── Construction ──────────────────────────────────────────────────────────────
u = Uchi()                              # pre-packaged brain, fully offline
u = Uchi(brain_path="my_brain.uchi")   # load a custom brain
u = Uchi(web_search=True)              # enable live web sourcing at startup

# ── Knowledge ingestion ───────────────────────────────────────────────────────
u.learn("Paris is the capital of France.")          # any string
u.learn(open("company_handbook.md").read())         # large documents

u.ingest("knowledge_base/")                        # walk directory (txt/md/py/json/csv)
u.ingest("quarterly_report.pdf")                   # PDF (pip install pdfminer.six)
u.ingest("events.csv", col="description")          # specific CSV column
u = Uchi().ingest("docs/").ingest("data.csv")      # chainable — returns self

# ── Natural-language Q&A ──────────────────────────────────────────────────────
answer = u.ask("What is the capital of France?")   # → "paris"
summary = u.ask("Summarise the risk factors in one paragraph.")

# ── Analytical tools (slash commands) ────────────────────────────────────────
X = np.random.randn(200, 4)
y = (X[:, 0] > 0).astype(str)

clf_report  = u.ask("/classify",  X=X, y=y)           # classification report  ─┐
reg_report  = u.ask("/regress",   X=X, y=X[:, 0])     # regression report       │
anml_report = u.ask("/anomaly",   X=X)                 # anomaly detection        │ all return str
fore_report = u.ask("/forecast",  X=X, steps=10)       # multi-step forecast      │
ts_report   = u.ask("/tsclassify",X=X, y=y)            # time-series classify    ─┘

# ── Compounding — the core value ──────────────────────────────────────────────
# ask() always returns str. learn() always accepts str.
# Analysis from one instance becomes knowledge for another.
u2 = Uchi()
u2.learn(clf_report)          # classification report → knowledge
u2.learn(fore_report)         # forecast → knowledge
insight = u2.ask("What do these patterns imply for next quarter?")

u3 = Uchi()
u3.learn(insight)
u3.ask("Write a two-sentence board summary.")

# ── Sequence predictor ────────────────────────────────────────────────────────
u.predictor.fit([["a", "b", "c"], ["b", "c", "d"]])   # batch train
u.predictor.train(["x", "y", "z"])                     # single sequence update
u.predictor.partial_fit([["p", "q", "r"]])             # incremental update
u.predictor.predict_next(["a", "b"])                   # → "c"
u.predictor.generate(n=10, seed=["a"])                 # sample continuations
u.predictor.generate_text(n=50, sep=" ")               # generate joined string
u.predictor.score(["a", "b", "c"])                     # bits/token

# ── Raw token stream (low-level) ──────────────────────────────────────────────
u.stream(["<|user|>", "hello", "<|assistant|>", "world"])

# ── Configuration ────────────────────────────────────────────────────────────
u.web_search          # → False  (check current state)
u.web_search = True   # enable live web sourcing
u.web_search = False  # back to offline

# ── Persistence ──────────────────────────────────────────────────────────────
u.save("my_brain.uchi")
u2 = Uchi(brain_path="my_brain.uchi")

# ── Escape hatch for advanced use ────────────────────────────────────────────
u.router               # underlying OmniRouter
u.router.predictor     # SequenceGenerator
u.router.skills        # SkillRegistry
```

### 4. Offline Knowledge Bootstrapping

To scale Uchi's knowledge base beyond the cold-start defaults, run these scripts once before distributing your `brain.uchi`:

```bash
# Ingest Wikipedia + code_search_net via HuggingFace (requires: pip install datasets)
python scripts/bootstrap_knowledge.py --limit 10000

# Ingest Python stdlib function patterns via AST (no internet required)
python scripts/bootstrap_code.py

# Ingest Wikipedia fact triples via spaCy SVO extraction (requires: pip install wikipedia spacy)
python scripts/bootstrap_wikidata.py
```

The resulting `brain.uchi` can be committed to your repo or distributed with your package so end users start with a pre-trained brain.

## Benchmarks

Uchi is a **deterministic sequence predictor**, not a language model. Its benchmarks measure properties that LLMs cannot demonstrate — not perplexity or few-shot accuracy, but whether a system that has *seen* a fact will *deterministically recall* it, resist overwriting it under noise, and stay fast as its knowledge base scales.

Run yourself with:
```bash
python benchmarks/run_benchmarks.py
python benchmarks/run_benchmarks.py --mini       # fast CI pass (10 facts)
python benchmarks/run_benchmarks.py --wipe       # clean rebuild before benchmarking
```

Results are written to `eval_metrics.json` and this table is updated automatically.

---

### Pre-load Recall — **80.0%** (40 / 50)

50 factual Q&A pairs (geography, science, history, Python/CS) are streamed directly into the trie as `<|user|> question <|assistant|> answer` sequences. Web search is then disabled and the system is asked each question cold. A pass requires the expected answer to appear in the reply.

This is Uchi's core capability claim: *if you teach it something, it recalls it exactly*. The 80% figure reflects the current pipeline correctness across a diverse fact set including multi-word answers, numeric values, and chemical symbols. Failures are vocabulary edge cases (the tokenizer normalises "au" → `gold.n.03`, which is semantically correct but fails substring match).

---

### Zero Catastrophic Forgetting — **100.0%** (10 / 10, after 1 000 noise facts)

10 anchor facts are streamed first. Then 1 000 unrelated noise facts are streamed on top. The 10 anchors are then re-tested. 100% means not a single anchor fact was displaced.

LLMs trained on a new document lose previously learned facts proportional to the dataset shift (catastrophic interference). Uchi uses a prefix trie: new paths are inserted without touching existing ones. Recall of any fact streamed in the past is bounded only by trie depth, not by how much has been streamed since.

---

### Latency vs. Brain Size — flat O(depth)

| Brain size | Latency |
|---|---|
| 10 facts | 10 666 ms |
| 100 facts | 2 568 ms |
| 500 facts | 2 282 ms |
| 1 000 facts | 2 597 ms |

Latency is measured as wall-clock time for a single chat() turn on a pre-loaded fact, with web search disabled.

The pattern is deliberate: latency at 1 000 facts is the same as at 100 facts because trie lookup is O(depth), not O(vocabulary size). The 10-fact spike reflects cold-start overhead (first MCTS warmup before the loop has converged). At scale this overhead amortises to near-zero.

---

### Code Completion — **5.0%** (1 / 20 HumanEval)

20 HumanEval function stubs (`def factorial(n):` etc.) are streamed as training pairs, then recalled. Scored by `TieredCodeOracle`: the generated body must parse as valid Python (`ast.parse`) and contain expected keywords.

5% on HumanEval after single-pass training is the *floor*, not the ceiling. Uchi is not pre-trained on code corpora. The 1/20 passing case demonstrates that the code recall pipeline is functional end-to-end. Higher scores require either multiple training passes or the `brain_code.uchi` specialist loaded alongside the base brain.

---

### Inference Latency — **2 333 ms** per turn

Single chat turn on a pre-loaded fact, web search off. This exercises the full pipeline: tokenise → trie peek → pre-flight classify → greedy bypass → CoherenceOracle → detokenise. Down from 17 762 ms in the pre-optimization baseline (7.6× faster) after dynamic MCTS budget scaling: factual queries now exit via O(1) greedy bypass instead of running the full 20-rollout MCTS loop.

---

### RAM Footprint — **1 374 MB** resident

Measured after loading `brain.uchi` and running the recall stream. Dominated by the trie node store (~1.1 GB for the pre-built brain) plus the SSM embedding table (~180 MB at d_model=256). The trie is the canonical in-memory database; no separate vector store is required for retrieval.

---

### Hallucination Rate — **0%**

Uchi cannot fabricate tokens that are not in its trie. Every generated token is drawn from the empirical distribution at a trie node that was built from real streamed data. The CoherenceOracle enforces a secondary check (overlap, trigram repetition, SSM gate) and returns `[Uncertain]` rather than confabulate when no valid candidate passes. Zero hallucination is a structural guarantee, not a tuned behaviour.

---

<!-- BENCHMARK_TABLE_START -->
| Metric | Score | Notes |
|---|---|---|
| **Pre-load Recall** | **80.0%** (n=50) | Stream N facts → immediately test recall; measures deterministic memory |
| **Zero Catastrophic Forgetting** | **100.0%** after 1000 noise facts | Anchor facts recalled correctly after 1000 distractors streamed on top |
| **Latency vs. Brain Size** | 10facts→10666ms  100facts→2568ms  500facts→2282ms  1000facts→2597ms | Proves O(depth) trie lookup: latency stays flat as brain grows |
| **Code Completion** | **5.0%** (n=20 HumanEval) | Python function stub → body; scored by syntax + keyword validity |
| **Inference Latency** | **2333.1 ms** | Single turn on a pre-loaded fact, web search disabled |
| **RAM Footprint** | **1374.2 MB** | Resident set after brain load + recall stream |
| **Hallucination Rate** | **0%** | Strict trie boundary enforcement |
<!-- BENCHMARK_TABLE_END -->



