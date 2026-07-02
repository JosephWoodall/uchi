# Python API — `Uchi`

`Uchi` is the single public entry point for the entire library. One import. Everything discoverable. Analysis compounds across instances without glue code.

```python
from uchi import Uchi
```

---

## The Compounding Mechanism

This is the most important concept in the entire API.

`ask()` **always returns a plain string.** `learn()` **always accepts a plain string.** This single design choice means the output of any analysis — a classification report, a forecast, a Q&A answer — is immediately learnable knowledge for any other `Uchi` instance.

No serialisation. No shared schema. No orchestration layer. The string is the interface.

```python
# Three instances. Each learns from the previous one's output.
u1 = Uchi()
u1.learn(open("quarterly_report.txt").read())
classification_report = u1.ask("/classify", X=X_sales, y=churn_labels)
forecast_report       = u1.ask("/forecast", X=revenue_series, steps=4)

u2 = Uchi()
u2.learn(classification_report)    # churn analysis becomes knowledge
u2.learn(forecast_report)          # forecast becomes knowledge
strategy = u2.ask("What do these results imply for Q4 headcount planning?")

u3 = Uchi()
u3.learn(strategy)
u3.ask("What should the board prioritise this quarter?")
```

Every `ask()` result is a first-class learnable artifact. Chain as many instances as you like.

---

## Constructor

```python
Uchi(brain_path=None, web_search=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `brain_path` | `str \| None` | `None` | Path to a `brain.uchi` file. Uses the pre-packaged brain when `None`. |
| `web_search` | `bool` | `False` | Enable live web sourcing when the brain has a knowledge gap. Fully offline by default. |

```python
u = Uchi()                                  # pre-packaged brain, offline
u = Uchi(brain_path="my_brain.uchi")        # custom brain
u = Uchi(web_search=True)                   # live web sourcing enabled
```

---

## `learn(text)`

Stream text into the brain's knowledge trie. For loading files or directories, use `ingest()` instead.

```python
u.learn("The boiling point of water is 100°C at sea level.")
u.learn(open("company_handbook.txt").read())
u.learn(some_ask_result)   # compounding: analysis becomes knowledge
```

Accepts any string. Tokenises on whitespace and feeds directly into the trie. There is no size limit, no training epoch, no retraining delay — the brain updates incrementally on every token.

---

## `ingest(path, col=None)`

Load files or directories into the brain. Returns `self` for chaining.

```python
u.ingest("knowledge_base/")            # walk directory — txt/md/py/json/csv
u.ingest("report.pdf")                 # PDF (requires pip install pdfminer.six)
u.ingest("events.csv", col="notes")    # specific CSV column only
u.ingest("handbook.md")                # single file

# chainable
u = Uchi().ingest("docs/").ingest("data.csv").ingest("handbook.md")
u.save("expanded_brain.uchi")
```

| Format | Behaviour |
|---|---|
| `.txt` `.md` `.rst` `.py` `.yaml` `.toml` `.sh` `.ini` `.cfg` | Read as UTF-8, passed to `learn()` |
| `.csv` | All text cells concatenated per row, or a single column when `col` is given |
| `.json` | All string values extracted recursively, passed to `learn()` |
| `.pdf` | Text extracted via `pdfminer.six`; skipped with a warning when the package is absent |
| Other extensions | Silently skipped |

Unreadable files are silently skipped so that an entire project directory can be ingested safely.

| Parameter | Type | Description |
|---|---|---|
| `path` | `str` | File or directory path. `~` is expanded. |
| `col` | `str \| None` | CSV only: name of the column to ingest. All text cells when `None`. |

---

## `ask(question, **data)`

Ask the brain a question or invoke an analytical tool.

**Natural-language questions** route through the three-lane router — factual
questions go to Generate-and-Ground (retrieve → generate → fact-check → answer or
abstain), social turns to the conversation engine, and skill commands to the
`SkillRegistry`. Multi-step factual questions go through the verified reasoner.
Uchi answers when it can ground the answer, and abstains otherwise:

```python
u.ask("What is the capital of France?")     # grounded answer, or "I don't have grounded knowledge…"
u.ask("Explain the water cycle.")
u.ask("Summarise the risk factors in one paragraph.")
```

**Slash commands with keyword data** invoke the corresponding analytical skill directly, bypassing string parsing:

```python
result = u.ask("/classify",  X=X_train, y=y_train)
result = u.ask("/regress",   X=X_train, y=y_train)
result = u.ask("/anomaly",   X=sensor_matrix)
result = u.ask("/forecast",  X=time_series, steps=20)
result = u.ask("/tsclassify",X=windows, y=labels)
```

`X` accepts a pandas DataFrame, numpy array, or list-of-lists. `y` accepts a list or 1-D array. `steps` is an integer.

**All forms return a plain string.** This is the compounding guarantee — every result feeds directly into `learn()` on any other instance.

The same slash commands work from the TUI with a CSV path:

```
/classify data.csv --label target_col
/anomaly  sensors.csv
/forecast series.csv --steps 20
```

---

## `stream(tokens)`

Low-level path: feed a raw token list directly into the trie, bypassing tokenisation.

```python
u.stream(["<|user|>", "hello", "<|assistant|>", "world"])
```

Use `learn()` for most cases. Use `stream()` when you need exact token control.

---

## `predictor`

The `SequenceGenerator` powering the brain's trie. Full sklearn-compatible sequence API.

```python
u.predictor.fit(sequences)              # train on a list of token sequences
u.predictor.partial_fit(sequences)      # online incremental update
u.predictor.generate(n=10, seed=["x"]) # sample n tokens from learned distribution
u.predictor.generate_text(n=50, sep=" ")# generate and join as a string
u.predictor.train(sequence)             # single online sequence update
u.predictor.predict_next(context)       # argmax next token given context
u.predictor.score(sequence)             # bits-per-token (lower = better fit)
```

`train()` and `predict_next()` are the recommended online interfaces:

```python
u.predictor.train(["a", "b", "c", "d", "e"])
u.predictor.predict_next(["c", "d"])   # → "e"
```

`fit()` is the batch interface — equivalent to calling `train()` for each sequence.

---

## `web_search`

Toggle live web sourcing on or off at any time.

```python
u.web_search          # → False  (default: fully offline)
u.web_search = True   # enable autonomous web sourcing on knowledge gaps
u.web_search = False  # back to offline
```

When `True`, the brain automatically queries the web when it detects a knowledge gap and cannot generate a confident answer from the trie alone.

---

## `save(path)`

Persist the current brain state to disk.

```python
u.save("my_brain.uchi")
u2 = Uchi(brain_path="my_brain.uchi")   # load in a new session
```

The file is gzip-compressed pickle. The same file format is used by the TUI and the REST API server.

---

## `router`

Escape hatch for power users. Direct access to the underlying `OmniRouter`.

```python
u.router                # OmniRouter instance
u.router.predictor      # SequenceGenerator (same as u.predictor)
u.router.skills         # SkillRegistry
u.router.chat("...")    # raw chat() without Uchi wrapping
```

Use this only when you need something `Uchi` does not yet expose. All common operations are available on the `Uchi` surface.

---

## Complete example

```python
import numpy as np
from uchi import Uchi

# ── Load domain knowledge ──────────────────────────────────────────────────
u_domain = Uchi()
u_domain.learn(open("product_docs.txt").read())

# ── Run tabular analysis ───────────────────────────────────────────────────
X = np.random.randn(200, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(str)

clf_report  = u_domain.ask("/classify", X=X[:160], y=y[:160])
anml_report = u_domain.ask("/anomaly",  X=X[:160])

# ── Compound into a strategy instance ─────────────────────────────────────
u_strategy = Uchi()
u_strategy.learn(clf_report)
u_strategy.learn(anml_report)
insight = u_strategy.ask("What patterns in the anomaly report overlap with the classification results?")

# ── Compound into an executive summary ───────────────────────────────────
u_exec = Uchi()
u_exec.learn(insight)
summary = u_exec.ask("Write a two-sentence board-level summary of the findings.")

# ── Persist the executive instance ────────────────────────────────────────
u_exec.save("executive_brain.uchi")
```

---

## FAQ

**Q: Does each `Uchi` instance have its own brain?**
Yes. Instances are independent. The compounding effect comes from passing `ask()` output strings to `learn()` — not from shared state.

**Q: Can I share a brain across instances?**
Load the same `brain.uchi` file: `Uchi(brain_path="shared.uchi")`. Each instance gets its own copy in memory; mutations do not propagate back to disk until you call `save()`.

**Q: What formats does `X` accept?**
pandas DataFrame, numpy array (2-D), or list-of-lists. Column order is preserved; column names are ignored.

**Q: Is there a size limit on `learn()`?**
No hard limit. The trie grows with observed sequences. RAM scales with vocabulary × context depth, not corpus size.

**Q: How is `learn()` different from `stream()`?**
`learn(text)` tokenises on whitespace then calls `stream()`. Use `learn()` for text input, `stream()` when you have already tokenised tokens and want exact control.
