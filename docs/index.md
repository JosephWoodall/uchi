# Uchi Documentation

> **The API has been dramatically simplified.** One import. Everything discoverable. Analysis compounds across instances without glue code. Start here.

---

## The New Way to Use Uchi

```python
from uchi import Uchi

u = Uchi()
u.learn("Q3 revenue was $4.2M, up 23% YoY.")
print(u.ask("What was Q3 revenue growth?"))
```

`Uchi` is the single public entry point for the entire library. Every feature — the sequence predictor, every analytical tool, web search, persistence — is reachable through this one class.

**[Full Python API reference →](python-api.md)**

---

## The Compounding Mechanism

`ask()` always returns a plain string. `learn()` always accepts a plain string. This means the output of any analysis is immediately learnable by any other `Uchi` instance — no serialisation, no schema, no glue code.

```python
# Instance 1: domain analysis
u_data = Uchi()
u_data.learn(open("sales_data.txt").read())
report = u_data.ask("/classify", X=X_train, y=churn_labels)

# Instance 2: learns from the analysis
u_strategy = Uchi()
u_strategy.learn(report)
u_strategy.ask("What does this churn pattern imply for Q4 headcount?")

# Instance 3: compounds further
u_exec = Uchi()
u_exec.learn(u_strategy.ask("Summarise the top three risks."))
u_exec.ask("What should the board prioritise this quarter?")
```

Every `ask()` result is a first-class learnable artifact. Pipelines of `Uchi` instances build compounding analytical context without any external orchestration layer.

---

## What Uchi Does

Uchi is an **online, instance-based sequence predictor** that learns to predict what comes next across any symbol type and domain — without neural weights, without pre-training, and without catastrophic forgetting.

Its clearest domain is **discrete event streams where the underlying pattern shifts over time**. It beats count-based methods (N-gram, PPM, CTW) and online neural methods specifically in non-stationary settings — with no retraining, no drift detector, and no forgetting window to tune.

The `Uchi` class exposes all of this through a single API: natural-language Q&A, tabular ML, time series forecasting, anomaly detection, and sequence generation.

---

## Quickstart

### Knowledge & Q&A

```python
u = Uchi()
u.learn("The boiling point of water is 100°C at sea level.")
u.ask("At what temperature does water boil?")   # → "100°C"
```

### Analytical tools

```python
result = u.ask("/classify",  X=X_train, y=y_train)    # classification report
result = u.ask("/regress",   X=X_train, y=y_train)    # regression report
result = u.ask("/anomaly",   X=sensor_matrix)          # anomaly detection report
result = u.ask("/forecast",  X=time_series, steps=20)  # forecast report
result = u.ask("/tsclassify",X=windows,    y=labels)   # time series classification
```

### Sequence prediction

```python
u.predictor.fit([["a", "b", "c", "d"]])
u.predictor.predict_next(["b", "c"])          # → "d"
u.predictor.train(["x", "y", "z"])            # online single-sequence update
u.predictor.generate(n=10, seed=["a"])        # sample continuations
```

### File and directory ingestion

```python
u.ingest("knowledge_base/")            # walk directory — txt/md/py/json/csv
u.ingest("report.pdf")                 # PDF (pip install pdfminer.six)
u.ingest("events.csv", col="notes")    # specific CSV column
u = Uchi().ingest("docs/").ingest("data.csv")  # chainable, returns self
```

### Persistence and configuration

```python
u.web_search = True          # enable live web sourcing on knowledge gaps
u.save("my_brain.uchi")
u2 = Uchi(brain_path="my_brain.uchi")
```

---

## Natural Fits

- **System observability** — log event codes, API call chains, process state transitions
- **User behavior** — clickstreams, navigation paths, in-app action sequences
- **Industrial / IoT** — machine state sequences, energy consumption, production line events
- **Financial regimes** — discretized price movements, order flow states, market microstructure
- **Anomaly detection** — confidence collapse before a human notices; no separate anomaly model
- **Game AI / opponent modeling** — predict next move, adapts to strategy shifts in real time

## Where It Is Not Competitive

- **Large stationary tabular data (>10K rows, no drift)** — gradient boosting wins by 5–10pp
- **Long-range dependencies** — context window is fixed at k; needs a transformer for longer memory
- **Smooth continuous regression** — binned output bounds precision below random forests

---

## Documentation

| Section | Contents |
|---|---|
| **[Python API](python-api.md)** | Full `Uchi` class reference — `learn`, `ask`, `stream`, `predictor`, `web_search`, `save` |
| **[Architecture](architecture.md)** | Trie algorithm, credibility update, routing layer design |
| **[Core Engine](core-engine.md)** | `UniversalPredictor`, `PredictorForest` — low-level sequence predictor API |
| **[OmniRouter](omni-router.md)** | Multi-modal routing, `ProceduralMemory`, SSM GRPO value head |
| **[Generative Models](generative.md)** | `SequenceGenerator`, `TabularGenerator`, `TimeSeriesGenerator` |
| **[Tabular ML](tabular.md)** | `TabularPredictor`, `TabularRegressor` — sklearn-compatible classifiers |
| **[Time Series](timeseries.md)** | `MultivariateTSPredictor`, `TimeSeriesClassifier`, `AnomalyDetector` |
| **[Convergent Engine](convergent-engine.md)** | MCTS, Oracles, vector ranking |
| **[Benchmarks](benchmarks.md)** | Standard and concept-drift benchmark results |
| **[Algorithmic Walkthrough](algorithmic-walkthrough.md)** | Step-by-step derivation of the core algorithm |
| **[Simulation Engine](simulation-engine.md)** | `SimulationEngine` for scenario modeling |
