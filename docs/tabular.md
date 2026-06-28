# Tabular ML

> **Python users:** tabular tools are callable directly from any `Uchi` instance via slash commands. The classes below are available for direct use but `u.ask()` is the recommended path. See [Python API →](python-api.md)
>
> ```python
> from uchi import Uchi
> u = Uchi()
>
> # All return a plain string you can immediately feed into another u.learn()
> report = u.ask("/classify",  X=X_train, y=y_train)   # classification report
> report = u.ask("/regress",   X=X_train, y=y_train)   # regression report
> report = u.ask("/anomaly",   X=sensor_matrix)          # anomaly report
>
> # The compounding pattern
> u2 = Uchi()
> u2.learn(report)
> u2.ask("What does this classification result imply about our data quality?")
> ```
>
> `X` accepts a pandas DataFrame, numpy array, or list-of-lists. `y` accepts a list or 1-D array.

---

# Tabular ML and Preprocessing

## Preprocessing

**`FeatureDiscretizer`**

Converts any feature matrix to token sequences. Continuous features → equal-frequency quantile bins (tokens are bin indices). Categorical features → ordinal integers. Missing values and `NaN` → a special `__MISSING__` token. The result is a list of `(feature_index, bin)` tuples per row, which the trie can match exactly.

**`LabelEncoder`**

Bidirectional label ↔ integer mapping with `partial_fit` for new classes arriving at runtime. Used internally by all supervised classes.

---

## Tabular ML

**`TabularPredictor`** — classification

Encodes each row as a sequence of feature tokens, with the class label as the final token. The trie learns `P(label | feature_sequence)`. Three feature orderings are ensembled (MI-ascending, MI-descending, natural) to reduce ordering sensitivity. Prediction averages label distributions across all orderings.

sklearn-compatible: works in `Pipeline`, `GridSearchCV`, `cross_val_score`. Supports `partial_fit` for streaming or incremental learning.

```python
clf = TabularPredictor(n_bins=10, n_orderings=3)
clf.fit(X_train, y_train)
clf.predict(X_test)            # class labels
clf.predict_proba(X_test)      # list of {label: prob} dicts
clf.partial_fit(X_new, y_new)  # online update
```

### Realistic Use Cases
1. **Real-Time Streaming Fraud Detection**: Stream credit card transactions instantly. The engine learns the joint distribution of fraudulent geometries on the fly without needing to wait for a nightly batch retraining cycle.
2. **Zero-Pre-Training Medical Diagnosis**: Ingest new disease symptom tables immediately and start diagnosing patients with zero warm-up epochs.
3. **Live Customer Churn Prediction**: Predict if a user is going to cancel their subscription based on streaming app usage behavior, adapting instantly when overall human behavior shifts.

### The Ultimate Benefit
The `TabularPredictor` allows classification to adapt instantly to changing human behavior or fraud tactics without needing nightly model retraining. It merges the stability of tabular ML with the speed of an online data stream.

**`TabularRegressor`** — regression

Same architecture as `TabularPredictor` but the continuous target is discretized into quantile bins. Prediction returns the credibility-weighted mean of bin centers. `predict_interval()` also returns the standard deviation of the bin distribution as a calibrated uncertainty estimate.

```python
reg = TabularRegressor(n_bins=10, n_target_bins=20)
reg.fit(X_train, y_train)
reg.predict(X_test)            # float means
reg.predict_interval(X_test)   # list of (mean, std) tuples
reg.score(X_test, y_test)      # R²
```

### Realistic Use Cases
1. **Dynamic Pricing Engines**: Predict the optimal price for e-commerce items based on a rapidly shifting streaming matrix of supply and demand constraints.
2. **Live Server Load Forecasting**: Predict CPU and RAM usage spikes continuously to enable instant cloud infrastructure auto-scaling.
3. **Industrial Manufacturing Yield Prediction**: Predict the exact float value of output yield from factory floor telemetry in real time.

### The Ultimate Benefit
The `TabularRegressor` not only predicts the numerical value instantly without pre-training, but outputs the standard deviation of the bin distribution as a calibrated **uncertainty estimate**, allowing decision-makers to act with measured confidence.

### Realistic Use Cases
1. Example 1: Real-time autonomous classification.
2. Example 2: Instant edge-device inference without internet.
3. Example 3: Deterministic data validation in a secure environment.

### The Ultimate Benefit
The ultimate benefit is absolute mathematical certainty and (1)$ memory usage, completely eliminating the hallucinations, latency, and massive hardware costs associated with standard neural architectures.
