# Time Series Models

**`MultivariateTSPredictor`**

Online step-ahead predictor for multivariate (or univariate) time series. Each timestep is encoded as a compound token `(bin_0, bin_1, ..., bin_{M-1})` — a hashable tuple the trie matches exactly. Context is the last k compound tokens. Adapts immediately to distribution shift without retraining.

```python
pred = MultivariateTSPredictor(n_bins=8, context_length=5)
pred.fit(X_train)              # warm up trie on historical data
pred.predict()                 # float vector (per-dimension means)
pred.observe(x_new)            # advance internal state
pred.feedback(x_new)           # update trie with true value
pred.forecast(n_steps=10)      # autoregressive multi-step forecast
pred.score(X_test)             # bits/step (lower = better fit)
```

**`TimeSeriesClassifier`**

Classifies fixed-length time series windows. Each window of T steps becomes T compound tokens; the class label is predicted as the next token after the full window. Supports `partial_fit` for streaming classification. Works in sklearn Pipeline.

```python
clf = TimeSeriesClassifier(n_bins=8, window_size=50)
clf.fit(X_windows, y_labels)
clf.predict(X_test)            # class labels
clf.predict_proba(X_test)      # list of {label: prob} dicts
```

**`AnomalyDetector`**

Trains a `MultivariateTSPredictor` on normal data. At inference, each timestep receives anomaly score = `-log2 P(actual | context)`. High score = low predictability = anomalous. The trie is not updated during scoring, so anomalous patterns do not contaminate the model of normal behavior.

sklearn `OutlierMixin` compliant: `predict()` returns 1 (anomaly) / -1 (normal); `decision_function()` returns negative anomaly scores for threshold-based pipelines.

```python
det = AnomalyDetector(n_bins=8, context_length=5)
det.fit(X_normal)
det.score_samples(X_test)      # float scores (higher = more anomalous)
det.predict(X_test)            # 1 or -1 per timestep
```
