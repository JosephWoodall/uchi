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

### Realistic Use Cases
1. **Stock Portfolio Correlation Forecasting**: Stream thousands of stock tickers natively. As the entire portfolio moves, the engine learns the multivariate correlation between all assets instantly.
2. **Hyper-Local Weather Grid Prediction**: Predict atmospheric changes across a grid of weather stations simultaneously without retraining heavy Recurrent Neural Networks every 24 hours.
3. **Multi-Sensor Engine Telemetry**: Track all fluid, temperature, and pressure sensors in an automotive or aerospace engine continuously to predict catastrophic cascading failures before they occur.

### The Ultimate Benefit
The `MultivariateTSPredictor` replaces heavy, brittle LSTMs with a lightweight predictor that **adapts instantly to physical regime shifts**. It maps out the exact dependencies across multiple continuous variables in real time.

**`TimeSeriesClassifier`**

Classifies fixed-length time series windows. Each window of T steps becomes T compound tokens; the class label is predicted as the next token after the full window. Supports `partial_fit` for streaming classification. Works in sklearn Pipeline.

```python
clf = TimeSeriesClassifier(n_bins=8, window_size=50)
clf.fit(X_windows, y_labels)
clf.predict(X_test)            # class labels
clf.predict_proba(X_test)      # list of {label: prob} dicts
```

### Realistic Use Cases
1. **Streaming ECG/Heartbeat Classification**: Classify incoming heart rhythms instantly on edge medical hardware as normal, atrial fibrillation, or anomalous with zero batching delay.
2. **Server Log DDOS Attack Classification**: Read incoming streams of packet metrics to classify normal spikes in traffic versus a coordinated attack sequence.
3. **Real-Time Gesture Recognition**: Classify human movements from mobile gyroscope and accelerometer sensors without draining the battery with heavy neural architectures.

### The Ultimate Benefit
The `TimeSeriesClassifier` provides **Sklearn-compatible, purely streaming time-series classification** without rigid padding, batching, or window truncation limitations.

**`AnomalyDetector`**

Trains a `MultivariateTSPredictor` on normal data. At inference, each timestep receives anomaly score = `-log2 P(actual | context)`. High score = low predictability = anomalous. The trie is not updated during scoring, so anomalous patterns do not contaminate the model of normal behavior.

sklearn `OutlierMixin` compliant: `predict()` returns 1 (anomaly) / -1 (normal); `decision_function()` returns negative anomaly scores for threshold-based pipelines.

```python
det = AnomalyDetector(n_bins=8, context_length=5)
det.fit(X_normal)
det.score_samples(X_test)      # float scores (higher = more anomalous)
det.predict(X_test)            # 1 or -1 per timestep
```

### Realistic Use Cases
1. **Zero-Day Cyber Attack Detection**: Rather than looking for known threat signatures, this flags any sequence of packet behaviors that heavily deviates from the learned baseline of your exact, normal network geometry.
2. **Predictive Maintenance for Factories**: Deploy to industrial robotic arms. The moment a motor begins wearing down, the exact acoustic and vibration patterns change, throwing an immediate anomaly score spike.
3. **Data Pipeline Corruption Detection**: Prevent bad data from entering your database. If a data stream begins logging corrupted or garbage strings, the engine's prediction confidence collapses and flags the corruption.

### The Ultimate Benefit
The `AnomalyDetector` ensures that **anomalous patterns never contaminate the underlying model of normal behavior**. It provides highly calibrated float scores where mathematically high unpredictability natively equals anomalous data.
