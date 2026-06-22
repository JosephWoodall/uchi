# Uchi Documentation

Online, instance-based sequence predictor. Given any stream of discrete observations, it learns to predict what comes next — for any symbol type, in any domain — without assuming a fixed distribution, a known alphabet, or a stationary process.

The `uchi` package extends this core engine to tabular classification, regression, multivariate time series forecasting, anomaly detection, and generative modeling. All classes are sklearn-compatible.

## What this is for

**Its clearest domain: discrete event streams where the underlying pattern shifts over time.**

If you have a stream of categorical states and need to predict the next one — without knowing in advance how the pattern will change — this is the right tool. It beats count-based methods (N-gram, PPM, CTW) and online neural methods specifically in non-stationary settings, and it does so with no retraining, no drift detector, and no forgetting window to tune.

**Natural fits:**

- **System observability** — sequences of log event codes, API call chains, process state transitions. Predicts next failure type. When a deployment changes the pattern, adaptation is automatic.
- **User behavior** — clickstreams, navigation paths, in-app action sequences. Next-action prediction that updates on every new user event without a retraining cycle.
- **Industrial / IoT** — machine state sequences (idle / running / warning / fault), energy consumption states, production line events. Works on tiny datasets where neural methods don't have enough data.
- **Financial regimes** — discretized price movements, order flow states, market microstructure events. Handles regime shifts that break count-based models.
- **Anomaly detection** — when the predictor is consistently wrong, something structurally unusual is happening. Confidence collapses before a human notices; no separate anomaly model needed.
- **Game AI / opponent modeling** — predict next move in any discrete-action game. Adapts to opponent strategy shifts in real time.

**Where it is not competitive:**

- **Tabular classification where the data is large and stationary** — on tabular datasets >10K rows without concept drift, gradient boosting will typically win by 5–10pp. The trie shines when data is small, streaming, or drifting.
- **Long-range sequence dependencies** — the context window is fixed at k. Anything requiring memory beyond the last k observations needs a transformer or RNN.
- **Large stationary corpora** — on 50K tokens of text or DNA, count-based methods (CTW, KN) hold a 1–2pp accuracy advantage because their unbounded counts eventually outcompete the credibility cap. The gap closes on noisy or drifting data.
- **Continuous regression targets** — the regressor bins the output; precision is bounded by `n_target_bins`. Point-prediction accuracy on smooth regression tasks is below random forests.
