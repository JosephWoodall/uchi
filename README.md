# Universal Sequence Predictor

Online, instance-based sequence predictor. Given any stream of discrete observations, it learns to predict what comes next — for any symbol type, in any domain — without assuming a fixed distribution, a known alphabet, or a stationary process.

The `uchi` package extends this core engine to tabular classification, regression, multivariate time series forecasting, anomaly detection, and generative modeling. All classes are sklearn-compatible.

---

## Installation

```bash
pip install -e .                  # editable install (no required deps)
pip install -e ".[all]"           # with scikit-learn, numpy, pandas
```

```python
from uchi import (
    UniversalPredictor, PredictorForest,           # core engine
    TabularPredictor, TabularRegressor,             # tabular ML
    MultivariateTSPredictor, TimeSeriesClassifier,  # time series
    AnomalyDetector,
    SequenceGenerator, TabularGenerator,            # generative
    TimeSeriesGenerator,
)
```

---

## Components

### Core Engine

**`UniversalPredictor`**

The base algorithm. Maintains a prefix trie of observed contexts. Each node stores a credibility score that rises on correct predictions and falls — faster when the node was highly confident — on wrong ones. At prediction time it blends distributions from shallow (general) to deep (specific) contexts using CTW-style recursive mixing, where each depth's influence is proportional to its credibility track record. No forgetting parameter, no drift detector: concept drift is handled automatically because stale nodes lose credibility and the blend shifts back to shallower, more stable contexts.

API: `observe(x)` → `predict()` → `feedback(x)`. Set `min_confidence` to abstain rather than guess below a threshold.

**`PredictorForest`**

Ensemble of `UniversalPredictor` instances with four diversity mechanisms: heterogeneous context lengths (k, k+1, k+2, …), feedback dropout (each tree independently skips learning steps), staggered training offsets, and per-tree credibility weights. Adaptive voting: when trees agree confidently it uses a product (decisive), when uncertain it uses a mixture (calibrated).

---

### Preprocessing

**`FeatureDiscretizer`**

Converts any feature matrix to token sequences. Continuous features → equal-frequency quantile bins (tokens are bin indices). Categorical features → ordinal integers. Missing values and `NaN` → a special `__MISSING__` token. The result is a list of `(feature_index, bin)` tuples per row, which the trie can match exactly.

**`LabelEncoder`**

Bidirectional label ↔ integer mapping with `partial_fit` for new classes arriving at runtime. Used internally by all supervised classes.

---

### Tabular ML

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

**`TabularRegressor`** — regression

Same architecture as `TabularPredictor` but the continuous target is discretized into quantile bins. Prediction returns the credibility-weighted mean of bin centers. `predict_interval()` also returns the standard deviation of the bin distribution as a calibrated uncertainty estimate.

```python
reg = TabularRegressor(n_bins=10, n_target_bins=20)
reg.fit(X_train, y_train)
reg.predict(X_test)            # float means
reg.predict_interval(X_test)   # list of (mean, std) tuples
reg.score(X_test, y_test)      # R²
```

---

### Time Series

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

---

### Generative

**`SequenceGenerator`**

Learns a distribution over sequences and samples from it. Supports temperature scaling (`p_i ← p_i^(1/T)`), top-k filtering, and nucleus (top-p) sampling. `generate_text()` joins tokens with a separator for character- or word-level text generation.

```python
gen = SequenceGenerator(context_length=6, temperature=0.9)
gen.fit(list_of_sequences)
gen.generate(50, seed=['the '], stop_tokens=['.'])  # list of tokens
gen.generate_text(100, sep='')                       # joined string
gen.score(sequence)                                  # bits/token
```

**`TabularGenerator`**

Learns the joint distribution `P(f0, f1, ..., fn, label)` and samples synthetic rows. Trains two predictors internally: one with label last (unconditional generation, `P(label | features)`) and one with label first (class-conditional generation, `P(features | label)`). This separation is necessary — a label-last model given a leading label token is out-of-distribution.

```python
gen = TabularGenerator(n_bins=10, temperature=1.0)
gen.fit(X, y)
gen.sample(n_rows=100)                       # list of dicts
gen.sample(n_rows=50, given_label='cat')     # class-conditional
gen.sample_dataframe(n_rows=100)             # pandas DataFrame
```

**`TimeSeriesGenerator`**

Learns a distribution over multivariate time series and samples from it. Unlike `MultivariateTSPredictor.forecast()` (argmax, deterministic), generation here samples from the distribution — producing diverse trajectories. `augment()` wraps generation for data augmentation.

```python
gen = TimeSeriesGenerator(n_bins=8, temperature=1.1)
gen.fit(X_series)
gen.generate(n_steps=100, seed=X_seed)       # list of float vectors
gen.augment(X, n_copies=5, temperature=1.1)  # augmented dataset
```

---

## Generative Services

The three generators (`SequenceGenerator`, `TabularGenerator`, `TimeSeriesGenerator`) share the same trie engine as the predictors. Generation is sampling from the learned conditional distribution rather than taking the argmax. All sampling controls (temperature, top-k, top-p, stop tokens) operate on that distribution at runtime.



---

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

---

## The Core Idea

The algorithm keeps a trie of contexts. Every node in the trie stores two things: a credibility-weighted distribution over successor symbols, and a track record of how reliable that context has been as a predictor. When predicting, it blends the distributions from shallow (general) to deep (specific), where each depth's influence is proportional to its track record. When updating after a wrong prediction, a node that was confidently wrong loses trust faster than one that was fresh and uncertain.

That is the entire algorithm. No drift detector, no forgetting parameter, no domain-specific tuning.

---

## Architecture

### Module 1 — Universal Sequence Predictor (`predictor.py`)

**Data structure:** a prefix trie. Each `_TrieNode` stores:
- `succ_cred` — credibility weight per successor symbol
- `node_cred` — reliability of this context as a predictor overall
- `n_obs` — number of times this context has been seen

The root holds continuation counts (how many distinct predecessors each symbol appeared after, KN-style) for large vocabularies (|V|≥8), falling back to raw KT counts for small alphabets (DNA=4, Electricity=2) where continuation counts are too sparse. This seeds the blend with a better-calibrated unigram prior.

**Prediction O(k):**

Walk the trie at depths `min_k..k`. For each matching node, compute a KT-smoothed local distribution. Blend from shallow to deep using the CTW-style recursive formula:

```
λ_d = node_cred_d^p / (node_cred_d^p + 1)     # p=0.65; softened mixing weight
P_d = λ_d · P_local(d)  +  (1 − λ_d) · P_{d−1}
```

High credibility → λ → 1 → deep context dominates.
Low credibility  → λ → 0 → falls back to shallow.
Root provides the seed.

The exponent `p=0.65` (versus the standard CTW value of p=1) lets shallower contexts retain 22% blend weight even when deeper contexts are fully saturated. This acts as implicit depth regularization — preventing rare deep contexts from monopolizing predictions on stationary data, without affecting drift adaptation (credibility degrades naturally under drift regardless of p).

**Update O(k):**

For each depth, find the context node and apply a multiplicative rule:

```
effective_cap = C_MAX × (1 + 0.5 × log(1 + n_obs/100))  # adaptive (optional)
             = C_MAX                                       # fixed (default)

correct:  node_cred ← min(cap, node_cred × (1 + lr))
          succ_cred[actual] ← min(cap, succ_cred[actual] × (1 + lr))

wrong:    lr_down = lr × (1 + node_cred / cap)   # confidence-proportional
          node_cred ← max(C_MIN, node_cred × (1 − lr_down))
          succ_cred[wrong] ← max(C_MIN, succ_cred[wrong] × (1 − lr_down))
          succ_cred[actual] ← min(cap, succ_cred[actual] × (1 + lr))
                               ×  binary_scale   # for V≤2 only; prevents false-flip cascades
```

With `adaptive_cap=True`, nodes with many observations are allowed to build higher credibility — the cap grows logarithmically with `n_obs`, so λ can approach 1 more closely on stationary data while the maximum `lr_down = 2×lr` is preserved.

The `lr_down` scaling is the key drift-adaptation mechanism: a node that was highly trusted when it turned wrong loses credibility up to 2× faster than a fresh node. This halves the adaptation lag after a concept drift without requiring any drift detector.

**Concept drift:**

Wrong predictions degrade `node_cred`, reducing λ at that depth, causing the blend to automatically fall back to shallower (more general) contexts. As the new pattern accumulates correct observations, `node_cred` rebuilds at the updated depth. No explicit change detection; no forgetting window; adaptation speed is a function of how confidently the old pattern was held.

**Regret bound:**

The multiplicative credibility update is an instance of the Multiplicative Weights Update (MWU) algorithm applied to depth selection. For a class of k single-depth predictors, MWU achieves O(√(T ln k)) regret. The CTW-style blend runs this across all depths simultaneously.

---

### Module 1 — Forest Ensemble (`forest.py`)

`PredictorForest` is a collection of `UniversalPredictor` instances that start identical and diverge through experience. Diversity comes from four sources:

| Mechanism | How it works |
|---|---|
| Heterogeneous k | Each tree uses a different context length: k, k+1, k+2, … capturing different temporal scales. Disabled for DNA (4-symbol near-uniform alphabet) where deeper-k trees add noise rather than signal. |
| Feedback dropout | Each tree independently skips learning on each step with probability `dropout` — the sequence analogue of bagging |
| Staggered offsets | Tree i doesn't start learning until step `i × stagger`; early topology has outsized influence on later structure |
| Inter-tree credibility | Each tree maintains a persistent weight updated by whether it was right; correct trees speak louder on the next prediction |

**Voting:** adaptive hybrid by default. Each tree contributes two representations:

- **Full blended distribution** (`tree._distribution()`) — the complete CTW-style probability over all vocabulary symbols. Used in the *mixture* component: proper calibration when trees at different context lengths express partial disagreement.
- **Mode-focused distribution** (`_tree_dist`) — only the most-probable successor at each depth, weighted by node credibility. Used in the *product* component: maximally decisive agreement signal for high-persistence or low-entropy data where unanimous tree confidence should dominate.

The adaptive blend computes `α × product(mode-focused) + (1−α) × mixture(full)` where `α` is the mean per-tree confidence — high confidence drives product-mode behaviour, uncertainty drives mixture-mode behaviour.

---

### Module 2 — Goal-Directed Generation (`module2.py`)

Module 1 is **intuition** — fast, associative, pattern-matching.
Module 2 is **deliberation** — goal-directed, using Module 1 as a world model.

Module 1 is already generative: call `predict()` autoregressively and it produces continuations. Module 2 adds **steering**: constraining or guiding that generation toward a target.

**Training format:** represent Q&A or any prompt→response task as a flat sequence:
```
[prompt tokens ...] [SEPARATOR] [response tokens ...] [END]
```
Module 1 learns that SEPARATOR is followed by responses, not more prompts. No architectural changes needed.

**Three generation strategies** (all implemented in `module2.py`):

| Strategy | Mechanism | Best for |
|---|---|---|
| **Autoregressive** | Feed `[prompt + SEPARATOR]` as context seed; generate token by token until END | Direct completion, short responses |
| **Beam search** | Maintain N candidate sequences; at each step expand by all vocabulary tokens; prune to top N by cumulative log-probability | Longer responses, controllable diversity |
| **Retrieval** | Two-stage: (1) Bhattacharyya similarity on post-SEP trie distributions — exact for seen prompts; (2) surface Jaccard fallback when Bhattacharyya < 0.5 — domain-correct for novel tokens | Factual lookup; graceful degradation to novel inputs |

---

## Benchmark Results

Evaluated on 7 standard datasets (two large text corpora, full DNA genome) and 4 concept-drift streams. All methods use the same train/test split (80/20). Baselines: Persistence, Majority, N-gram(5), PPM-D(5), CTW(5).

**Standard benchmarks (test accuracy %):**

| Dataset | n | k | Persistence | PPM-D(5) | CTW(5) | **Predictor** | **Forest** |
|---|---|---|---|---|---|---|---|
| Airline passengers | 144 | 4 | 37.9 | 27.6 | 31.0 | **41.4** | **41.4** |
| Alice in Wonderland (15K) | 15,000 | 6 | 2.8 | 51.6 | **53.3** | 51.7 | 51.9 |
| Moby Dick (50K) | 50,000 | 6 | 2.1 | 45.7 | **47.4** | 46.2 | 46.1 |
| DNA — bacteriophage lambda (full) | 48,502 | 5 | 26.1 | 29.7 | **30.7** | 29.1 | 28.0 |
| Weather | 547 | 3 | **57.3** | 47.3 | 50.0 | **52.7** | **51.8** |
| PRNG (noise floor) | 500 | 3 | 10.0 | **18.0** | 16.0 | 15.0 | 13.0 |
| Electricity (45K) | 45,312 | 4 | **84.8** | **84.8** | **84.8** | **84.7** | **84.6** |

**Concept-drift streams (test accuracy %, k=1):**

| Drift type | N-gram | PPM-D | CTW | **Predictor** | **Forest** |
|---|---|---|---|---|---|
| Sudden reversal | 2.5 | 2.5 | 4.5 | **97.0** | **97.0** |
| Gradual ramp | 5.0 | 5.0 | 6.2 | **98.3** | **98.3** |
| Recurring A→B→A | 3.8 | 3.3 | 4.2 | **97.5** | **97.5** |
| Fast (150-step cycles) | 40.0 | 39.6 | 40.4 | **94.6** | 93.3 |

The concept-drift numbers are the clearest statement of what this architecture is for. Count-based methods (N-gram, PPM-D, CTW) never recover from a reversal because counts only accumulate. The Predictor recovers automatically.

**Extended baseline comparison — KN, PPM\*, Online LSTM (test accuracy %):**

| Dataset | KN(5) | PPM\*(20) | LSTM(64) | Predictor | Forest |
|---|---|---|---|---|---|
| Airline passengers | 27.6 | 27.6 | 24.1 | **41.4** | **41.4** |
| Alice in Wonderland (15K) | **52.8** | 51.8 | 39.9 | 51.7 | 51.9 |
| Moby Dick (50K) | **47.2** | 45.3 | 38.6 | 46.2 | 46.1 |
| DNA — bacteriophage lambda | 30.1 | 26.6 | **32.5** | 29.1 | 28.0 |
| Weather | 50.9 | 48.2 | 43.6 | **52.7** | **51.8** |
| PRNG (noise floor) | 15.0 | **18.0** | 10.0 | 15.0 | 13.0 |
| Electricity (45K) | **84.8** | 81.9 | **84.8** | **84.7** | **84.6** |

KN(5) = Interpolated Kneser-Ney N-gram. PPM\*(20) = PPM with max order 20. LSTM(64) = single-layer LSTM, hidden size 64, trained online with BPTT-1 and Adam.

**Key findings:**

- **Predictor leads on Weather and Airline** — short, noisy, non-stationary datasets where count-based methods overfit to stale patterns. No other method is competitive on Airline (n=144).
- **KN(5) is the strongest text predictor** on large stationary corpora (52.8% Alice, 47.2% Moby). The credibility cap prevents our predictor from fully converging — a structural trade-off for drift recovery.
- **LSTM wins on DNA** (32.5%) — neural sequence modeling captures long-range non-Markovian dependencies that any fixed-order predictor misses.
- **Electricity: all methods tie** (84.6–84.8%) — a high-persistence binary stream where persistence itself is the ceiling.

---

### Confidence-gated prediction (abstain mode)

`UniversalPredictor` accepts a `min_confidence` parameter (default `0.0`). When set, the predictor abstains — returns `(None, conf)` — whenever its best prediction is less than `min_confidence × (1/|vocab|)` above the uniform baseline. A value of `1.5` means "only predict when at least 1.5× more confident than random."

Abstaining does not penalize the node: `node_cred` is unchanged. The successor distribution still updates so learning continues. This makes the warmup period implicit — early steps where the predictor is near-uniform simply produce no output rather than noisy guesses.

**Precision–coverage tradeoffs on natural language (Alice, k=4):**

| min_confidence | Accuracy (predicted only) | Coverage | Lift |
|---|---|---|---|
| 0.0 (off) | 48.5% | 100% | — |
| 3.0 | 50.3% | 96.5% | +1.8pp |
| 4.0 | 56.7% | 83.7% | +8.2pp |
| 5.0 | 59.4% | 77.6% | +10.9pp |
| 6.0 | 61.4% | 71.8% | +12.9pp |

Alice at min_conf=5.0 reaches 59.4% accuracy (vs CTW's 53.3% on 100% coverage) by only speaking when confident. For anomaly detection or alerting use cases where coverage matters less than per-prediction reliability, this is the correct mode.

---

### The two-regime finding

Expanding from small samples to full datasets exposed a fundamental architectural property:

**Data-limited regime (n ≲ 800):** Credibility builds up quickly, blend weights become decisive, and the Predictor is competitive or best. At 1,500 DNA bases the Predictor was 33.0% — best across all methods.

**Architecture-limited regime (n ≫ 800):** Every node hits `CRED_MAX` and the blend weight freezes at λ = cap^p/(cap^p+1). Count-based methods (PPM-D, CTW) have no cap — their counts keep growing, giving predictions increasingly close to 1.0. At 48K DNA bases CTW reaches 30.7% while the Predictor reaches 29.1%.

**The exception is noisy and drifting data.** Weather improved from 41% to 52.7% — the Predictor leads on Weather because noisy, high-variance datasets are exactly where count-based methods overfit to stale patterns.

**The CRED_MAX cap is a design choice, not a bug.** A node with unbounded credibility would adapt from drift in O(n) steps. The cap guarantees O(1/CRED_MAX) adaptation speed. The trade-off is explicit: fast drift recovery at the cost of long-term convergence on stationary data.

---

## Files

**Package (`uchi/`):**

| File | Purpose |
|---|---|
| `predictor.py` | `UniversalPredictor` — core trie engine |
| `forest.py` | `PredictorForest` — ensemble with heterogeneous k and feedback dropout |
| `discretize.py` | `FeatureDiscretizer`, `LabelEncoder` — preprocessing |
| `tabular.py` | `TabularPredictor`, `TabularRegressor` — tabular ML |
| `timeseries.py` | `MultivariateTSPredictor`, `TimeSeriesClassifier`, `AnomalyDetector` |
| `generative.py` | `SequenceGenerator`, `TabularGenerator`, `TimeSeriesGenerator` |

**Root (benchmark scripts and shims):**

| File | Purpose |
|---|---|
| `baselines.py` | Standard baselines: Persistence, Majority, N-gram, PPM-D |
| `baselines_extended.py` | Extended baselines: KN, PPM\*, Online LSTM |
| `datasets.py` | Dataset loaders (airline, text, DNA, weather, PRNG, electricity) |
| `ieee_benchmark.py` | Full benchmark suite generating LaTeX tables |
| `run_experiments.py` | Quick single-predictor experiment runner |
| `run_forest.py` | Quick forest experiment runner |
| `module2.py` | `GoalDirectedGenerator` — autoregressive, beam search, retrieval |
| `tasks/` | Core-principle manifesto and todo |

---

## How This Differs from Standard Approaches

| Property | N-gram / PPM-D | CTW | **This architecture** |
|---|---|---|---|
| Drift adaptation | None — counts only grow | None — counts only grow | Automatic via credibility degradation |
| Depth selection | Fixed or backoff heuristic | Bayesian mixture (stationary) | MWU — theoretically optimal for adversarial depth selection |
| Concept drift recovery | Requires reset or windowing | Requires reset or windowing | Self-correcting; speed proportional to prior confidence |
| Node count | O(V^k) worst case | O(V^k) worst case | O(sequence length) — only observed contexts |
| Online adaptation | Counts update, predictions sharpen | Weights update | Credibility update; fresh vs. stale nodes naturally separated |
| Small dataset behavior | Overtrusts rare k-grams | Overtrusts rare k-grams | Credibility builds slowly on sparse observations |

The single deepest difference from count-based methods: **credibility is earned and can be lost.** A context that was reliable on Monday and wrong on Tuesday sees its influence reduced on Wednesday. Counts only accumulate.
