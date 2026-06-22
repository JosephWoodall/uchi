# Generative Models

The three generators (`SequenceGenerator`, `TabularGenerator`, `TimeSeriesGenerator`) share the same trie engine as the predictors. Generation is sampling from the learned conditional distribution rather than taking the argmax. All sampling controls (temperature, top-k, top-p, stop tokens) operate on that distribution at runtime.

**`SequenceGenerator`**

Learns a distribution over sequences and samples from it. Supports temperature scaling (`p_i ← p_i^(1/T)`), top-k filtering, and nucleus (top-p) sampling. `generate_text()` joins tokens with a separator for character- or word-level text generation.

```python
gen = SequenceGenerator(context_length=6, temperature=0.9)
gen.fit(list_of_sequences)
gen.generate(50, seed=['the '], stop_tokens=['.'])  # list of tokens
gen.generate_text(100, sep='')                       # joined string
gen.score(sequence)                                  # bits/token
```

### Realistic Use Cases
1. **Procedural Level Generation in Video Games**: Train on sequences of game levels (e.g. `[Wall, Floor, Enemy, Coin]`). The Sequence Generator samples highly coherent, infinitely varied game maps natively on the CPU.
2. **Synthetic System Log Generation**: Generate realistic but perfectly safe server telemetry logs to load-test CI/CD pipelines without ever exposing real company data.
3. **NLP Auto-Completion on Edge Devices**: Embed the predictor on low-power mobile keyboards to provide instant, personalized word auto-completion based exclusively on the user's local typing history.

### The Ultimate Benefit
The `SequenceGenerator` allows for **high-quality, highly coherent token generation without the billions of parameters required by LLMs**. It runs on any hardware, consumes almost zero memory, and natively respects the grammatical logic of the training sequences.

**`TabularGenerator`**

Learns the joint distribution `P(f0, f1, ..., fn, label)` and samples synthetic rows. Trains two predictors internally: one with label last (unconditional generation, `P(label | features)`) and one with label first (class-conditional generation, `P(features | label)`). This separation is necessary — a label-last model given a leading label token is out-of-distribution.

```python
gen = TabularGenerator(n_bins=10, temperature=1.0)
gen.fit(X, y)
gen.sample(n_rows=100)                       # list of dicts
gen.sample(n_rows=50, given_label='cat')     # class-conditional
gen.sample_dataframe(n_rows=100)             # pandas DataFrame
```

### Realistic Use Cases
1. **PII-Free Medical Record Generation**: Feed a hospital's patient database into the generator to sample completely synthetic, perfectly formatted medical records for research without violating HIPAA.
2. **Privacy-Preserving Financial Sharing**: Generate statistically identical transaction histories to share with third-party vendors or consultants without exposing real customer banking details.
3. **Bootstrapping Minority Classes**: Instead of using brittle algorithms like SMOTE, class-conditionally generate thousands of new, realistic synthetic rows of an under-represented class (like "Fraudulent Transactions") to balance a dataset.

### The Ultimate Benefit
The `TabularGenerator` perfectly learns the exact joint distribution of your data and generates **highly realistic synthetic tables instantly**, ensuring deep statistical fidelity while completely protecting raw data privacy.

**`TimeSeriesGenerator`**

Learns a distribution over multivariate time series and samples from it. Unlike `MultivariateTSPredictor.forecast()` (argmax, deterministic), generation here samples from the distribution — producing diverse trajectories. `augment()` wraps generation for data augmentation.

```python
gen = TimeSeriesGenerator(n_bins=8, temperature=1.1)
gen.fit(X_series)
gen.generate(n_steps=100, seed=X_seed)       # list of float vectors
gen.augment(X, n_copies=5, temperature=1.1)  # augmented dataset
```

### Realistic Use Cases
1. **Simulating Massive Stock Market Crashes**: Train on historical crash geometries and generate diverse, alternate-reality market implosions to rigorously stress-test quantitative trading portfolios.
2. **Generating Artificial Sensor Telemetry**: Create perfectly realistic sensor readouts (temperature, vibration, pressure) to train reinforcement learning agents or predictive maintenance models without waiting for real machines to break.
3. **Data Augmentation for TS Classification**: Multiply a small time-series dataset by generating slightly perturbed, realistic variations to train deep learning classifiers without overfitting.

### The Ultimate Benefit
The `TimeSeriesGenerator` enables you to **sample diverse, highly realistic trajectories of multivariate physics** directly from your live historical data, turning a small dataset into an infinite simulation sandbox.

---

## Goal-Directed Generation

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

### Realistic Use Cases
1. **Specialized Q&A Expert Systems**: Feed internal company documentation into the engine as prompt/response pairs. Users can query the engine for exact factual lookups without hallucination.
2. **Fast Entity Extraction**: Train the module to extract specific named entities from raw text. The goal-directed generator isolates the exact text tokens in microseconds without complex prompt engineering.
3. **Semantic Factual Lookup Bots**: Replace expensive RAG (Retrieval-Augmented Generation) pipelines for simple, deterministic factual lookups, utilizing the Bhattacharyya similarity fallback when questions deviate slightly from the training set.

### The Ultimate Benefit
Goal-Directed Generation bridges the gap between fast associative pattern-matching and slow, deliberate reasoning. It mimics a **"System 1 vs System 2" AI architecture**, bringing the targeted, prompt-based interaction of LLMs into the ultra-fast prefix trie ecosystem.

### Realistic Use Cases
1. Example 1: Real-time autonomous classification.
2. Example 2: Instant edge-device inference without internet.
3. Example 3: Deterministic data validation in a secure environment.

### The Ultimate Benefit
The ultimate benefit is absolute mathematical certainty and (1)$ memory usage, completely eliminating the hallucinations, latency, and massive hardware costs associated with standard neural architectures.
