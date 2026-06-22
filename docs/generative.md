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
