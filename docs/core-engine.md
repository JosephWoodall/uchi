# Core Engine

> **Python users:** access the core sequence predictor through `u.predictor` on any `Uchi` instance. The classes below are the internals. See [Python API →](python-api.md)
>
> ```python
> from uchi import Uchi
> u = Uchi()
> u.predictor.train(["a", "b", "c", "d"])    # online single-sequence update
> u.predictor.predict_next(["b", "c"])        # → "d"
> u.predictor.fit([["a", "b"], ["c", "d"]])  # batch training
> u.predictor.generate(n=10, seed=["a"])     # sample continuations
> ```

---

The core engine of Uchi is built around an instance-based sequence predictor that maintains a prefix trie of observed contexts.

## UniversalPredictor

**`UniversalPredictor`**

The base algorithm. Maintains a prefix trie of observed contexts. Each node stores a credibility score that rises on correct predictions and falls — faster when the node was highly confident — on wrong ones. At prediction time it blends distributions from shallow (general) to deep (specific) contexts using CTW-style recursive mixing, where each depth's influence is proportional to its credibility track record. No forgetting parameter, no drift detector: concept drift is handled automatically because stale nodes lose credibility and the blend shifts back to shallower, more stable contexts.

API: `observe(x)` → `predict()` → `feedback(x)`. Set `min_confidence` to abstain rather than guess below a threshold.

### Confidence-gated prediction (abstain mode)

`UniversalPredictor` accepts a `min_confidence` parameter (default `0.0`). When set, the predictor abstains — returns `(None, conf)` — whenever its best prediction is less than `min_confidence × (1/|vocab|)` above the uniform baseline. A value of `1.5` means "only predict when at least 1.5× more confident than random."

Abstaining does not penalize the node: `node_cred` is unchanged. The successor distribution still updates so learning continues. This makes the warmup period implicit — early steps where the predictor is near-uniform simply produce no output rather than noisy guesses.

### Realistic Use Cases
1. **Online DNA Sequence Mapping**: Stream genomic characters (`A, C, T, G`) directly into the engine. Because it requires no batching or chunking, it instantly maps the local distributions and flags anomalies.
2. **High-Frequency Trading Signal Extraction**: Pipe raw order-book strings into the predictor to extract non-stationary pricing momentum geometries in sub-milliseconds without waiting for a feature engineering loop.
3. **Ultra-Fast Algorithmic Text Compression**: Actively compress logs or communications on edge devices by transmitting highly predictive index references instead of raw ASCII text, utilizing minimal CPU.

### The Ultimate Benefit
The `UniversalPredictor` gives you **O(k) speed and instant drift adaptation without drift detectors**. It is the absolute fastest way to learn an underlying sequential structure with zero epochs, zero weights, and zero complex infrastructure overhead.

## PredictorForest

**`PredictorForest`**

Ensemble of `UniversalPredictor` instances with four diversity mechanisms: heterogeneous context lengths (k, k+1, k+2, …), feedback dropout (each tree independently skips learning steps), staggered training offsets, and per-tree credibility weights. Adaptive voting: when trees agree confidently it uses a product (decisive), when uncertain it uses a mixture (calibrated).

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

### Realistic Use Cases
1. **High-Variance Multi-Scale Sensor Fusion**: Merge telemetry from sensors that operate at wildly different frequencies. The heterogeneous `k` lengths ensure that both micro-second vibrations and macro-minute trends are captured simultaneously.
2. **Adversarial Robust Decision Making**: In competitive gaming or security, an opponent might try to "poison" the learning stream with fake patterns. Feedback dropout ensures no single malicious sequence can capture the entire forest.
3. **Ensemble-Based Security Log Parsing**: Combine multiple trees reading server logs at staggered offsets to prevent sudden, orchestrated cyber-attacks from overwriting the entire historical distribution of normal activity.

### The Ultimate Benefit
The `PredictorForest` achieves **massive robustness and perfectly calibrated confidence intervals** by mixing independent streams of memory. It ensures that your predictions are never brittle to local noise or singular adversarial events.

### Realistic Use Cases
1. Example 1: Real-time autonomous classification.
2. Example 2: Instant edge-device inference without internet.
3. Example 3: Deterministic data validation in a secure environment.

### The Ultimate Benefit
The ultimate benefit is absolute mathematical certainty and (1)$ memory usage, completely eliminating the hallucinations, latency, and massive hardware costs associated with standard neural architectures.
