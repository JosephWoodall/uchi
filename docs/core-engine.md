# Core Engine

The core engine of Uchi is built around an instance-based sequence predictor that maintains a prefix trie of observed contexts.

## UniversalPredictor

**`UniversalPredictor`**

The base algorithm. Maintains a prefix trie of observed contexts. Each node stores a credibility score that rises on correct predictions and falls — faster when the node was highly confident — on wrong ones. At prediction time it blends distributions from shallow (general) to deep (specific) contexts using CTW-style recursive mixing, where each depth's influence is proportional to its credibility track record. No forgetting parameter, no drift detector: concept drift is handled automatically because stale nodes lose credibility and the blend shifts back to shallower, more stable contexts.

API: `observe(x)` → `predict()` → `feedback(x)`. Set `min_confidence` to abstain rather than guess below a threshold.

### Confidence-gated prediction (abstain mode)

`UniversalPredictor` accepts a `min_confidence` parameter (default `0.0`). When set, the predictor abstains — returns `(None, conf)` — whenever its best prediction is less than `min_confidence × (1/|vocab|)` above the uniform baseline. A value of `1.5` means "only predict when at least 1.5× more confident than random."

Abstaining does not penalize the node: `node_cred` is unchanged. The successor distribution still updates so learning continues. This makes the warmup period implicit — early steps where the predictor is near-uniform simply produce no output rather than noisy guesses.

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
