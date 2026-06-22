# The Core Idea

The algorithm keeps a trie of contexts. Every node in the trie stores two things: a credibility-weighted distribution over successor symbols, and a track record of how reliable that context has been as a predictor. When predicting, it blends the distributions from shallow (general) to deep (specific), where each depth's influence is proportional to its track record. When updating after a wrong prediction, a node that was confidently wrong loses trust faster than one that was fresh and uncertain.

That is the entire algorithm. No drift detector, no forgetting parameter, no domain-specific tuning.

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

---

## Architectural Details

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

## The Two-Regime Finding

Expanding from small samples to full datasets exposed a fundamental architectural property:

**Data-limited regime (n ≲ 800):** Credibility builds up quickly, blend weights become decisive, and the Predictor is competitive or best. At 1,500 DNA bases the Predictor was 33.0% — best across all methods.

**Architecture-limited regime (n ≫ 800):** Every node hits `CRED_MAX` and the blend weight freezes at λ = cap^p/(cap^p+1). Count-based methods (PPM-D, CTW) have no cap — their counts keep growing, giving predictions increasingly close to 1.0. At 48K DNA bases CTW reaches 30.7% while the Predictor reaches 29.1%.

**The exception is noisy and drifting data.** Weather improved from 41% to 52.7% — the Predictor leads on Weather because noisy, high-variance datasets are exactly where count-based methods overfit to stale patterns.

**The CRED_MAX cap is a design choice, not a bug.** A node with unbounded credibility would adapt from drift in O(n) steps. The cap guarantees O(1/CRED_MAX) adaptation speed. The trade-off is explicit: fast drift recovery at the cost of long-term convergence on stationary data.
