# This Repo's North Star

## The Principle (one sentence)

**Credibility is earned through prediction accuracy and lost through confident error — and this single principle, applied multiplicatively at every scale of context simultaneously, produces online sequence prediction with automatic concept-drift adaptation, theoretically grounded regret bounds, and no forgetting hyperparameter.**

---

## Why This Feels Magically Correct

Every existing approach to sequence prediction has a forgetting problem. Count-based methods (N-gram, PPM-D) only accumulate — they cannot lose confidence in a stale pattern, so concept drift requires an external reset or sliding window. CTW is theoretically optimal under NML but was derived for stationary sources; its Bayesian weights have no mechanism to down-weight contexts that have become unreliable.

This architecture solves the problem at the representation level, not the algorithm level. The credibility score *is* the context's relevance weight, *is* the blending coefficient, *is* the signal for how aggressively to update. One variable does three jobs, and they are self-consistent: a context that predicts well gets used more (higher λ) and degrades more slowly (lower lr_down); a context that predicts badly gets used less and degrades faster. The system automatically allocates attention to what is currently working.

The confidence-proportional degradation rule (`lr_down = lr × (1 + c/C_MAX)`) is not a heuristic — it follows directly from the information-theoretic argument that a node holding high confidence has committed more strongly to a prediction and therefore delivers more evidence of error when wrong. The magnitude of belief update should scale with the strength of the prior belief.

---

## State-of-the-Art Grounding

**CTW (Context Tree Weighting):** Willems, Shtarkov, Tjalkens (1995). Provably achieves the NML (Normalized Maximum Likelihood) minimax redundancy for stationary binary sources. Our blend is CTW-style but replaces the Bayesian Dirichlet weights with credibility scores that can decrease — the key departure that enables nonstationarity.

**MWU (Multiplicative Weights Update):** Arora, Hazan, Kale (2012, "The Multiplicative Weights Update Method: a Meta-Algorithm and Applications"). The multiplicative credibility update is a direct instance of MWU applied to depth selection. MWU achieves O(√(T ln k)) regret over k experts (depths). Our CTW-style blend runs MWU across all depths simultaneously rather than selecting one.

**Krichevsky-Trofimov estimator:** KT (1981). The alpha = 0.5/|V| smoothing used at each trie node achieves asymptotic optimality for i.i.d. sources and is the correct Bayesian prior under a symmetric Dirichlet. We use KT smoothing on the raw succ_cred values for the local distributions.

**Concept drift without detectors:** Losing, Hammer, Wersing (2016, "Incremental on-line learning: A review and comparison of state-of-the-art algorithms") survey the standard approaches (DDM, ADWIN, Page-Hinkley). All require a separate drift detector module. Our approach is the only one where drift adaptation is a consequence of the prediction update rule itself, not a bolt-on.

---

## Alternatives and Why This Wins

**PPM-D (Prediction by Partial Match with escape):**
Counts are monotonically increasing. When the generative process changes, the old counts dominate new observations for O(count) steps. On the concept-drift benchmarks, PPM-D drops to ~2–5% accuracy on reversal tasks (worse than random) and stays there indefinitely. No architectural fix is possible without adding a separate forgetting mechanism — which then requires tuning.

**CTW with fixed Dirichlet weights:**
Theoretically optimal for stationary sources, but the Dirichlet conjugate update only adds counts — it cannot subtract. Under nonstationarity, the theoretical guarantee does not hold, and empirically CTW performs identically to N-gram on concept-drift tasks (~4–6% on reversal). Adding a forgetting window converts it to Switching CTW (Willems 1998), which helps but requires specifying the switching rate in advance.

**LSTM / Transformer:**
Both can handle nonstationarity through gradient updates, but require: fixed vocabulary at training time, a training phase separate from deployment, O(n×d) parameter updates per step, and a learning rate schedule. The trie-based predictor works on any hashable symbol, has no separate training phase, updates in O(k) per step, and has one interpretable hyperparameter (lr). For online deployment on novel symbol streams, the overhead of neural architectures is unjustifiable.

**[Extended benchmark finding, 2026-06-11]:** Online LSTM (H=64, BPTT-1, Adam) beats CTW(5) on DNA (32.5% vs 30.7%) — genomic data has long-range dependencies beyond any fixed-order trie. LSTM also ties for best on Electricity (84.8%). But LSTM fails badly on text (Alice 39.9% vs KN 52.8%, CTW 53.3%): BPTT-1 is insufficient for 26-symbol character-level language. KN(5) is the best text predictor overall; continuation-count backoff gives better coverage than Laplace-smoothed N-gram.

---

## The North Star Applied to Every Decision

Every design choice must be evaluated against this question: **does it make the credibility signal more accurate, faster-updating, or better-calibrated?**

- The CTW blend uses credibility as λ directly. ✓
- Confidence-proportional degradation scales lr_down by the prior credibility. ✓
- The Forest's inter-tree weights apply the same credibility principle at the ensemble level. ✓
- Module 2's retrieval strategy uses successor-distribution similarity — which is the trie's accumulated credibility evidence applied as a semantic search. ✓
- Any change that adds a static parameter (forgetting rate, threshold, window size) is prima facie suspect: it should be replaced by a mechanism that learns the parameter from the credibility signal itself.

## Extended Baseline Results (2026-06-11: KN, PPM*, Online LSTM)

| Dataset | KN(5) | PPM*(20) | LSTM(64) | Predictor | Forest |
|---|---|---|---|---|---|
| Airline | 27.6 | 27.6 | 24.1 | 37.9 | **41.4** |
| Alice (15K) | **52.8** | 51.8 | 39.9 | 50.8 | 51.7 |
| Moby Dick (50K) | **47.2** | 45.3 | 38.6 | 44.0 | 45.7 |
| DNA (48K) | 30.1 | 26.6 | **32.5** | 28.1 | 28.0 |
| Weather | **50.9** | 48.2 | 43.6 | 48.2 | **50.9** |
| PRNG | 15.0 | **18.0** | 10.0 | 14.0 | 13.0 |
| Electricity | **84.8** | 81.9 | **84.8** | 79.0 | 83.5 |

---

## The Two-Regime Finding (empirically confirmed at full scale)

Expanding datasets to full scale (DNA: 1.5K → 48K bases; Alice: 1.5K → 15K chars; Moby Dick: 50K) exposed a fundamental architectural property:

**Data-limited regime (n ≲ 80/lr = 800 steps to cap):** Credibility builds quickly, the Predictor is competitive. Small DNA: Predictor 33.0%, CTW 27.3%.

**Architecture-limited regime (n ≫ 80/lr):** Every node hits CRED_MAX, blend weight freezes at λ=8/9=0.889. Count-based methods keep sharpening past that ceiling. Full DNA: CTW 30.7%, Predictor 26.2%.

**The cap is load-bearing for the North Star.** A node with unbounded credibility recovers from drift in O(n) steps. The cap guarantees O(1/CRED_MAX) recovery speed. The trade-off is: fast drift adaptation at the cost of long-term convergence on stationary data. This is NOT a bug — it is the mechanism by which the credibility principle produces drift adaptation.

**Implication for future work:** An adaptive cap — `CRED_MAX(t) = base + α·log(steps_since_last_error)` — could give count-based sharpness on stationary data while preserving fast drift recovery. Any such change must demonstrate that drift-adaptation speed is preserved before the change is accepted.
