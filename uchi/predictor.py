"""
Universal Sequence Predictor — Credibility-Weighted Context Tree

Architecture
------------
Contexts live in a prefix trie.  Each node in the trie stores a
credibility-weighted distribution over successor symbols.

Prediction  O(k):
    Walk the trie at depths min_k..k, collecting matching context nodes.
    Blend their distributions from shallow to deep using a CTW-style
    recursive mixture:

        λ_d = c_d / (c_d + 1)          # credibility → mixing weight
        P_d = λ_d · P_local(d) + (1−λ_d) · P_{d−1}

    High credibility → λ close to 1 → deep context dominates.
    Low credibility  → λ close to 0 → falls back to shallower.
    Root provides a KT-smoothed unigram as the seed distribution.

Update  O(k):
    For each depth, update the matching node's per-successor credibility
    and the node's overall credibility using a multiplicative rule:

        correct:  c ← min(C_MAX, c × (1 + lr))
        wrong:    c ← max(C_MIN, c × (1 − lr))

    Wrong predictions also boost the correct successor's credibility
    at the same node — the in-trie correction mechanism.

Concept drift:
    Wrong predictions degrade node_cred, reducing λ and causing automatic
    fallback to shallower contexts.  New correct observations rebuild
    credibility for the updated pattern.  No drift detector; no forgetting
    parameter; adaptation speed is a direct function of how confidently the
    wrong pattern was held.

Regret sketch:
    The multiplicative credibility update is the Multiplicative Weights
    Update (MWU) algorithm applied to depth selection.  For the class of k
    single-depth predictors MWU achieves O(√(T ln k)) regret in hindsight.
    The CTW-style blend implements this across all depths simultaneously.
"""

import math
from typing import Any, Callable, Sequence

_CRED_MIN = 0.01
_CRED_MAX = 8.0


class _TrieNode:
    """One node in the credibility-weighted context tree."""
    __slots__ = ['children', 'succ_cred', 'node_cred', 'n_obs', 'last_step']

    def __init__(self):
        self.children:  dict = {}     # symbol → _TrieNode
        self.succ_cred: dict = {}     # symbol → float  (credibility weight)
        self.node_cred: float = 1.0   # reliability of this context as a predictor
        self.n_obs:     int   = 0     # times this context was seen
        self.last_step: int   = 0     # last global step this node was updated

    def prune_stale(self, current_step: int, max_age: int) -> int:
        """Recursively deletes child nodes that haven't been accessed in max_age steps. Returns num deleted."""
        deleted = 0
        stale_keys = []
        for symbol, child in self.children.items():
            if current_step - child.last_step > max_age:
                stale_keys.append(symbol)
            else:
                deleted += child.prune_stale(current_step, max_age)
                
        for key in stale_keys:
            del self.children[key]
            if key in self.succ_cred:
                del self.succ_cred[key]
            deleted += 1
            
        return deleted


class UniversalPredictor:

    def __init__(
        self,
        context_length: int | None,
        similarity_fn: Callable[[Sequence, Sequence], float] | None = None,
        learning_rate: float = 0.1,
        vigilance: float = 0.5,
        min_context_length: int = 1,
        min_confidence: float = 0.0,
        adaptive_cap: bool = False,
        cont_count_min_vocab: int = 8,
        binary_correction_scale: float | None = None,
        cred_max: float = 8.0,
        lambda_power: float = 1.0,
        use_similarity_fallback: bool = False,
        similarity_max_candidates: int = 8,
        use_positional_weights: bool = False,
        compressor: Any = None,
        **kwargs,   # absorb legacy args (coupling_lr, feedback_strength, etc.)
    ):
        self.k              = context_length   # None = infinite context (PPM-Star)
        self.min_k          = max(0, min_context_length)
        self.lr             = learning_rate
        self.vigilance      = vigilance
        self.min_confidence = min_confidence        # abstain if max(blend) < this × (1/|vocab|)
        self._adaptive_cap  = adaptive_cap          # allow CRED_MAX to grow with node observations
        self._cred_max_base = cred_max              # base credibility ceiling (8.0 = default)
        self._cc_min_vocab  = cont_count_min_vocab  # V threshold for cont-count seed
        # For binary streams (V≤2), full in-trie correction causes false-flip cascades
        # after genuine transitions.  Scaling the boost reduces this at the cost of
        # slightly slower cold-start on any other stream that passes through V=2.
        # Only effective for V≤2; V≥3 always gets the full boost.
        self._binary_corr_scale = binary_correction_scale
        # Softens the credibility→mixing-weight mapping: λ = cred^p / (cred^p + 1).
        # p=1 is the standard CTW formula; p<1 lets shallower contexts retain more
        # influence even at high credibility, acting as implicit depth regularization.
        self._lambda_power  = lambda_power
        self._surface_sim   = similarity_fn         # kept for API compat with forest

        # Problem 2: Similarity fallback — when exact context match fails,
        # find trie nodes sharing tokens with the current context.
        self._use_sim_fallback = use_similarity_fallback
        self._sim_max_cand     = similarity_max_candidates

        # Problem 9: Positional weights — track which context positions
        # historically contribute most to correct predictions.
        self._use_pos_weights = use_positional_weights
        if context_length is None:
            self._pos_correct: list[float] = []
            self._pos_total:   list[float] = []
        else:
            self._pos_correct: list[float] = [0.0] * context_length
            self._pos_total:   list[float] = [0.0] * context_length

        # Problem 6: Node compressor — optional two-tier memory management.
        self._compressor = compressor
        self._global_step: int = 0

        self._root:  _TrieNode      = _TrieNode()
        self.history: list[Any]     = []
        self._vocab:  set           = set()

        # Continuation-count unigram (KN-style):
        # _cont_counts[w] = number of distinct 1-gram predecessors u such that
        # bigram (u, w) has been seen at least once.  Used as the seed
        # distribution in _blend() instead of raw KT counts.
        self._cont_counts: dict = {}
        self._seen_bigrams: set = set()

        # predict() → feedback() state
        self._last_prediction:    Any   = None
        self._last_distribution:  dict  = {}
        self._last_context:       list  = []
        self._last_max_sim:       float = 0.0
        self._last_contributions: dict  = {}   # depth → (node_cred, top_successor)
        self._last_abstained:     bool  = False
        self._last_prediction_depth: int = 0

        # Backward-compat stubs (coupling removed; ablation showed ~0 effect)
        self.coupling:          dict  = {}
        self._coupling_counts:  dict  = {}
        self.lam:               float = 0.0

    def prune_stale_branches(self, max_age: int = 100000) -> int:
        """
        LRU Eviction (RAM Optimization): Triggers the pruning of nodes 
        that have not been accessed or updated in `max_age` global steps.
        Returns the total number of nodes pruned.
        """
        if self._root:
            return self._root.prune_stale(self._global_step, max_age)
        return 0

    # ── public interface ──────────────────────────────────────────────────────

    def observe(self, value: Any) -> None:
        self.history.append(value)
        self._vocab.add(value)

    def predict(self) -> tuple[Any, float]:
        if not self._vocab:
            return None, 0.0

        if self.k is None:
            self._last_context = list(self.history)
        else:
            self._last_context = list(self.history[-self.k:]) if self.history else []
            
        active = self._get_active_nodes()
        dist   = self._blend(active)

        if not dist:
            return None, 0.0

        pred = max(dist, key=dist.get)
        conf = dist[pred]

        self._last_distribution  = dist
        self._last_max_sim       = max((n.node_cred for n, _ in active), default=0.0)
        self._last_prediction_depth = max((d for _, d in active), default=0)
        self._last_contributions = {
            d: (n.node_cred,
                max(n.succ_cred, key=n.succ_cred.get) if n.succ_cred else pred)
            for n, d in active
        }

        # Abstain when confidence is below the threshold (expressed as a
        # multiple of the uniform baseline 1/|vocab|).  A factor of 1.5 means
        # "only predict when at least 1.5× more confident than random".
        # Does NOT change learning: feedback() still updates the trie.
        if self.min_confidence > 0.0:
            V = len(self._vocab)
            if conf * V < self.min_confidence:
                self._last_prediction = None
                self._last_abstained  = True
                return None, conf

        self._last_prediction = pred
        self._last_abstained  = False
        return pred, conf

    def feedback(self, actual: Any) -> None:
        self._vocab.add(actual)
        self._global_step += 1
        n_hist    = len(self.history)
        abstained = self._last_abstained
        correct   = (not abstained) and (self._last_prediction == actual)

        # Root stores raw unigram counts (kept for fallback during cold start)
        self._root.succ_cred[actual] = self._root.succ_cred.get(actual, 0) + 1.0
        self._root.n_obs += 1

        # Continuation-count update: when a new bigram (prev, actual) is first
        # seen, increment the continuation count for actual.
        if len(self.history) >= 2:
            bigram = (self.history[-2], actual)
            if bigram not in self._seen_bigrams:
                self._seen_bigrams.add(bigram)
                self._cont_counts[actual] = self._cont_counts.get(actual, 0) + 1

        # Per-depth context nodes (depths min_k .. k)
        max_d = (n_hist - 1) if self.k is None else min(self.k, n_hist - 1)
        
        # To avoid O(N^2) explosion with infinite context, we only create
        # nodes up to the longest existing match + 1.
        node = self._root
        for d in range(self.min_k, max_d + 1):
            ctx  = tuple(self.history[-(d + 1):-1])
            # This logic needs to traverse backwards from the end of history.
            # But ctx is built backwards. Let's stick to _ensure_node for now 
            # and just enforce a reasonable hard cap if k is None to avoid hanging.
            if self.k is None and d > 64: 
                break
                
            node = self._feedback_get_node(ctx)
            if node is None:
                continue
            node.n_obs += 1
            node.last_step = self._global_step
            if actual not in node.succ_cred:
                node.succ_cred[actual] = 1.0

            if abstained:
                self._update_node_abstained(node, actual)
            elif correct:
                self._update_node_correct(node, actual)
            else:
                self._update_node_wrong(node, self._last_prediction, actual)

            # Problem 9: update positional weight tracking
            if self._use_pos_weights and (self.k is None or d <= self.k):
                idx = d - 1  # depth 1 → index 0
                # Extend the pos tracking lists if we are going deeper than before
                if idx >= len(self._pos_total):
                    self._pos_total.extend([0.0] * (idx - len(self._pos_total) + 1))
                    self._pos_correct.extend([0.0] * (idx - len(self._pos_correct) + 1))
                if idx < len(self._pos_total):
                    self._pos_total[idx] += 1.0
                    if correct:
                        self._pos_correct[idx] += 1.0

        # Problem 6: periodic compression pass
        if (self._compressor is not None
                and self._global_step % 500 == 0):
            self._compressor.compress_pass(self._root, self._cred_max_base)

    def unlearn(self, actual: Any, strength: float = 1.0) -> None:
        """
        Surgical Synaptic Pruning with exponential trailing decay.
        
        Instead of uniformly destroying all edge weights, applies an
        exponentially decaying penalty from the END of the sequence
        backward. The trailing tokens (most likely to contain the error)
        receive the heaviest penalty, while foundational tokens at the 
        start are preserved.
        
        Args:
            actual: The sequence (list) or single token to unlearn.
            strength: 0.0-1.0 scalar. 1.0 = hard unlearn (code errors),
                      0.3 = soft unlearn (conversational mistakes).
        """
        if isinstance(actual, list):
            # Surgical sequence unlearning with positional decay
            seq = actual
            n = len(seq)
            for i, token in enumerate(seq):
                # Exponential decay: tokens near the end get the heaviest penalty
                # position_weight ranges from ~0.1 (start) to 1.0 (end)
                position_weight = (i + 1) / n
                decay_factor = 0.1 + (0.9 * (1.0 - position_weight * strength))
                
                # Walk the trie and apply decay at every matching depth
                n_hist = len(self.history)
                max_d = (n_hist - 1) if self.k is None else min(self.k, n_hist - 1)
                for d in range(self.min_k, max_d + 1):
                    if self.k is None and d > 64:
                        break
                    ctx = tuple(self.history[-(d + 1):-1])
                    node = self._feedback_get_node(ctx)
                    if node and token in node.succ_cred:
                        node.succ_cred[token] *= decay_factor
                        if node.succ_cred[token] < 0.05:
                            del node.succ_cred[token]
                self.history.append(token)
            return
        
        # Single-token unlearn (legacy path)
        n_hist = len(self.history)
        max_d = (n_hist - 1) if self.k is None else min(self.k, n_hist - 1)
        decay_factor = 0.1 * strength  # Hard: 0.1, Soft: 0.03
        for d in range(self.min_k, max_d + 1):
            if self.k is None and d > 64:
                break
            ctx = tuple(self.history[-(d + 1):-1])
            node = self._feedback_get_node(ctx)
            if node and actual in node.succ_cred:
                node.succ_cred[actual] *= decay_factor
                if node.succ_cred[actual] < 0.05:
                    del node.succ_cred[actual]



    def _distribution(self) -> dict:
        """Return last predictive distribution (for log-loss evaluation)."""
        return dict(self._last_distribution)

    # ── hooks for subclass ablation ───────────────────────────────────────────

    def _update_node_correct(self, node: _TrieNode, actual: Any) -> None:
        cap = self._effective_cred_max(node)
        node.succ_cred[actual] = min(cap, node.succ_cred[actual] * (1 + self.lr))
        node.node_cred         = min(cap, node.node_cred         * (1 + self.lr))

    def _update_node_abstained(self, node: _TrieNode, actual: Any) -> None:
        if actual not in node.succ_cred:
            node.succ_cred[actual] = 1.0
        cap = self._effective_cred_max(node)
        node.succ_cred[actual] = min(cap, node.succ_cred[actual] * (1 + self.lr))

    def _update_node_wrong(self, node: _TrieNode, predicted: Any, actual: Any) -> None:
        cap = self._effective_cred_max(node)
        # lr_down scales from lr (fresh node) to 2×lr (maximally trusted node).
        lr_down = self.lr * (1.0 + node.node_cred / cap)
        if predicted is not None and predicted in node.succ_cred:
            node.succ_cred[predicted] = max(_CRED_MIN,
                node.succ_cred[predicted] * (1 - lr_down))
        # In-trie correction: immediately boost correct successor.
        # For binary streams (V≤2), reduce the boost to limit false-flip cascades
        # after genuine transitions.  V≥3 always gets full correction.
        if self._binary_corr_scale is not None and len(self._vocab) <= 2:
            eff_boost = self.lr * self._binary_corr_scale
        else:
            eff_boost = self.lr
        node.succ_cred[actual] = min(cap,
            node.succ_cred.get(actual, 1.0) * (1 + eff_boost))
        node.node_cred = max(_CRED_MIN, node.node_cred * (1 - lr_down))

    def _effective_cred_max(self, node: _TrieNode) -> float:
        if not self._adaptive_cap:
            return self._cred_max_base
        return self._cred_max_base * (1.0 + 0.5 * math.log(1.0 + node.n_obs / 100.0))

    def _blend_lambda(self, node_cred: float) -> float:
        """CTW-style mixing coefficient. Override to disable credibility effect."""
        if self._lambda_power == 1.0:
            return node_cred / (node_cred + 1.0)
        x = node_cred ** self._lambda_power
        return x / (x + 1.0)

    def _feedback_get_node(self, ctx: tuple) -> _TrieNode | None:
        """Return (creating if needed) the node for ctx. Override for ablation."""
        return self._ensure_node(ctx)

    # ── internal ──────────────────────────────────────────────────────────────

    def _get_active_nodes(self) -> list[tuple[_TrieNode, int]]:
        """
        Return [(node, depth)] for matching context depths min_k..k.
        Includes similarity fallback when exact match fails (Problem 2).
        """
        result = []
        max_d = min(64, len(self.history)) if self.k is None else min(self.k, len(self.history))
        for d in range(self.min_k, max_d + 1):
            ctx  = tuple(self.history[-d:])
            node = self._walk(ctx)
            if node is not None and node.succ_cred:
                result.append((node, d))
            elif self._use_sim_fallback and d >= 2:
                # Problem 2: similarity fallback — find nodes sharing tokens
                sim_nodes = self._similarity_fallback(ctx, d)
                result.extend(sim_nodes)
            # Problem 6: check compressed nodes
            if node is None and self._compressor is not None:
                comp = self._compressor.get_compressed(ctx)
                if comp is not None:
                    # Create a lightweight proxy node from the compressed distribution
                    proxy = _TrieNode()
                    proxy.node_cred = comp.node_cred
                    proxy.n_obs = comp.n_obs
                    total = sum(comp.distribution.values()) or 1.0
                    proxy.succ_cred = {
                        k: v / total * comp.node_cred
                        for k, v in comp.distribution.items()
                    }
                    result.append((proxy, d))
        return result

    def _walk(self, ctx: tuple) -> _TrieNode | None:
        node = self._root
        for sym in ctx:
            if sym not in node.children:
                return None
            node = node.children[sym]
        return node

    def _ensure_node(self, ctx: tuple) -> _TrieNode:
        node = self._root
        for sym in ctx:
            if sym not in node.children:
                node.children[sym] = _TrieNode()
            node = node.children[sym]
        return node

    def _blend(self, active: list[tuple[_TrieNode, int]]) -> dict:
        """
        CTW-style credibility-weighted blend, shallow to deep.
        Uses KT prior (alpha = 0.5/|V|) at each node for smoothing.
        Incorporates positional weights when enabled (Problem 9).
        """
        if not self._vocab:
            return {}

        V     = len(self._vocab)
        alpha = 0.5 / V    # Krichevsky-Trofimov prior

        # Seed: continuation-count unigram (KN-style) for |V| >= cont_count_min_vocab.
        # Uses how many distinct 1-gram predecessors each symbol appeared after,
        # rather than raw frequency.  Better calibrated for diverse vocabularies
        # (text ~26+, Airline 8 bins).  Threshold keeps small alphabets
        # (DNA=4, Electricity=2) on raw KT where cont-counts are too sparse.
        cont_total = sum(self._cont_counts.values()) if self._cont_counts else 0
        if V >= self._cc_min_vocab and cont_total > 0:
            blended = {
                s: (self._cont_counts.get(s, 0) + alpha) / (cont_total + alpha * V)
                for s in self._vocab
            }
        else:
            root_total = sum(self._root.succ_cred.values()) or 1.0
            blended = {
                s: (self._root.succ_cred.get(s, 0) + alpha) / (root_total + alpha * V)
                for s in self._vocab
            }

        # Compute positional weight multipliers (Problem 9)
        pos_multiplier = None
        if self._use_pos_weights:
            pos_multiplier = self._positional_multipliers()

        by_depth = {d: n for n, d in active}
        max_d = self.k if self.k is not None else (max(by_depth.keys()) if by_depth else 0)
        for d in range(1, max_d + 1):
            if d not in by_depth:
                continue
            node  = by_depth[d]
            total = sum(node.succ_cred.values()) or 1.0
            local = {
                s: (node.succ_cred.get(s, 0) + alpha) / (total + alpha * V)
                for s in self._vocab
            }
            lam     = self._blend_lambda(node.node_cred)
            # Problem 9: scale lambda by positional weight
            if pos_multiplier is not None and d - 1 < len(pos_multiplier):
                lam = min(1.0, lam * pos_multiplier[d - 1])
            blended = {s: lam * local[s] + (1 - lam) * blended[s]
                       for s in self._vocab}

        total = sum(blended.values())
        if total < 1e-12:
            return {s: 1.0 / V for s in self._vocab}
        return {s: v / total for s, v in blended.items()}

    # ── Problem 2: similarity fallback ─────────────────────────────────────────

    def _similarity_fallback(
        self, ctx: tuple, depth: int,
    ) -> list[tuple[_TrieNode, int]]:
        """
        When exact context match fails at depth d, find trie nodes sharing
        tokens with the current context and blend their distributions
        weighted by token-overlap (Jaccard).
        """
        candidates = []
        ctx_set = set(ctx)

        # Try progressively shorter prefixes to find a branch point
        for trim in range(1, len(ctx)):
            prefix = ctx[trim:]
            branch = self._walk(prefix)
            if branch is None or not branch.children:
                continue
            # Enumerate children of this branch
            for sym, child in branch.children.items():
                if not child.succ_cred:
                    continue
                # Build the full context this child represents
                child_ctx_set = set(prefix) | {sym}
                # Jaccard overlap
                union = len(ctx_set | child_ctx_set)
                overlap = len(ctx_set & child_ctx_set) / union if union > 0 else 0.0
                if overlap > 0.0:
                    candidates.append((child, depth, overlap))
            if candidates:
                break  # found matches at this trim level

        if not candidates:
            return []

        # Keep top candidates by overlap score
        candidates.sort(key=lambda x: x[2], reverse=True)
        top = candidates[:self._sim_max_cand]

        # Create a blended proxy node from the top candidates
        if len(top) == 1:
            return [(top[0][0], top[0][1])]

        # Blend multiple candidates into a single proxy
        proxy = _TrieNode()
        total_weight = sum(c[2] for c in top)
        for node, d, w in top:
            weight = w / total_weight
            for sym, cred in node.succ_cred.items():
                proxy.succ_cred[sym] = proxy.succ_cred.get(sym, 0.0) + cred * weight
            proxy.node_cred += node.node_cred * weight
            proxy.n_obs += int(node.n_obs * weight)

        return [(proxy, depth)]

    # ── Problem 9: positional weight helpers ──────────────────────────────────

    def _positional_multipliers(self) -> list[float]:
        """
        Return per-depth multipliers based on historical accuracy contribution.
        Positions that historically helped more get multiplier > 1.0.
        """
        weights = []
        for i in range(self.k):
            if self._pos_total[i] > 0:
                acc = self._pos_correct[i] / self._pos_total[i]
            else:
                acc = 0.5  # neutral prior
            weights.append(acc)

        mean_w = sum(weights) / len(weights) if weights else 1.0
        if mean_w < 1e-12:
            return [1.0] * self.k
        return [w / mean_w for w in weights]

    # ── Problem 6: compression helpers ────────────────────────────────────────

    def compress_pass(self) -> dict:
        """Run a compression pass on the trie. Returns stats dict."""
        if self._compressor is None:
            return {'compressed': 0, 'skipped': 0, 'active': 0}
        return self._compressor.compress_pass(self._root, self._cred_max_base)

    def memory_stats(self) -> dict:
        """Return memory usage stats including compressed nodes."""
        active = len(self._nodes)
        compressed = 0
        if self._compressor is not None:
            compressed = self._compressor.stats().get('n_compressed', 0)
        return {
            'active_nodes': active,
            'compressed_nodes': compressed,
            'total_nodes': active + compressed,
            'global_step': self._global_step,
        }

    # ── backward-compat API (used by forest.py and diagnostics) ──────────────

    def sim(self, ctx_a: Sequence, ctx_b: Sequence) -> float:
        """Surface similarity — kept for forest API compat."""
        if self._surface_sim is not None:
            try:
                return float(self._surface_sim(ctx_a, ctx_b))
            except Exception:
                pass
        return 1.0 if list(ctx_a) == list(ctx_b) else 0.0

    @property
    def max_k(self) -> int:
        return self.k

    @property
    def _nodes(self) -> list:
        """All trie nodes as a flat list (for node-count reporting)."""
        result = []
        stack  = [self._root]
        while stack:
            n = stack.pop()
            result.append(n)
            stack.extend(n.children.values())
        return result

    def node_stats(self) -> dict:
        nodes = self._nodes
        total = len(nodes)
        return {
            'total_nodes':           total,
            'observed':              total,
            'exploration':           0,
            'correction':            0,
            'coupling_links':        0,
            'mean_coupling':         0.0,
            'max_coupling':          0.0,
            'lambda':                0.0,
            'optimizer_budget':      self.k,
            'optimizer_rolling_acc': 0.0,
            'allocator_trials':      sum(n.n_obs for n in nodes),
        }

    def similarity_quality(self) -> float:
        nodes = [n for n in self._nodes if n.node_cred != 1.0]
        if not nodes:
            return 1.0
        creds = sorted((n.node_cred for n in nodes), reverse=True)
        top_n = max(1, len(creds) // 4)
        return sum(creds[:top_n]) / top_n

    def convergence_state(self) -> dict:
        nodes = self._nodes
        if not nodes:
            return {'plateau': None, 'tau': None, 'quality_now': 0.0,
                    'steps_to_95pct': None, 'converged': False}
        quality = sum(n.node_cred for n in nodes) / len(nodes)
        return {'plateau': quality, 'tau': None, 'quality_now': quality,
                'steps_to_95pct': None, 'converged': False}

    def lookahead_quality(self, n_steps: int) -> float:
        return self.convergence_state()['quality_now']
