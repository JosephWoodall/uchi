import math
import random
from collections import defaultdict
from typing import Any, Callable, Sequence

from .predictor import UniversalPredictor


class PredictorForest:
    """
    A forest of UniversalPredictor instances that diverge through:
      1. Heterogeneous k       — tree i uses context_length + i
      2. Feedback dropout      — each tree independently skips learning steps
      3. Staggered offsets     — tree i defers learning for i * stagger steps
      4. Inter-tree credibility — trees weighted by recent track record

    Dynamic sizing
    --------------
    auto_grow  — spawn a new tree (k = current_max_k + 1) after grow_threshold
                 consecutive steps of unanimous correlated failure across all
                 active trees (all active trees predicted the same wrong answer).
                 Capped at max_trees.

    auto_prune — deactivate a tree after its inter-tree credibility stays below
                 prune_floor × mean_active_credibility for prune_window consecutive
                 steps.  At least 2 active trees are always preserved.

    Voting modes
    ------------
    'mixture'  — confidence × credibility weighted sum of distributions
    'product'  — weighted geometric mean (agreement required to win)
    'adaptive' — α·product + (1-α)·mixture where α = mean confidence of active
                 trees.  Automatically selects product when trees are certain,
                 mixture when uncertain.  Default.

    Task types
    ----------
    'sequence' / 'classification'
                 Predict the most probable next successor (argmax of blended
                 distribution).  Default.
    'regression'
                 Successors are numeric.  predict() returns the credibility-
                 weighted mean of the blended successor distribution together
                 with a peakedness-based confidence.  Useful for discretised
                 numeric series where you want a continuous-valued output.
    """

    def __init__(
        self,
        context_length: int,
        similarity_fn: Callable[[Sequence, Sequence], float] | None = None,
        learning_rate: float = 0.1,
        coupling_lr: float = 0.3,
        feedback_strength: float = 0.3,
        vigilance: float = 0.7,
        min_context_length: int = 1,
        coupling_ema: bool = True,
        n_trees: int = 5,
        dropout: float = 0.2,
        binary_correction_scale: float | None = None,
        seed: int = 42,
        voting: str = 'adaptive',
        heterogeneous_k: bool = True,
        stagger: int = 0,
        tree_lr: float = 0.1,
        max_trees: int = 20,
        auto_grow: bool = True,
        auto_prune: bool = True,
        prune_floor: float = 0.15,
        prune_window: int = 50,
        grow_threshold: int = 8,
        task: str = 'sequence',
    ):
        self.dropout        = dropout
        self._tree_bcs      = binary_correction_scale
        self.voting         = voting
        self.tree_lr        = tree_lr
        self.task           = task
        self.max_trees      = max_trees
        self.auto_grow      = auto_grow
        self.auto_prune     = auto_prune
        self.prune_floor    = prune_floor
        self.prune_window   = prune_window
        self.grow_threshold = grow_threshold

        # Stored for spawning new trees
        self._base_k   = context_length
        self._sim_fn   = similarity_fn
        self._lr       = learning_rate
        self._coup_lr  = coupling_lr
        self._fb_str   = feedback_strength
        self._vig      = vigilance
        self._min_k    = min_context_length
        self._coup_ema = coupling_ema

        self._master_rng = random.Random(seed)

        k_values = (
            [context_length + i for i in range(n_trees)]
            if heterogeneous_k
            else [context_length] * n_trees
        )

        self.trees: list[UniversalPredictor] = [
            UniversalPredictor(
                k_values[i], similarity_fn,
                learning_rate=learning_rate, coupling_lr=coupling_lr,
                feedback_strength=feedback_strength, vigilance=vigilance,
                min_context_length=min_context_length, coupling_ema=coupling_ema,
                cont_count_min_vocab=16,
                binary_correction_scale=binary_correction_scale,
            )
            for i in range(n_trees)
        ]

        n = n_trees
        self._rngs:       list[random.Random] = [random.Random(self._master_rng.randint(0, 2**32)) for _ in range(n)]
        self._offsets:    list[int]           = [i * stagger for i in range(n)]
        self._steps:      list[int]           = [0] * n
        self._tree_creds: list[float]         = [1.0] * n
        self._last_preds: list[Any]           = [None] * n
        self._prune_ctrs: list[int]           = [0] * n

        self._inactive:      set[int] = set()
        self._corr_fail_str: int      = 0
        self._n_spawned:     int      = 0

    # ── active subset ─────────────────────────────────────────────────────────

    @property
    def prune_stale_branches(self, max_age: int = 100000) -> int:
        """
        LRU Eviction (RAM Optimization): Triggers the pruning of nodes 
        that have not been accessed or updated in `max_age` global steps
        across all trees in the forest.
        """
        total_pruned = 0
        for tree in self.trees:
            total_pruned += tree.prune_stale_branches(max_age)
        return total_pruned

    def _get_active_trees(self) -> list[int]:
        return [i for i in range(len(self.trees)) if i not in self._inactive]

    @property
    def _active(self) -> list[int]:
        return [i for i in range(len(self.trees)) if i not in self._inactive]

    # ── distribution helpers ──────────────────────────────────────────────────

    def _tree_dist(self, tree: UniversalPredictor) -> dict[Any, float]:
        contrib = tree._last_contributions
        if not contrib:
            return {}
        total = sum(w for w, _ in contrib.values()) or 1e-12
        d: dict[Any, float] = defaultdict(float)
        for w, succ in contrib.values():
            d[succ] += w / total
        return dict(d)

    def _mixture_dist(
        self,
        dists: list[dict[Any, float]],
        confs: list[float],
        creds: list[float],
    ) -> dict[Any, float]:
        """Confidence × credibility weighted sum of distributions."""
        total_w = sum(c * cr for c, cr in zip(confs, creds) if c > 0) or 1e-12
        result: dict[Any, float] = defaultdict(float)
        for d, c, cr in zip(dists, confs, creds):
            if c > 0 and d:
                w = c * cr / total_w
                for v, p in d.items():
                    result[v] += w * p
        return dict(result)

    def _product_dist(
        self,
        dists: list[dict[Any, float]],
        confs: list[float],
        creds: list[float],
    ) -> dict[Any, float]:
        """Weighted geometric mean of distributions."""
        active_pairs = [(d, cr) for d, c, cr in zip(dists, confs, creds) if c > 0 and d]
        if not active_pairs:
            return {}
        vocab      = set().union(*(d.keys() for d, _ in active_pairs))
        total_cred = sum(cr for _, cr in active_pairs) or 1e-12
        floor      = 1.0 / max(len(vocab), 1)
        product: dict[Any, float] = {}
        for v in vocab:
            log_p = sum(
                (cr / total_cred) * math.log(max(d.get(v, floor), floor))
                for d, cr in active_pairs
            )
            product[v] = math.exp(log_p)
        total = sum(product.values())
        if total < 1e-12:
            return {}
        return {v: p / total for v, p in product.items()}

    def _adaptive_dist(
        self,
        dists: list[dict[Any, float]],
        confs: list[float],
        creds: list[float],
    ) -> dict[Any, float]:
        alpha = sum(c for c in confs if c > 0) / (len(confs) or 1)
        mix  = self._mixture_dist(dists, confs, creds)
        prod = self._product_dist(dists, confs, creds)
        if not prod:
            return mix
        if not mix:
            return prod
        vocab   = set(mix) | set(prod)
        blended = {v: alpha * prod.get(v, 0.0) + (1.0 - alpha) * mix.get(v, 0.0)
                   for v in vocab}
        total   = sum(blended.values())
        if total < 1e-12:
            return mix
        return {v: p / total for v, p in blended.items()}

    def _dist_to_prediction(self, dist: dict[Any, float]) -> tuple[Any, float]:
        if not dist:
            return None, 0.0

        if self.task == 'regression':
            try:
                prediction = sum(float(v) * p for v, p in dist.items())
                n_vals = len(dist)
                if n_vals > 1:
                    entropy = -sum(p * math.log(p + 1e-12) for p in dist.values())
                    conf    = 1.0 - entropy / math.log(n_vals)
                else:
                    conf = 1.0
                return prediction, max(0.0, conf)
            except (TypeError, ValueError):
                pass

        best = max(dist, key=dist.get)
        return best, float(dist[best])

    # ── public interface ──────────────────────────────────────────────────────

    def observe(self, value: Any) -> None:
        for i, tree in enumerate(self.trees):
            if i not in self._inactive:
                tree.observe(value)

    def predict(self) -> tuple[Any, float]:
        active  = self._active
        n_total = len(self.trees)

        dists_full:  list[dict[Any, float]] = [{} for _ in range(n_total)]
        dists_crude: list[dict[Any, float]] = [{} for _ in range(n_total)]
        confs: list[float]                  = [0.0] * n_total

        for i in active:
            pred, conf          = self.trees[i].predict()
            self._last_preds[i] = pred
            crude               = self._tree_dist(self.trees[i])
            full                = self.trees[i]._distribution()
            dists_full[i]       = full if full else crude
            dists_crude[i]      = crude
            confs[i]            = conf

        active_full  = [dists_full[i]       for i in active]
        active_crude = [dists_crude[i]      for i in active]
        active_confs = [confs[i]            for i in active]
        active_creds = [self._tree_creds[i] for i in active]

        if self.voting == 'mixture':
            dist = self._mixture_dist(active_full, active_confs, active_creds)
        elif self.voting == 'product':
            dist = self._product_dist(active_crude, active_confs, active_creds)
            if not dist:
                dist = self._mixture_dist(active_full, active_confs, active_creds)
        else:
            alpha = sum(c for c in active_confs if c > 0) / (len(active_confs) or 1)
            mix  = self._mixture_dist(active_full,  active_confs, active_creds)
            prod = self._product_dist(active_crude, active_confs, active_creds)
            if not prod:
                dist = mix
            elif not mix:
                dist = prod
            else:
                vocab   = set(mix) | set(prod)
                blended = {v: alpha * prod.get(v, 0.0) + (1.0 - alpha) * mix.get(v, 0.0)
                           for v in vocab}
                total   = sum(blended.values())
                dist    = {v: p / total for v, p in blended.items()} if total > 1e-12 else mix

        return self._dist_to_prediction(dist)

    def feedback(self, actual: Any) -> None:
        active = self._active

        for i in active:
            self._steps[i] += 1
            if self._steps[i] <= self._offsets[i]:
                continue
            if self._rngs[i].random() < self.dropout:
                continue

            self.trees[i].feedback(actual)

            if self._last_preds[i] is not None:
                correct = self._last_preds[i] == actual
                factor  = 1.0 + self.tree_lr if correct else 1.0 - self.tree_lr
                self._tree_creds[i] = max(0.1, self._tree_creds[i] * factor)

        if active:
            max_cred = max(self._tree_creds[i] for i in active)
            if max_cred > 5.0:
                scale = 5.0 / max_cred
                for i in active:
                    self._tree_creds[i] *= scale

        if self.auto_grow and len(active) < self.max_trees:
            self._check_grow(active, actual)
        if self.auto_prune and len(self._active) > 2:
            self._check_prune(self._active)

    # ── dynamic sizing ────────────────────────────────────────────────────────

    def _check_grow(self, active: list[int], actual: Any) -> None:
        if not active:
            return
        wrong = [i for i in active
                 if self._last_preds[i] is not None and self._last_preds[i] != actual]
        if (len(wrong) == len(active)
                and len({self._last_preds[i] for i in wrong}) == 1):
            self._corr_fail_str += 1
            if self._corr_fail_str >= self.grow_threshold:
                self._spawn_tree()
                self._corr_fail_str = 0
        else:
            self._corr_fail_str = 0

    def _check_prune(self, active: list[int]) -> None:
        if len(active) <= 2:
            return
        mean_cred = sum(self._tree_creds[i] for i in active) / len(active)
        for i in active:
            if self._tree_creds[i] < self.prune_floor * mean_cred:
                self._prune_ctrs[i] += 1
                if self._prune_ctrs[i] >= self.prune_window:
                    self._inactive.add(i)
            else:
                self._prune_ctrs[i] = 0

    def _spawn_tree(self) -> None:
        if len(self.trees) >= self.max_trees:
            return
        active = self._active
        new_k  = (max(self.trees[i].k for i in active) + 1) if active else self._base_k + 1

        self.trees.append(
            UniversalPredictor(
                new_k, self._sim_fn,
                learning_rate=self._lr, coupling_lr=self._coup_lr,
                feedback_strength=self._fb_str, vigilance=self._vig,
                min_context_length=self._min_k, coupling_ema=self._coup_ema,
                cont_count_min_vocab=16,
                binary_correction_scale=self._tree_bcs,
            )
        )
        self._rngs.append(random.Random(self._master_rng.randint(0, 2**32)))
        self._offsets.append(0)
        self._steps.append(0)
        self._tree_creds.append(1.0)
        self._last_preds.append(None)
        self._prune_ctrs.append(0)
        self._n_spawned += 1

    # ── diagnostics ───────────────────────────────────────────────────────────

    def node_stats(self) -> dict:
        active    = self._active
        all_stats = [self.trees[i].node_stats() for i in active]
        if not all_stats:
            return {
                'total_nodes': 0, 'observed': 0, 'exploration': 0, 'correction': 0,
                'coupling_links': 0, 'mean_coupling': 0.0, 'max_coupling': 0.0,
                'lambda': 0.0, 'optimizer_budget': 0, 'optimizer_rolling_acc': 0.0,
                'allocator_trials': 0, 'n_active': 0, 'n_total': 0,
                'n_spawned': 0, 'n_inactive': 0,
            }
        n_active = len(active)
        result: dict = {}
        for key in ('total_nodes', 'observed', 'exploration',
                    'correction', 'coupling_links', 'allocator_trials'):
            result[key] = sum(s[key] for s in all_stats)
        for key in ('mean_coupling', 'lambda', 'optimizer_rolling_acc'):
            result[key] = sum(s[key] for s in all_stats) / n_active
        result['max_coupling']     = max(s['max_coupling']     for s in all_stats)
        result['optimizer_budget'] = int(sum(s['optimizer_budget'] for s in all_stats) / n_active)
        result['n_active']         = n_active
        result['n_total']          = len(self.trees)
        result['n_spawned']        = self._n_spawned
        result['n_inactive']       = len(self._inactive)
        return result

    def similarity_quality(self) -> float:
        active = self._active
        if not active:
            return 0.0
        return sum(self.trees[i].similarity_quality() for i in active) / len(active)

    def convergence_state(self) -> dict:
        active = self._active
        if not active:
            return {'plateau': None, 'tau': None, 'quality_now': 0.0,
                    'steps_to_95pct': None, 'converged': False}
        states    = [self.trees[i].convergence_state() for i in active]
        qualities = [s['quality_now'] for s in states]
        median_q  = sorted(qualities)[len(active) // 2]
        idx_local = min(range(len(active)), key=lambda j: abs(qualities[j] - median_q))
        return states[idx_local]

    def lookahead_quality(self, n_steps: int) -> float:
        active = self._active
        if not active:
            return 0.0
        return sum(self.trees[i].lookahead_quality(n_steps) for i in active) / len(active)
