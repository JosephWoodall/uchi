"""
tabular.py
==========
Tabular classification and regression via the Universal Sequence Predictor.

How it works
------------
Each row is encoded as an ordered sequence of (feature_index, bin) tokens.
The class label (or regression bin) is the next token after all feature tokens.
The predictor learns the conditional distribution P(label | feature_sequence).

Diversity comes from running multiple predictors with different feature orderings:
  • MI-ascending   — least informative feature first, most informative last
  • MI-descending  — most informative feature first
  • natural        — as supplied
  • shuffled       — random permutation (adds variety)

At inference, label probability distributions are averaged across all predictors.

Classes
-------
  TabularPredictor    — classification  (sklearn-compatible)
  TabularRegressor    — regression      (sklearn-compatible)
"""

from __future__ import annotations

import math
import random
from typing import Any

from .predictor import UniversalPredictor
from .discretize import FeatureDiscretizer, LabelEncoder

try:
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    _SKLEARN = True
except ImportError:
    class BaseEstimator:
        pass
    class ClassifierMixin:
        pass
    class RegressorMixin:
        pass
    _SKLEARN = False

_LABEL_NS  = '__label__'
_TARGET_NS = '__target__'


# ══════════════════════════════════════════════════════════════════════════════
# Shared internals
# ══════════════════════════════════════════════════════════════════════════════

def _make_predictor(
    k: int, lr: float, cred_max: float, lp: float,
    use_similarity_fallback: bool = False,
    use_positional_weights: bool = False,
    min_context_length: int = 1,
) -> UniversalPredictor:
    return UniversalPredictor(
        k, None,
        learning_rate=lr,
        vigilance=0.3,
        adaptive_cap=True,
        binary_correction_scale=0.05,
        cred_max_base=cred_max,
        lambda_power=lp,
        cont_count_min_vocab=4,
        use_similarity_fallback=use_similarity_fallback,
        use_positional_weights=use_positional_weights,
        min_context_length=min_context_length,
    )


def _mi_order(token_rows: list, y_encoded: list, n_features: int) -> list:
    """Feature indices sorted by MI with y (ascending). Falls back to natural order."""
    try:
        import numpy as np
        from sklearn.feature_selection import mutual_info_classif

        X_arr = np.zeros((len(token_rows), n_features), dtype=float)
        for i, row in enumerate(token_rows):
            for j, val in row:
                X_arr[i, j] = -1 if val == '__MISSING__' else (
                    float(val) if isinstance(val, int) else hash(val) % 100)
        y_arr = np.array(y_encoded)
        mi = mutual_info_classif(X_arr, y_arr, discrete_features=True, random_state=0)
        return list(np.argsort(mi))
    except (ImportError, Exception):
        return list(range(n_features))


def _apply_order(token_row: list, order: list) -> list:
    idx_map = {tok[0]: tok for tok in token_row}
    return [idx_map[i] for i in order if i in idx_map]


def _set_history(p: UniversalPredictor, tokens: list) -> None:
    p.history = list(tokens)


def _train_one(p: UniversalPredictor, ordered_tokens: list, next_token: Any) -> None:
    _set_history(p, ordered_tokens)
    p.predict()
    p.observe(next_token)
    p.feedback(next_token)
    p.history.clear()


def _infer_dist(p: UniversalPredictor, ordered_tokens: list) -> dict:
    saved = p.history[:]
    _set_history(p, ordered_tokens)
    p.predict()
    dist = dict(p._last_distribution)
    p.history = saved
    return dist


def _build_orders(rows, y_enc, n_feat, n_orderings, rng) -> list:
    natural  = list(range(n_feat))
    mi_asc   = _mi_order(rows, y_enc, n_feat)
    mi_desc  = list(reversed(mi_asc))
    candidates = [mi_asc, mi_desc, natural]
    for seed in range(n_orderings - len(candidates)):
        perm = natural[:]
        random.Random(seed).shuffle(perm)
        candidates.append(perm)
    return candidates[:n_orderings]


# ══════════════════════════════════════════════════════════════════════════════
# TabularPredictor
# ══════════════════════════════════════════════════════════════════════════════

class TabularPredictor(BaseEstimator, ClassifierMixin):
    """
    Tabular classification via feature-as-sequence encoding.

    sklearn-compatible: works in Pipeline, GridSearchCV, cross_val_score.
    Supports partial_fit for online / incremental learning.

    Parameters
    ----------
    n_bins : int
        Quantile bins for continuous features (default 10).
    context_length : int | None
        Trie depth k.  None = number of features (recommended).
    n_orderings : int
        Number of feature orderings to ensemble (default 3).
    n_epochs : int
        Training passes over the data (default 1).
    learning_rate, cred_max, lambda_power : float
    random_seed : int
    """

    def __init__(
        self,
        n_bins:         int        = 10,
        context_length: int | None = None,
        n_orderings:    int        = 3,
        n_epochs:       int        = 1,
        learning_rate:  float      = 0.08,
        cred_max:       float      = 6.05,
        lambda_power:   float      = 0.65,
        random_seed:    int        = 42,
    ):
        self.n_bins         = n_bins
        self.context_length = context_length
        self.n_orderings    = n_orderings
        self.n_epochs       = n_epochs
        self.learning_rate  = learning_rate
        self.cred_max       = cred_max
        self.lambda_power   = lambda_power
        self.random_seed    = random_seed
        self._replay_buffer = []
        self._replay_batch_size = 100

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X, y) -> 'TabularPredictor':
        self._disc   = FeatureDiscretizer(n_bins=self.n_bins)
        self._lenc   = LabelEncoder()
        self._rng    = random.Random(self.random_seed)
        self._preds  = []
        self._orders = []

        rows   = self._disc.fit_transform(X)
        labels = list(y)
        self._lenc.fit(labels)
        y_enc  = [self._lenc.encode(lbl) for lbl in labels]

        n_feat = self._disc.n_features
        k      = n_feat if self.context_length is None else self.context_length
        self._orders = _build_orders(rows, y_enc, n_feat, self.n_orderings, self._rng)
        self._preds  = [_make_predictor(k, self.learning_rate, self.cred_max, self.lambda_power)
                        for _ in self._orders]

        for _ in range(self.n_epochs):
            pairs = list(zip(rows, labels))
            self._rng.shuffle(pairs)
            for tok_row, label in pairs:
                self._train_row(tok_row, label)

        self.is_fitted_ = True
        return self

    def partial_fit(self, X, y, classes=None) -> 'TabularPredictor':
        if not hasattr(self, '_disc'):
            return self.fit(X, y)
        self._disc.partial_fit(X)
        rows   = self._disc.transform(X)
        labels = list(y)
        self._lenc.partial_fit(labels)
        
        # Experience Replay Buffer Logic
        for tok_row, label in zip(rows, labels):
            self._replay_buffer.append((tok_row, label))
            
            if len(self._replay_buffer) >= self._replay_batch_size:
                # Train on the buffer multiple times to stabilize
                for _ in range(self.n_epochs):
                    buffer_copy = self._replay_buffer[:]
                    self._rng.shuffle(buffer_copy)
                    for r, lbl in buffer_copy:
                        self._train_row(r, lbl)
                        
                # Keep the last 20% to mix with incoming data (sliding window overlap)
                keep = int(self._replay_batch_size * 0.2)
                self._replay_buffer = self._replay_buffer[-keep:]
                
        return self

    def predict(self, X) -> list:
        proba = self.predict_proba(X)
        return [max(d, key=d.get) for d in proba]

    def predict_proba(self, X) -> list:
        rows = self._disc.transform(X)
        return [self._infer_row(r) for r in rows]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        return sum(p == t for p, t in zip(preds, y)) / max(len(list(y)), 1)

    @property
    def classes_(self) -> list:
        return self._lenc.classes_ if hasattr(self, '_lenc') else []

    # ── internal ──────────────────────────────────────────────────────────────

    def _label_token(self, label) -> tuple:
        return (_LABEL_NS, self._lenc.encode(label))

    def _train_row(self, tok_row: list, label) -> None:
        lt = self._label_token(label)
        for p, order in zip(self._preds, self._orders):
            _train_one(p, _apply_order(tok_row, order), lt)

    def _infer_row(self, tok_row: list) -> dict:
        classes = self._lenc.classes_
        if not classes:
            return {}
        totals = {c: 0.0 for c in classes}
        for p, order in zip(self._preds, self._orders):
            dist = _infer_dist(p, _apply_order(tok_row, order))
            for c in classes:
                totals[c] += dist.get(self._label_token(c), 0.0)
        total = sum(totals.values())
        if total < 1e-12:
            u = 1.0 / len(classes)
            return {c: u for c in classes}
        return {c: v / total for c, v in totals.items()}


# ══════════════════════════════════════════════════════════════════════════════
# TabularRegressor
# ══════════════════════════════════════════════════════════════════════════════

class TabularRegressor(BaseEstimator, RegressorMixin):
    """
    Tabular regression via binned-target sequence encoding.

    The continuous target is discretised into quantile bins at fit time.
    Prediction returns the credibility-weighted mean of bin centres.
    predict_interval() also returns the bin distribution std as an
    uncertainty estimate.

    sklearn-compatible: works in Pipeline, GridSearchCV, cross_val_score.

    Parameters
    ----------
    n_bins : int
        Bins for continuous features AND for the regression target.
    n_target_bins : int | None
        Bins specifically for the target (defaults to n_bins).
    All other parameters: same as TabularPredictor.
    """

    def __init__(
        self,
        n_bins:         int        = 10,
        n_target_bins:  int | None = None,
        context_length: int | None = None,
        n_orderings:    int        = 3,
        n_epochs:       int        = 1,
        learning_rate:  float      = 0.08,
        cred_max:       float      = 6.05,
        lambda_power:   float      = 0.65,
        random_seed:    int        = 42,
    ):
        self.n_bins        = n_bins
        self.n_target_bins = n_target_bins
        self.context_length = context_length
        self.n_orderings   = n_orderings
        self.n_epochs      = n_epochs
        self.learning_rate = learning_rate
        self.cred_max      = cred_max
        self.lambda_power  = lambda_power
        self.random_seed   = random_seed

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X, y) -> 'TabularRegressor':
        n_tgt = self.n_target_bins if self.n_target_bins is not None else self.n_bins
        self._n_tgt_bins = n_tgt
        self._disc     = FeatureDiscretizer(n_bins=self.n_bins)
        self._tgt_disc = FeatureDiscretizer(n_bins=n_tgt)
        self._rng      = random.Random(self.random_seed)
        self._preds    = []
        self._orders   = []

        rows   = self._disc.fit_transform(X)
        y_list = list(y)
        y_rows = self._tgt_disc.fit_transform([[v] for v in y_list])
        y_bins = [r[0][1] for r in y_rows]

        n_feat = self._disc.n_features
        y_enc  = [b if isinstance(b, int) else 0 for b in y_bins]
        self._orders = _build_orders(rows, y_enc, n_feat, self.n_orderings, self._rng)

        k = n_feat if self.context_length is None else self.context_length
        self._preds = [_make_predictor(k, self.learning_rate, self.cred_max, self.lambda_power)
                       for _ in self._orders]

        for _ in range(self.n_epochs):
            triples = list(zip(rows, y_bins))
            self._rng.shuffle(triples)
            for tok_row, y_bin in triples:
                self._train_row(tok_row, y_bin)

        self.is_fitted_ = True
        return self

    def partial_fit(self, X, y) -> 'TabularRegressor':
        if not hasattr(self, '_disc'):
            return self.fit(X, y)
        rows   = self._disc.transform(X)
        y_rows = self._tgt_disc.transform([[v] for v in y])
        y_bins = [r[0][1] for r in y_rows]
        for tok_row, y_bin in zip(rows, y_bins):
            self._train_row(tok_row, y_bin)
        return self

    def predict(self, X) -> list:
        return [mu for mu, _ in self.predict_interval(X)]

    def predict_interval(self, X) -> list:
        """Return list of (mean, std) tuples."""
        rows = self._disc.transform(X)
        return [self._infer_row(r) for r in rows]

    def score(self, X, y) -> float:
        preds  = self.predict(X)
        y_list = list(y)
        y_mean = sum(y_list) / len(y_list)
        ss_res = sum((p - t) ** 2 for p, t in zip(preds, y_list))
        ss_tot = sum((t - y_mean) ** 2 for t in y_list)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # ── internal ──────────────────────────────────────────────────────────────

    def _target_token(self, y_bin) -> tuple:
        return (_TARGET_NS, y_bin)

    def _train_row(self, tok_row: list, y_bin) -> None:
        tt = self._target_token(y_bin)
        for p, order in zip(self._preds, self._orders):
            _train_one(p, _apply_order(tok_row, order), tt)

    def _infer_row(self, tok_row: list) -> tuple:
        n_bins    = self._n_tgt_bins
        bin_probs = [0.0] * n_bins
        for p, order in zip(self._preds, self._orders):
            dist = _infer_dist(p, _apply_order(tok_row, order))
            for b in range(n_bins):
                bin_probs[b] += dist.get(self._target_token(b), 0.0)
        total = sum(bin_probs)
        if total < 1e-12:
            probs = [1.0 / n_bins] * n_bins
        else:
            probs = [v / total for v in bin_probs]
        centers = [self._tgt_disc.bin_center(0, b) for b in range(n_bins)]
        mean = sum(p * c for p, c in zip(probs, centers))
        var  = sum(p * (c - mean) ** 2 for p, c in zip(probs, centers))
        return mean, math.sqrt(max(var, 0.0))
