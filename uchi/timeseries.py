"""
timeseries.py
=============
Time series modeling via the Universal Sequence Predictor.

Each multivariate time step is encoded as a compound token
    (bin_0, bin_1, ..., bin_{M-1})
which is directly hashable and exact-matchable in the trie. Vocabulary is
sparse — only observed transitions are stored.

For large M, keep n_bins small (e.g. 4–6); theoretical vocab = n_bins^M.

Classes
-------
  MultivariateTSPredictor  — online step-ahead prediction for multivariate series
  TimeSeriesClassifier      — classify fixed-length windows (e.g. ECG, HAR)
  AnomalyDetector           — online anomaly scoring via prediction log-loss
"""

from __future__ import annotations

import math
import random
import warnings

from .predictor import UniversalPredictor
from .discretize import FeatureDiscretizer, LabelEncoder, _to_rows
from .tabular import _set_history, _infer_dist, _train_one

try:
    from sklearn.base import BaseEstimator, ClassifierMixin, OutlierMixin
    _SKLEARN = True
except ImportError:
    class BaseEstimator: pass
    class ClassifierMixin: pass
    class OutlierMixin: pass
    _SKLEARN = False

_LABEL_NS_TS = '__ts_label__'


# ══════════════════════════════════════════════════════════════════════════════
# Shared internals
# ══════════════════════════════════════════════════════════════════════════════

def _make_predictor(k: int, lr: float, cred_max: float, lp: float) -> UniversalPredictor:
    return UniversalPredictor(
        k, None,
        learning_rate=lr,
        vigilance=0.3,
        adaptive_cap=True,
        binary_correction_scale=0.05,
        cred_max=cred_max,
        lambda_power=lp,
        cont_count_min_vocab=4,
    )


def _compound_token(token_row: list) -> tuple:
    return tuple(b for _, b in sorted(token_row, key=lambda x: x[0]))


def _to_window_rows(w) -> list:
    rows = _to_rows(w)
    if rows and not isinstance(rows[0], (list, tuple)):
        rows = [[v] for v in rows]
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# MultivariateTSPredictor
# ══════════════════════════════════════════════════════════════════════════════

class MultivariateTSPredictor(BaseEstimator):
    """
    Online step-ahead predictor for multivariate (or univariate) time series.

    Learns P(x_{t+1} | x_{t-k+1}, ..., x_t) via the trie.  Each timestep
    becomes one compound token; the context window is k steps.

    For large M, set n_bins small (e.g. 4–6) to bound vocabulary.

    Parameters
    ----------
    n_bins : int
        Quantile bins per dimension.
    context_length : int
        Number of prior steps k used as context.
    learning_rate, cred_max, lambda_power : float

    Streaming API
    -------------
    pred.fit(X)            — fit bins and warm-up trie on historical data
    pred.predict()         — float vector of per-dimension bin-center means
    pred.observe(x)        — consume one true timestep (advances history)
    pred.feedback(x)       — update trie with true value
    pred.forecast(n)       — auto-regressive multi-step ahead forecast
    pred.score(X)          — average bits-per-step (lower = better)
    """

    def __init__(
        self,
        n_bins:         int   = 8,
        context_length: int   = 5,
        learning_rate:  float = 0.08,
        cred_max:       float = 6.05,
        lambda_power:   float = 0.65,
    ):
        self.n_bins         = n_bins
        self.context_length = context_length
        self.learning_rate  = learning_rate
        self.cred_max       = cred_max
        self.lambda_power   = lambda_power

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X, y=None) -> 'MultivariateTSPredictor':
        rows = _to_rows(X)
        if not rows:
            return self
        if not isinstance(rows[0], (list, tuple)):
            rows = [[v] for v in rows]

        self._n_dims = len(rows[0])
        B, M = self.n_bins, self._n_dims
        if B ** M > 100_000:
            warnings.warn(
                f"Trie vocab ≈ n_bins^M = {B}^{M} = {B**M:,}. "
                f"Reduce n_bins for high-dimensional series.",
                stacklevel=2,
            )

        self._disc = FeatureDiscretizer(n_bins=self.n_bins)
        self._disc.fit(rows)
        self._pred = _make_predictor(
            self.context_length, self.learning_rate, self.cred_max, self.lambda_power,
        )

        for row in rows:
            token = _compound_token(self._disc._encode_row(row))
            self._pred.predict()
            self._pred.observe(token)
            self._pred.feedback(token)

        self.is_fitted_ = True
        return self

    def observe(self, x) -> 'MultivariateTSPredictor':
        self._check_fitted()
        self._pred.observe(self._tokenize(x))
        return self

    def predict(self, X=None) -> list:
        """
        Predict next timestep as per-dimension float means.
        X is ignored (streaming API uses internal state); present for sklearn compat.
        """
        self._check_fitted()
        self._pred.predict()
        return self._decode_dist(dict(self._pred._last_distribution))

    def predict_distribution(self) -> dict:
        self._check_fitted()
        self._pred.predict()
        return dict(self._pred._last_distribution)

    def feedback(self, x) -> 'MultivariateTSPredictor':
        self._check_fitted()
        self._pred.feedback(self._tokenize(x))
        return self

    def forecast(self, n_steps: int) -> list:
        """
        Auto-regressive multi-step ahead forecast.
        Returns list of n_steps float vectors.  History is temporarily extended
        then restored; trie not modified.
        """
        self._check_fitted()
        saved = self._pred.history[:]
        results = []
        for _ in range(n_steps):
            self._pred.predict()
            means = self._decode_dist(dict(self._pred._last_distribution))
            results.append(means)
            self._pred.observe(self._tokenize(means))
        self._pred.history = saved
        return results

    def score(self, X, y=None) -> float:
        """
        Average bits-per-step on held-out data (lower = better).
        History temporarily advanced for context; trie not updated.
        """
        self._check_fitted()
        rows = _to_rows(X)
        if not rows:
            return float('inf')
        if not isinstance(rows[0], (list, tuple)):
            rows = [[v] for v in rows]

        saved = self._pred.history[:]
        total = 0.0
        for row in rows:
            token = _compound_token(self._disc._encode_row(row))
            self._pred.predict()
            prob   = max(self._pred._last_distribution.get(token, 1e-12), 1e-12)
            total += -math.log2(prob)
            self._pred.observe(token)
        self._pred.history = saved
        return total / len(rows)

    # ── internal ──────────────────────────────────────────────────────────────

    def _tokenize(self, x) -> tuple:
        row = [x] if isinstance(x, (int, float)) else list(x)
        return _compound_token(self._disc._encode_row(row))

    def _decode_dist(self, dist: dict) -> list:
        mid      = self.n_bins // 2
        fallback = [self._disc.bin_center(d, mid) for d in range(self._n_dims)]
        if not dist:
            return fallback
        total = sum(dist.values())
        if total < 1e-12:
            return fallback
        means = [0.0] * self._n_dims
        for token, prob in dist.items():
            if not isinstance(token, tuple) or len(token) != self._n_dims:
                continue
            w = prob / total
            for d, b in enumerate(token):
                if isinstance(b, int):
                    means[d] += w * self._disc.bin_center(d, b)
        return means

    def _check_fitted(self):
        if not hasattr(self, '_pred'):
            raise RuntimeError("Call fit() first.")


# ══════════════════════════════════════════════════════════════════════════════
# TimeSeriesClassifier
# ══════════════════════════════════════════════════════════════════════════════

class TimeSeriesClassifier(BaseEstimator, ClassifierMixin):
    """
    Classify fixed-length time series windows.

    Each window of T timesteps is encoded as T compound tokens; the class label
    is the next token after the full window.

    sklearn-compatible: works in Pipeline, GridSearchCV, cross_val_score.

    Parameters
    ----------
    n_bins : int
    window_size : int | None
        Expected window length (inferred from first fit call if None).
    n_epochs : int
    learning_rate, cred_max, lambda_power : float
    """

    def __init__(
        self,
        n_bins:         int        = 8,
        window_size:    int | None = None,
        n_epochs:       int        = 1,
        learning_rate:  float      = 0.08,
        cred_max:       float      = 6.05,
        lambda_power:   float      = 0.65,
        random_seed:    int        = 42,
    ):
        self.n_bins        = n_bins
        self.window_size   = window_size
        self.n_epochs      = n_epochs
        self.learning_rate = learning_rate
        self.cred_max      = cred_max
        self.lambda_power  = lambda_power
        self.random_seed   = random_seed

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X, y) -> 'TimeSeriesClassifier':
        windows = [_to_window_rows(w) for w in X]
        labels  = list(y)

        if not windows:
            return self

        T = len(windows[0])
        if self.window_size is not None and T != self.window_size:
            raise ValueError(f"window_size mismatch: expected {self.window_size}, got {T}")
        self._T = T

        all_steps = [step for w in windows for step in w]
        self._disc = FeatureDiscretizer(n_bins=self.n_bins)
        self._disc.fit(all_steps)
        self._lenc = LabelEncoder()
        self._lenc.fit(labels)
        self._rng  = random.Random(self.random_seed)

        self._pred = _make_predictor(self._T, self.learning_rate,
                                      self.cred_max, self.lambda_power)

        for _ in range(self.n_epochs):
            pairs = list(zip(windows, labels))
            self._rng.shuffle(pairs)
            for window, label in pairs:
                self._train_window(window, label)

        self.is_fitted_ = True
        return self

    def partial_fit(self, X, y, classes=None) -> 'TimeSeriesClassifier':
        if not hasattr(self, '_disc'):
            return self.fit(X, y)
        windows = [_to_window_rows(w) for w in X]
        labels  = list(y)
        self._lenc.partial_fit(labels)
        for window, label in zip(windows, labels):
            self._train_window(window, label)
        return self

    def predict(self, X) -> list:
        proba = self.predict_proba(X)
        return [max(d, key=d.get) for d in proba]

    def predict_proba(self, X) -> list:
        return [self._infer_window(_to_window_rows(w)) for w in X]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        return sum(p == t for p, t in zip(preds, y)) / max(len(list(y)), 1)

    @property
    def classes_(self) -> list:
        return self._lenc.classes_ if hasattr(self, '_lenc') else []

    # ── internal ──────────────────────────────────────────────────────────────

    def _label_token(self, label) -> tuple:
        return (_LABEL_NS_TS, self._lenc.encode(label))

    def _window_to_tokens(self, window: list) -> list:
        return [_compound_token(self._disc._encode_row(step)) for step in window]

    def _train_window(self, window: list, label) -> None:
        _train_one(self._pred, self._window_to_tokens(window), self._label_token(label))

    def _infer_window(self, window: list) -> dict:
        tokens  = self._window_to_tokens(window)
        dist    = _infer_dist(self._pred, tokens)
        classes = self._lenc.classes_

        if not classes:
            return {}

        totals = {c: dist.get(self._label_token(c), 0.0) for c in classes}
        total  = sum(totals.values())
        if total < 1e-12:
            u = 1.0 / len(classes)
            return {c: u for c in classes}
        return {c: v / total for c, v in totals.items()}


# ══════════════════════════════════════════════════════════════════════════════
# AnomalyDetector
# ══════════════════════════════════════════════════════════════════════════════

class AnomalyDetector(BaseEstimator, OutlierMixin):
    """
    Online anomaly detection via prediction surprise.

    Trains a MultivariateTSPredictor on normal data.  At inference each
    timestep receives score = -log2 P(actual | context).  High score = anomalous.
    The trie is NOT updated during scoring.

    sklearn-compatible: works in Pipeline.

    Parameters
    ----------
    All parameters forwarded to MultivariateTSPredictor.
    """

    def __init__(
        self,
        n_bins:         int   = 8,
        context_length: int   = 5,
        learning_rate:  float = 0.08,
        cred_max:       float = 6.05,
        lambda_power:   float = 0.65,
    ):
        self.n_bins         = n_bins
        self.context_length = context_length
        self.learning_rate  = learning_rate
        self.cred_max       = cred_max
        self.lambda_power   = lambda_power

    def fit(self, X, y=None) -> 'AnomalyDetector':
        self._ts = MultivariateTSPredictor(
            n_bins=self.n_bins, context_length=self.context_length,
            learning_rate=self.learning_rate, cred_max=self.cred_max,
            lambda_power=self.lambda_power,
        )
        self._ts.fit(X)
        self.is_fitted_ = True
        return self

    def score_samples(self, X) -> list:
        """Anomaly score per timestep (higher = more anomalous)."""
        self._check_fitted()
        rows = _to_rows(X)
        if not rows:
            return []
        if not isinstance(rows[0], (list, tuple)):
            rows = [[v] for v in rows]

        saved  = self._ts._pred.history[:]
        scores = []
        for row in rows:
            token = _compound_token(self._ts._disc._encode_row(row))
            self._ts._pred.predict()
            prob   = max(self._ts._pred._last_distribution.get(token, 1e-12), 1e-12)
            scores.append(-math.log2(prob))
            self._ts._pred.observe(token)
        self._ts._pred.history = saved
        return scores

    def predict(self, X) -> list:
        """Binary labels: True = anomaly (score > mean + 2*std)."""
        scores = self.score_samples(X)
        if not scores:
            return []
        mu        = sum(scores) / len(scores)
        var       = sum((s - mu) ** 2 for s in scores) / len(scores)
        threshold = mu + 2.0 * math.sqrt(max(var, 0.0))
        return [1 if s > threshold else -1 for s in scores]

    def decision_function(self, X) -> list:
        """sklearn OutlierMixin: negative anomaly score (higher = more normal)."""
        return [-s for s in self.score_samples(X)]

    def _check_fitted(self):
        if not hasattr(self, '_ts'):
            raise RuntimeError("Call fit() first.")
