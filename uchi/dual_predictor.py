"""
DualPredictor
=============
Solves Problem 7 — Stationary/drift tradeoff.

Two UniversalPredictors run in parallel:
  stable  — high cred_max, high lambda_power: best on stationary data
  drift   — low  cred_max, low  lambda_power: fastest drift recovery

Their output distributions are blended by a weight that tracks the recent
error rate on a rolling window.  When the error rate is high (something
changed), weight shifts toward the drift predictor.  When low and stable,
weight shifts toward the stability predictor.

The routing is fully automatic and requires no configuration beyond the
window size.

API
---
dp = DualPredictor(context_length=4)
dp.observe(token)
pred, conf = dp.predict()
dp.feedback(actual)
dp.error_rate          # current rolling error rate
dp.active_predictor    # 'stable' | 'drift' | 'blend'
"""

from collections import deque
from .predictor import UniversalPredictor


class DualPredictor:
    """
    Parameters
    ----------
    context_length : int
    window : int
        Rolling window size for error rate tracking (default 50).
    drift_threshold : float
        Error rate above which the drift predictor dominates (default 0.4).
    stable_threshold : float
        Error rate below which the stability predictor dominates (default 0.15).
    stable_cred_max, stable_lambda_power : float
        Hyperparameters for the stability-tuned predictor.
    drift_cred_max, drift_lambda_power : float
        Hyperparameters for the drift-tuned predictor.
    learning_rate : float
        Shared learning rate for both predictors.
    """

    def __init__(
        self,
        context_length: int,
        window: int = 50,
        drift_threshold: float = 0.4,
        stable_threshold: float = 0.15,
        stable_cred_max: float = 8.0,
        stable_lambda_power: float = 0.8,
        drift_cred_max: float = 3.0,
        drift_lambda_power: float = 0.4,
        learning_rate: float = 0.08,
        **kwargs,
    ):
        self._stable = UniversalPredictor(
            context_length,
            learning_rate=learning_rate,
            cred_max=stable_cred_max,
            lambda_power=stable_lambda_power,
            **kwargs,
        )
        self._drift = UniversalPredictor(
            context_length,
            learning_rate=learning_rate,
            cred_max=drift_cred_max,
            lambda_power=drift_lambda_power,
            **kwargs,
        )
        self._window = deque(maxlen=window)
        self._drift_threshold = drift_threshold
        self._stable_threshold = stable_threshold
        self._error_rate: float = 0.5
        self._last_pred = None
        self._last_dist: dict = {}
        self._history: list[dict] = []  # per-step routing log

    # ── public API ────────────────────────────────────────────────────────────

    def observe(self, token) -> 'DualPredictor':
        self._stable.observe(token)
        self._drift.observe(token)
        return self

    def predict(self):
        """
        Blend stable and drift distributions based on current error rate.
        Returns (predicted_token, confidence).
        """
        self._stable.predict()
        self._drift.predict()
        p_s = self._stable._last_distribution
        p_d = self._drift._last_distribution

        w_drift, w_stable = self._routing_weights()

        all_keys = set(p_s) | set(p_d)
        if not all_keys:
            self._last_pred = None
            self._last_dist = {}
            return None, 0.0

        blended = {
            k: w_stable * p_s.get(k, 0.0) + w_drift * p_d.get(k, 0.0)
            for k in all_keys
        }
        total = sum(blended.values()) or 1.0
        self._last_dist = {k: v / total for k, v in blended.items()}
        self._last_pred = max(self._last_dist, key=self._last_dist.get)
        return self._last_pred, self._last_dist[self._last_pred]

    def feedback(self, actual) -> None:
        """Update error rate and both predictors."""
        wrong = (self._last_pred != actual)
        self._window.append(1 if wrong else 0)
        self._error_rate = sum(self._window) / len(self._window) if self._window else 0.5
        self._stable.feedback(actual)
        self._drift.feedback(actual)

    # ── diagnostics ───────────────────────────────────────────────────────────

    @property
    def error_rate(self) -> float:
        return self._error_rate

    @property
    def active_predictor(self) -> str:
        w_d, w_s = self._routing_weights()
        if w_s > 0.8:
            return 'stable'
        if w_d > 0.8:
            return 'drift'
        return 'blend'

    @property
    def _last_distribution(self) -> dict:
        return self._last_dist

    @property
    def _vocab(self) -> set:
        return self._stable._vocab

    @property
    def history(self) -> list:
        return self._stable.history

    # ── internal ──────────────────────────────────────────────────────────────

    def _routing_weights(self) -> tuple[float, float]:
        """
        Returns (w_drift, w_stable).
        Linear interpolation between thresholds:
          error_rate >= drift_threshold  → (1.0, 0.0)
          error_rate <= stable_threshold → (0.0, 1.0)
          in between                     → smooth blend
        """
        e = self._error_rate
        if e >= self._drift_threshold:
            return 1.0, 0.0
        if e <= self._stable_threshold:
            return 0.0, 1.0
        span = self._drift_threshold - self._stable_threshold
        w_d = (e - self._stable_threshold) / span
        return w_d, 1.0 - w_d
