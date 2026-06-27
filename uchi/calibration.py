"""Confidence calibration for the SSM value head.

Applies temperature scaling (Guo et al., 2017 — "On Calibration of Modern
Neural Networks") to the raw SparseMoEValueHead output, converting unbounded
scalars to calibrated probabilities in (0, 1).

Temperature scaling is a single-parameter post-hoc calibration:
    p = sigmoid(raw_score / T)

where T > 0 is fit by minimising NLL on a held-out set. T = 1.0 is the
identity (no calibration). T > 1 softens (spreads) the distribution;
T < 1 sharpens it.

Rationale for temperature over Platt scaling:
  - Platt (two parameters: a, b) can overfit small calibration sets.
  - Temperature (one parameter) is sufficient when the model's ranking is
    already good; it only adjusts the probability magnitude, not the ordering.
  - Easier to audit: a single scalar fully describes the calibration state.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_log = logging.getLogger(__name__)

_DEFAULT_TEMPERATURE = 1.0
_DEFAULT_CALIB_PATH  = "ssm_calibration.json"


class TemperatureCalibrator(nn.Module):
    """Post-hoc temperature scaling for the SSM value head.

    Wraps the raw scalar output of SparseMoEValueHead with:
        p = sigmoid(raw / T)

    T is optimized via NLL on a small held-out sample.
    Falls back to T=1.0 (identity) if no calibration data is available.
    """

    def __init__(self, temperature: float = _DEFAULT_TEMPERATURE):
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor(max(temperature, 1e-3), dtype=torch.float32)
        )

    def calibrate(self, raw_scores: Sequence[float],
                  binary_labels: Sequence[float]) -> float:
        """Fit temperature T to minimise NLL on (raw_score, label) pairs.

        Args:
            raw_scores:    Raw SSM value head outputs (unbounded floats).
            binary_labels: Soft or hard labels in [0, 1] indicating whether
                           the prediction was "correct" / "rewarded".

        Returns:
            Final temperature T.
        """
        if len(raw_scores) < 4:
            _log.warning("Too few calibration samples (%d); keeping T=%.2f.",
                         len(raw_scores), self.temperature.item())
            return self.temperature.item()

        scores = torch.tensor(raw_scores, dtype=torch.float32)
        labels = torch.tensor(binary_labels, dtype=torch.float32)

        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=0.1, max_iter=100,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            logits = scores / self.temperature.clamp(min=1e-3)
            loss   = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            return loss

        with torch.enable_grad():
            for _ in range(3):
                optimizer.step(closure)

        T = self.temperature.item()
        _log.info("Calibration complete: T=%.4f", T)
        return T

    def forward(self, raw_score: torch.Tensor) -> torch.Tensor:
        """Apply temperature and sigmoid. Returns probability in (0, 1)."""
        return torch.sigmoid(raw_score / self.temperature.clamp(min=1e-3))

    def predict(self, raw_score: float) -> float:
        """Scalar convenience wrapper for inference."""
        with torch.no_grad():
            t = torch.tensor(raw_score, dtype=torch.float32)
            return self.forward(t).item()

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = _DEFAULT_CALIB_PATH) -> None:
        data = {"temperature": self.temperature.item()}
        with open(path, "w") as f:
            json.dump(data, f)
        _log.debug("Calibration saved: T=%.4f → %s", self.temperature.item(), path)

    @classmethod
    def load(cls, path: str = _DEFAULT_CALIB_PATH) -> "TemperatureCalibrator":
        if not os.path.exists(path):
            _log.debug("No calibration file at %s; using T=1.0.", path)
            return cls()
        try:
            with open(path) as f:
                data = json.load(f)
            T = float(data.get("temperature", _DEFAULT_TEMPERATURE))
            _log.info("Loaded calibration T=%.4f from %s", T, path)
            return cls(temperature=T)
        except Exception as e:
            _log.warning("Calibration load failed (%s); using T=1.0.", e)
            return cls()


def collect_calibration_data(
    router,
    n_samples: int = 200,
) -> Tuple[List[float], List[float]]:
    """Collect (raw_score, label) pairs from the router's trie + SSM.

    Samples random trie paths of length 4–8, asks the SSM to score the
    last state, and uses trie path probability as the soft label (high
    trie probability = the path is well-supported = positive signal).

    Returns:
        raw_scores, labels
    """
    from uchi.neuro_symbolic import get_ssm

    ssm = get_ssm()
    raw_scores: List[float] = []
    labels:     List[float] = []

    trie = getattr(router, "trie", None)
    if trie is None:
        _log.warning("Router has no trie attribute; cannot collect calibration data.")
        return [], []

    def _sample_path(node, depth: int = 0) -> Optional[Tuple[List[str], float]]:
        """Random walk from node, returning (token_path, path_probability)."""
        if depth >= 8 or not getattr(node, "children", {}):
            return None
        children = node.children
        if not children:
            return None
        total = sum(c.count for c in children.values())
        if total == 0:
            return None
        import random
        token = random.choice(list(children.keys()))
        child = children[token]
        prob  = child.count / total
        rest  = _sample_path(child, depth + 1)
        if rest is None:
            return [token], prob
        sub_tokens, sub_prob = rest
        return [token] + sub_tokens, prob * sub_prob

    root = trie.root if hasattr(trie, "root") else trie
    sampled = 0
    attempts = 0
    max_attempts = n_samples * 10

    while sampled < n_samples and attempts < max_attempts:
        attempts += 1
        result = _sample_path(root)
        if result is None or len(result[0]) < 2:
            continue
        tokens, path_prob = result

        try:
            with torch.no_grad():
                state = ssm.get_state(tokens)
                raw   = ssm.value(state).item()
        except Exception:
            continue

        raw_scores.append(raw)
        labels.append(min(path_prob * 10.0, 1.0))  # scale: 0.1 path prob → 1.0 label
        sampled += 1

    _log.info("Collected %d calibration samples (%d attempts).", sampled, attempts)
    return raw_scores, labels


def run_calibration(
    router,
    calib_path: str = _DEFAULT_CALIB_PATH,
    n_samples: int = 200,
) -> TemperatureCalibrator:
    """End-to-end calibration: collect data, fit T, save, return calibrator."""
    raw_scores, labels = collect_calibration_data(router, n_samples=n_samples)
    calibrator = TemperatureCalibrator()
    if raw_scores:
        calibrator.calibrate(raw_scores, labels)
    calibrator.save(calib_path)
    return calibrator
