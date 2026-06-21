"""
discretize.py
=============
Feature discretization for the Universal Sequence Predictor.

Converts continuous and categorical features into discrete symbol tokens
compatible with UniversalPredictor / PredictorForest.

  FeatureDiscretizer   — fits bins per-column, transforms rows to token lists
  LabelEncoder         — encodes classification targets to/from integers
"""

import math
from typing import Any


# ── sentinel for missing / unseen values ─────────────────────────────────────

MISSING = object()      # unique identity; cannot equal any real feature value
MISSING_STR = '__MISSING__'


# ══════════════════════════════════════════════════════════════════════════════
# FeatureDiscretizer
# ══════════════════════════════════════════════════════════════════════════════

class FeatureDiscretizer:
    """
    Transforms a feature matrix into sequences of (feature_index, bin) tokens.

    Column types detected automatically:
      • numeric  → equal-frequency (quantile) bins, labelled 0..n_bins-1
      • other    → ordinal integer encoding of unique values seen at fit time

    Missing values (None, NaN, empty string) map to MISSING_STR so they form
    their own trie branch rather than crashing or biasing bin counts.

    Parameters
    ----------
    n_bins : int
        Number of quantile bins for numeric columns.
    feature_names : list[str] | None
        Optional column names, used only for repr/debugging.

    Usage
    -----
    disc = FeatureDiscretizer(n_bins=10)
    token_rows = disc.fit_transform(X_train)   # list of [(col, bin), ...]
    test_rows  = disc.transform(X_test)
    """

    def __init__(self, n_bins: int = 10, feature_names: list | None = None):
        self.n_bins        = n_bins
        self.feature_names = feature_names
        self._n_features:  int  = 0
        self._types:       list = []   # 'numeric' | 'categorical' per column
        self._edges:       dict = {}   # col → sorted list of quantile cut-points
        self._cat_maps:    dict = {}   # col → {value: int}
        self._bin_centers: dict = {}   # col → list of float centers (numeric only)
        
        # Reservoir sampling state for dynamic online splitting
        self._reservoirs:  dict = {}   # col -> list of sampled float values
        self._reservoir_max: int = 2000

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X) -> 'FeatureDiscretizer':
        X = _to_rows(X)
        if not X:
            return self
        self._n_features = len(X[0])
        self._types = []
        self._edges = {}
        self._cat_maps = {}
        self._bin_centers = {}

        for j in range(self._n_features):
            col = [row[j] for row in X]
            if _is_numeric_col(col):
                self._types.append('numeric')
                self._reservoirs[j] = [float(v) for v in col if not _is_missing(v) and not (isinstance(v, float) and math.isnan(v))]
                edges, centers = _quantile_edges(col, self.n_bins)
                self._edges[j]       = edges
                self._bin_centers[j] = centers
            else:
                self._types.append('categorical')
                unique = sorted(
                    {_safe_str(v) for v in col if not _is_missing(v)})
                self._cat_maps[j] = {v: i for i, v in enumerate(unique)}
        self._total_seen = len(X)
        return self

    def partial_fit(self, X) -> 'FeatureDiscretizer':
        """
        Online dynamic splitting: Update the reservoir sample of numeric columns
        and re-calculate quantile bins if enough new data has arrived.
        """
        if self._n_features == 0:
            return self.fit(X)
            
        rows = _to_rows(X)
        if not rows:
            return self
            
        import random
        for row in rows:
            self._total_seen += 1
            for j in range(self._n_features):
                v = row[j]
                if self._types[j] == 'numeric':
                    if _is_missing(v):
                        continue
                    v = float(v)
                    # Reservoir sampling
                    if len(self._reservoirs[j]) < self._reservoir_max:
                        self._reservoirs[j].append(v)
                    else:
                        idx = random.randint(0, self._total_seen - 1)
                        if idx < self._reservoir_max:
                            self._reservoirs[j][idx] = v
                else:
                    # Categorical: dynamically add unseen categories
                    if not _is_missing(v):
                        s = _safe_str(v)
                        if s not in self._cat_maps[j]:
                            self._cat_maps[j][s] = len(self._cat_maps[j])
                            
        # Re-compute bins periodically (e.g., every time we process a batch)
        for j in range(self._n_features):
            if self._types[j] == 'numeric' and self._reservoirs[j]:
                edges, centers = _quantile_edges(self._reservoirs[j], self.n_bins)
                self._edges[j] = edges
                self._bin_centers[j] = centers
                
        return self

    def transform(self, X) -> list:
        """Return list-of-lists of (col_idx, bin_or_code) tokens."""
        rows = _to_rows(X)
        return [self._encode_row(row) for row in rows]

    def fit_transform(self, X) -> list:
        return self.fit(X).transform(X)

    def bin_center(self, col: int, bin_idx: int) -> float:
        """Inverse of numeric binning: approximate value at bin centre."""
        centers = self._bin_centers.get(col, [])
        if not centers:
            return 0.0
        return centers[min(bin_idx, len(centers) - 1)]

    @property
    def n_features(self) -> int:
        return self._n_features

    # ── internal ──────────────────────────────────────────────────────────────

    def _encode_row(self, row: list) -> list:
        tokens = []
        for j, v in enumerate(row):
            tokens.append((j, self._encode_val(j, v)))
        return tokens

    def _encode_val(self, j: int, v) -> Any:
        if _is_missing(v):
            return MISSING_STR
        if self._types[j] == 'numeric':
            return _bin_search(float(v), self._edges[j])
        else:
            return self._cat_maps[j].get(_safe_str(v), MISSING_STR)

    def __repr__(self) -> str:
        return (f'FeatureDiscretizer(n_bins={self.n_bins}, '
                f'n_features={self._n_features})')


# ══════════════════════════════════════════════════════════════════════════════
# LabelEncoder
# ══════════════════════════════════════════════════════════════════════════════

class LabelEncoder:
    """
    Bi-directional map between raw class labels and integer codes.
    New labels seen during partial_fit are assigned the next integer.
    """

    def __init__(self):
        self._enc: dict = {}   # label → int
        self._dec: dict = {}   # int   → label
        self.classes_: list = []

    def fit(self, y) -> 'LabelEncoder':
        for label in y:
            self._add(label)
        return self

    def partial_fit(self, y) -> 'LabelEncoder':
        return self.fit(y)

    def encode(self, label) -> int:
        if label not in self._enc:
            self._add(label)
        return self._enc[label]

    def decode(self, code: int):
        return self._dec[code]

    def __len__(self) -> int:
        return len(self._enc)

    def _add(self, label):
        if label not in self._enc:
            code = len(self._enc)
            self._enc[label] = code
            self._dec[code]  = label
            self.classes_.append(label)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _to_rows(X) -> list:
    """Accept numpy array, pandas DataFrame, or list-of-lists."""
    try:
        import numpy as np
        if isinstance(X, np.ndarray):
            return X.tolist()
    except ImportError:
        pass
    try:
        import pandas as pd
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.values.tolist()
    except ImportError:
        pass
    # Already a list; wrap scalars (1-D input → single-column matrix)
    out = list(X)
    if out and not isinstance(out[0], (list, tuple)):
        out = [[v] for v in out]
    return out


def _is_missing(v) -> bool:
    if v is None or v is MISSING:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and v.strip() == '':
        return True
    return False


def _is_numeric_col(col: list) -> bool:
    """True if any non-missing value is a float/int (non-bool)."""
    for v in col:
        if _is_missing(v):
            continue
        if isinstance(v, bool):
            return False
        if isinstance(v, (int, float)):
            return True
        return False
    return False


def _safe_str(v) -> str:
    return str(v)


def _quantile_edges(col: list, n_bins: int) -> tuple:
    """
    Compute (n_bins-1) quantile cut-points and n_bins bin centres.
    Values beyond edges map to the first or last bin (clamped).
    """
    valid = sorted(float(v) for v in col if not _is_missing(v)
                   and not (isinstance(v, float) and math.isnan(v)))
    if not valid:
        return [], [0.0]

    n = len(valid)
    edges = []
    for i in range(1, n_bins):
        idx = int(i * n / n_bins)
        edges.append(valid[min(idx, n - 1)])
    # Deduplicate while keeping order
    edges = sorted(set(edges))

    # Bin centres: midpoint between consecutive edges
    boundaries = [valid[0] - 1e-9] + edges + [valid[-1] + 1e-9]
    centers = [
        (boundaries[i] + boundaries[i + 1]) / 2.0
        for i in range(len(boundaries) - 1)
    ]
    return edges, centers


def _bin_search(v: float, edges: list) -> int:
    """Binary-search bin index for value v given sorted cut-points."""
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) // 2
        if v <= edges[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo
