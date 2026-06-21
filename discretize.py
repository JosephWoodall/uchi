# Root shim — kept for backward compatibility with research scripts.
# Source of truth: uchi/discretize.py
from uchi.discretize import *  # noqa: F401,F403
from uchi.discretize import (  # noqa: F401
    FeatureDiscretizer, LabelEncoder, MISSING, MISSING_STR,
    _to_rows, _is_missing, _is_numeric_col, _safe_str,
    _quantile_edges, _bin_search,
)
