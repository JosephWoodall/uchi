# Root shim — kept for backward compatibility with research scripts.
# Source of truth: uchi/tabular.py
from uchi.tabular import *  # noqa: F401,F403
from uchi.tabular import (  # noqa: F401
    TabularPredictor, TabularRegressor,
    _set_history, _infer_dist, _train_one,
    _make_predictor, _mi_order, _apply_order,
    _LABEL_NS, _TARGET_NS,
)
