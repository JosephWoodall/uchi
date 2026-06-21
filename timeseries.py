# Root shim — kept for backward compatibility with research scripts.
# Source of truth: uchi/timeseries.py
from uchi.timeseries import *  # noqa: F401,F403
from uchi.timeseries import (  # noqa: F401
    MultivariateTSPredictor, TimeSeriesClassifier, AnomalyDetector,
)
