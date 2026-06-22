"""
uchi
===============
Online credibility-weighted sequence predictor for tabular, time series,
and generative machine learning tasks.

Quick start
-----------
    from uchi import TabularPredictor, TabularRegressor
    from uchi import MultivariateTSPredictor, TimeSeriesClassifier
    from uchi import AnomalyDetector
    from uchi import UniversalPredictor, PredictorForest

All classes are sklearn-compatible (Pipeline, GridSearchCV, cross_val_score).
TabularPredictor / TabularRegressor / TimeSeriesClassifier all support
partial_fit() for online / incremental learning.
"""

__version__ = "0.2.0"

from .predictor  import UniversalPredictor
from .forest     import PredictorForest
from .discretize import FeatureDiscretizer, LabelEncoder
from .tabular    import TabularPredictor, TabularRegressor
from .timeseries import MultivariateTSPredictor, TimeSeriesClassifier, AnomalyDetector
from .generative import SequenceGenerator, TabularGenerator, TimeSeriesGenerator

# Generative services fixes
from .long_term_store   import LongTermStore
from .dual_predictor    import DualPredictor
from .online_tokenizer  import OnlineTokenizer
from .node_compressor   import NodeCompressor
from .process           import ProcessPredictor, OntologicalState, OntologicalAction
from .simulation_engine import LifelongSimulationEngine
from .wrappers          import InfiniteContextPredictor
from .memory            import AssociativeMemory
from .omni_router       import OmniRouter
from .omni_tokenizer    import OmniTokenizer

__version__ = "0.2.0"

__all__ = [
    # Core engine
    "UniversalPredictor",
    "PredictorForest",
    # Feature engineering
    "FeatureDiscretizer",
    "LabelEncoder",
    # Tabular ML
    "TabularPredictor",
    "TabularRegressor",
    # Time series
    "MultivariateTSPredictor",
    "TimeSeriesClassifier",
    "AnomalyDetector",
    # Generative
    "SequenceGenerator",
    "TabularGenerator",
    "TimeSeriesGenerator",
    # Generative services fixes
    "LongTermStore",
    "DualPredictor",
    "OnlineTokenizer",
    "NodeCompressor",
    "ProcessPredictor",
    "OntologicalState",
    "OntologicalAction",
    "LifelongSimulationEngine",
    "InfiniteContextPredictor",
    "OmniRouter",
    "OmniTokenizer"
]
