"""
uchi
===============
Online, instance-based sequence predictor — no neural weights, no pre-training,
zero catastrophic forgetting. Runs tabular classification, regression, time
series forecasting, anomaly detection, and generative modeling all from a
single entry point.

Quick start
-----------
    from uchi import Uchi

    u = Uchi()
    u.learn("Q3 revenue was $4.2M, up 23% YoY.")
    print(u.ask("What was Q3 revenue growth?"))

    # Analytical tools
    report = u.ask("/classify", X=X_train, y=y_train)

    # Compounding — ask() always returns str, learn() always accepts str
    u2 = Uchi()
    u2.learn(report)
    u2.ask("What does this classification imply for Q4?")

All classes are sklearn-compatible (Pipeline, GridSearchCV, cross_val_score).
TabularPredictor / TabularRegressor / TimeSeriesClassifier all support
partial_fit() for online / incremental learning.
"""

__version__ = "0.3.0"

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

from .memory            import AssociativeMemory
from .omni_router       import OmniRouter
from .omni_tokenizer    import OmniTokenizer
from .simple            import Uchi

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

    "AssociativeMemory",
    "OmniRouter",
    "OmniTokenizer",
    "Uchi",
]
