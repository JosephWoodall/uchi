import json
import os
from typing import Optional

class ProceduralMemory:
    """JSON-backed store of task-type → procedure step lists."""

    _DEFAULTS = {
        "physics":    ["identify relevant formula", "substitute values and compute"],
        "code":       ["write the function or script", "verify syntax if possible"],
        "math":       ["break into arithmetic steps", "compute each step"],
        "search":     ["use web_search to retrieve relevant knowledge", "synthesize retrieved context into answer"],
        "document":   ["ingest document", "retrieve relevant sections"],
        # Analytical ML intents
        "classify":   ["load data file", "train TabularPredictor", "report accuracy"],
        "regress":    ["load data file", "train TabularRegressor", "report mean absolute error"],
        "anomaly":    ["load data file", "train AnomalyDetector", "report anomalous rows"],
        "forecast":   ["load time series", "fit MultivariateTSPredictor", "forecast N steps ahead"],
        "tsclassify": ["load window data", "train TimeSeriesClassifier", "report accuracy"],
    }

    def __init__(self, path: str = "uchi_procedural_memory.json"):
        self.path = path
        if os.path.exists(path):
            with open(path) as f:
                self._store = json.load(f)
        else:
            self._store = dict(self._DEFAULTS)
            self._save()

    # Synonym map: query terms → intent key
    _SYNONYMS = {
        # Code
        "python": "code", "function": "code", "script": "code",
        "debug": "code", "program": "code", "implement": "code",
        # Physics / math (note: "class" kept for code, not misrouted)
        "formula": "physics", "force": "physics", "energy": "physics",
        "velocity": "physics", "kinetic": "physics", "momentum": "physics",
        "calculate": "math", "equation": "math", "compute": "math",
        # Search / document
        "retrieve": "search", "look up": "search",
        "document": "document", "pdf": "document",
        # Classification
        "classify": "classify", "classification": "classify",
        "predict class": "classify", "label": "classify",
        "categorize": "classify", "category": "classify",
        "churn": "classify", "diagnose": "classify",
        # Regression
        "regression": "regress", "regress": "regress",
        "predict value": "regress", "estimate": "regress",
        # Anomaly detection
        "anomaly": "anomaly", "anomalies": "anomaly",
        "outlier": "anomaly", "outliers": "anomaly",
        "unusual": "anomaly", "abnormal": "anomaly",
        "suspicious": "anomaly", "weird": "anomaly",
        "strange": "anomaly", "detect": "anomaly",
        # Forecasting
        "forecast": "forecast", "forecasting": "forecast",
        "predict future": "forecast", "future": "forecast",
        "next step": "forecast", "trend": "forecast",
        # Time series classification
        "time series": "tsclassify", "timeseries": "tsclassify",
        "window": "tsclassify", "ecg": "tsclassify", "har": "tsclassify",
    }

    def retrieve(self, query: str) -> Optional[str]:
        q = query.lower()
        for key, steps in self._store.items():
            if key in q:
                return f"Procedure ({key}): " + " → ".join(steps)
        for term, key in self._SYNONYMS.items():
            if term in q and key in self._store:
                return f"Procedure ({key}): " + " → ".join(self._store[key])
        return None

    def get_intent_key(self, query: str) -> Optional[str]:
        """Return the raw intent key for this query, without the formatted procedure."""
        q = query.lower()
        for key in self._store:
            if key in q:
                return key
        for term, key in self._SYNONYMS.items():
            if term in q and key in self._store:
                return key
        return None

    def update(self, task_type: str, step: str):
        if task_type not in self._store:
            self._store[task_type] = [step]
        elif step not in self._store[task_type]:
            self._store[task_type].append(step)
        self._save()

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self._store, f, indent=2)
