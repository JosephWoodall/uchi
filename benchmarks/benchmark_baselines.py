#!/usr/bin/env python3
"""
benchmark_baselines.py
======================
Benchmarks markov_exploration against standard ML/statistical baselines.
"""

import math
import time
import numpy as np
from collections import defaultdict, Counter

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# ── project imports ──────────────────────────────────────────────────────────

import sys
sys.path.insert(0, '.')
from uchi.tabular import TabularPredictor, TabularRegressor
from uchi.hoeffding import HoeffdingPredictor
from uchi.timeseries import MultivariateTSPredictor
from uchi.generative import SequenceGenerator
from uchi.distributional import DistributionalTokenizer
from tests import _datasets

# ── Baselines ────────────────────────────────────────────────────────────────

class NgramBaseline:
    """Simple Maximum Likelihood N-gram model."""
    def __init__(self, n=3):
        self.n = n
        self.counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab = set()
        
    def fit(self, tokens):
        for i in range(len(tokens) - self.n):
            ctx = tuple(tokens[i:i+self.n-1])
            nxt = tokens[i+self.n-1]
            self.counts[ctx][nxt] += 1
            self.context_counts[ctx] += 1
            self.vocab.add(nxt)
            
    def score(self, tokens):
        total_bits = 0.0
        vocab_size = len(self.vocab)
        for i in range(len(tokens) - self.n):
            ctx = tuple(tokens[i:i+self.n-1])
            nxt = tokens[i+self.n-1]
            
            ctx_c = self.context_counts[ctx]
            if ctx_c > 0:
                prob = self.counts[ctx][nxt] / ctx_c
                # Laplace smoothing for zero-prob tokens to avoid inf
                if prob == 0:
                    prob = 1.0 / (ctx_c + vocab_size)
            else:
                prob = 1.0 / vocab_size
                
            total_bits += -math.log2(max(prob, 1e-12))
            
        return total_bits / max(1, (len(tokens) - self.n))


class AutoRegressiveRF:
    """Time series forecasting using RandomForest on lagged features."""
    def __init__(self, lags=3):
        self.lags = lags
        self.rf = RandomForestRegressor(n_estimators=50, random_state=42)
        
    def fit(self, series):
        X, y = [], []
        # series is list of scalars or vectors
        is_multivariate = isinstance(series[0], (list, tuple))
        
        for i in range(len(series) - self.lags):
            window = series[i:i+self.lags]
            if is_multivariate:
                flat_window = [val for pt in window for val in pt]
                target = series[i+self.lags]
            else:
                flat_window = window
                target = series[i+self.lags]
            X.append(flat_window)
            y.append(target)
            
        self.rf.fit(X, y)
        self._is_multi = is_multivariate
        
    def predict_stream(self, series):
        preds = []
        for i in range(len(series) - self.lags):
            window = series[i:i+self.lags]
            if self._is_multi:
                flat_window = [val for pt in window for val in pt]
            else:
                flat_window = window
            pred = self.rf.predict([flat_window])[0]
            preds.append(pred)
        return preds


# ── Benchmark functions ──────────────────────────────────────────────────────

def benchmark_tabular_classification():
    print("\n" + "="*70)
    print("  TABULAR CLASSIFICATION: markov vs RandomForest")
    print("="*70)
    
    # Generate some synthetic tabular classification data
    np.random.seed(42)
    X = np.random.randn(1000, 5)
    # Target is a non-linear combination of features
    y = ((X[:, 0] * X[:, 1] > 0) ^ (X[:, 2] > 0)).astype(int)
    
    # Split
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    
    # Baseline: Random Forest
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_time = time.time() - t0
    
    # Markov Explorer:    
    # Markov Tabular (Hoeffding Tries)
    t0 = time.time()
    mp_model = HoeffdingPredictor(n_features=X_train.shape[1], grace_period=50)
    for row, label in zip(X_train.tolist(), y_train.tolist()):
        mp_model.partial_fit(row, label)
    
    y_pred_mp = []
    for row in X_test.tolist():
        dist = mp_model.predict_proba(row)
        if not dist:
            y_pred_mp.append(0) # fallback
        else:
            y_pred_mp.append(max(dist, key=dist.get))
            
    mp_acc = accuracy_score(y_test, y_pred_mp)
    mp_time = time.time() - t0
    
    print(f"  RandomForest Acc : {rf_acc:.3f}  (Time: {rf_time:.3f}s)")
    print(f"  Hoeffding Tries  : {mp_acc:.3f}  (Time: {mp_time:.3f}s)")
    
    if mp_acc > rf_acc:
        print("  ✓ Markov Explorer outperformed Baseline!")
    else:
        print("  ✗ Baseline performed better.")

def benchmark_timeseries():
    print("\n" + "="*70)
    print("  TIME SERIES FORECASTING: markov vs Autoregressive RF")
    print("="*70)
    
    # Synthetic time series with seasonality and noise
    t = np.linspace(0, 100, 1000)
    series = np.sin(t) + np.sin(3*t)*0.5 + np.random.randn(1000)*0.1
    series = series.tolist()
    
    train, test = series[:800], series[800:]
    
    # Baseline: AR-RF
    t0 = time.time()
    ar = AutoRegressiveRF(lags=5)
    ar.fit(train)
    rf_preds = ar.predict_stream(test)
    rf_mse = mean_squared_error(test[5:], rf_preds)
    rf_time = time.time() - t0
    
    # Markov Explorer: MultivariateTSPredictor
    t0 = time.time()
    # To use the TS predictor, we wrap scalars in lists for multivariate interface
    train_multi = [[v] for v in train]
    test_multi = [[v] for v in test]
    mp = MultivariateTSPredictor(n_bins=20, context_length=5)
    mp.fit(train_multi)
    
    # Predict step by step
    mp_preds = []
    mp._pred.history = []
    for i in range(len(test_multi) - 5):
        # build history
        window = test_multi[i:i+5]
        # reset and observe window
        mp._pred.history = []
        for pt in window:
            mp.observe(pt)
            
        pred_means = mp.predict()
        mp_preds.append(pred_means[0])
        
    mp_mse = mean_squared_error(test[5:], mp_preds)
    mp_time = time.time() - t0
    
    print(f"  AR-RandomForest MSE: {rf_mse:.4f}  (Time: {rf_time:.3f}s)")
    print(f"  Markov TS Predictor: {mp_mse:.4f}  (Time: {mp_time:.3f}s)")
    
    if mp_mse < rf_mse:
        print("  ✓ Markov Explorer outperformed Baseline!")
    else:
        print("  ✗ Baseline performed better.")

def benchmark_generative():
    print("\n" + "="*70)
    print("  GENERATIVE SEQUENCE: markov vs N-gram MLE")
    print("="*70)
    
    datasets = _datasets()
    for name, seq, sim_fn, ctx_len in datasets:
        if seq is None or len(seq) < 1000:
            continue
            
        print(f"\n  Dataset: {name}")
        train, test = seq[:int(len(seq)*0.8)], seq[int(len(seq)*0.8):]
        
        # Baseline: N-gram MLE
        t0 = time.time()
        ngram = NgramBaseline(n=ctx_len)
        ngram.fit(train)
        ng_bpt = ngram.score(test)
        ng_time = time.time() - t0
        
        # Markov: SequenceGenerator (Infinite Context + Distributional Semantics)
        t0 = time.time()
        sg = SequenceGenerator(
            context_length=None, 
            use_online_tokenizer=True,
            use_semantic_hashing=False,
            use_skip_grams=False
        )
        # Manually plug in DistributionalTokenizer
        sg.semantic_tokenizer = DistributionalTokenizer(merge_threshold_jsd=0.2)
        sg.fit(train)
        sg_bpt = sg.score(test)
        sg_time = time.time() - t0
        
        print(f"    N-gram ({ctx_len}-gram) bits/token: {ng_bpt:.3f}  (Time: {ng_time:.3f}s)")
        print(f"    Markov Generator bits/token : {sg_bpt:.3f}  (Time: {sg_time:.3f}s)")
        
        if sg_bpt < ng_bpt:
            print("    ✓ Markov Explorer outperformed Baseline!")
        else:
            print("    ✗ Baseline performed better.")

if __name__ == "__main__":
    benchmark_tabular_classification()
    benchmark_timeseries()
    benchmark_generative()
    print("\n  Benchmarking Complete!")
