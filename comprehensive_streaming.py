import numpy as np
import time
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datasets import load_gutenberg_text, load_electricity
from sklearn.datasets import make_classification

from uchi.hoeffding import HoeffdingPredictor
from uchi.generative import SequenceGenerator
from uchi.distributional import DistributionalTokenizer
from benchmark_baselines import NgramBaseline

# Ensure T is the same across all tasks
TOTAL_T = 20000
EVAL_WINDOW = 2000

def run_tabular_streaming():
    print("\n" + "="*70)
    print("  STREAMING EVALUATION: Tabular (Hoeffding Tries vs Random Forest)")
    print("="*70)
    
    # Simulate a stream with concept drift at T/2
    X1, y1 = make_classification(n_samples=TOTAL_T // 2, n_features=5, n_informative=3, flip_y=0.1, random_state=42)
    X2, y2 = make_classification(n_samples=TOTAL_T // 2, n_features=5, n_informative=3, flip_y=0.1, random_state=43)
    y2 = 1 - y2  # concept drift!
    
    X_stream = np.vstack([X1, X2])
    y_stream = np.concatenate([y1, y2])
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    hoeffding = HoeffdingPredictor(n_features=X_stream.shape[1], grace_period=10, delta=1e-1)
    
    initial_batch_size = 1000
    rf.fit(X_stream[:initial_batch_size], y_stream[:initial_batch_size])
    for row, label in zip(X_stream[:initial_batch_size].tolist(), y_stream[:initial_batch_size].tolist()):
        hoeffding.partial_fit(row, label)
        
    print(f"{'Time (T)':<15} | {'Random Forest Acc':<20} | {'Hoeffding Tries Acc':<20}")
    print("-" * 60)
    
    for t in range(initial_batch_size, TOTAL_T, EVAL_WINDOW):
        X_window = X_stream[t:t+EVAL_WINDOW]
        y_window = y_stream[t:t+EVAL_WINDOW]
        
        # Test Phase
        rf_preds = rf.predict(X_window)
        rf_acc = accuracy_score(y_window, rf_preds)
        
        hoeffding_preds = []
        for row in X_window.tolist():
            dist = hoeffding.predict_proba(row)
            hoeffding_preds.append(max(dist, key=dist.get) if dist else 0)
        hoeffding_acc = accuracy_score(y_window, hoeffding_preds)
        
        print(f"T={t:<13} | {rf_acc:<20.3f} | {hoeffding_acc:<20.3f}")
        
        # Train Phase (Online)
        for row, label in zip(X_window.tolist(), y_window.tolist()):
            hoeffding.partial_fit(row, label)

def run_generative_streaming():
    print("\n" + "="*70)
    print("  STREAMING EVALUATION: Generative (Infinite Markov vs 3-gram)")
    print("="*70)
    
    print("Loading Alice in Wonderland (Character Stream)...")
    text = load_gutenberg_text(n_chars=TOTAL_T + EVAL_WINDOW)
    
    # We will measure bits-per-token dynamically over time.
    ngram = NgramBaseline(n=3)
    
    markov = SequenceGenerator(
        context_length=None, # Infinite Context
        use_online_tokenizer=False,
        use_semantic_hashing=False,
        use_skip_grams=False
    )
    # Inject Distributional Semantics
    markov.semantic_tokenizer = DistributionalTokenizer(merge_threshold_jsd=0.2, min_obs=50)
    
    initial_batch = text[:1000]
    ngram.fit(initial_batch)
    markov.fit(initial_batch)
    
    print(f"{'Time (T)':<15} | {'3-Gram bits/char':<20} | {'Markov bits/char':<20}")
    print("-" * 60)
    
    for t in range(1000, TOTAL_T, EVAL_WINDOW):
        window = text[t:t+EVAL_WINDOW]
        
        # Test Phase
        ng_score = ngram.score(window)
        mk_score = markov.score(window)
        
        print(f"T={t:<13} | {ng_score:<20.3f} | {mk_score:<20.3f}")
        
        # Train Phase (Online)
        ngram.fit(window)
        markov.fit(window)

if __name__ == '__main__':
    run_tabular_streaming()
    run_generative_streaming()
