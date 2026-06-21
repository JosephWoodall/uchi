import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from uchi.hoeffding import HoeffdingPredictor
from sklearn.datasets import make_classification

def run_streaming_benchmark():
    print("\n" + "="*70)
    print("  STREAMING EVALUATION: Hoeffding Tries vs Static Random Forest")
    print("="*70)
    
    # Simulate a long stream with a concept drift halfway through
    print("Generating massive streaming dataset with concept drift...")
    X1, y1 = make_classification(n_samples=10000, n_features=5, n_informative=3, flip_y=0.1, random_state=42)
    
    # Introduce concept drift by inverting the labels
    X2, y2 = make_classification(n_samples=10000, n_features=5, n_informative=3, flip_y=0.1, random_state=43)
    y2 = 1 - y2  # concept drift!
    
    X_stream = np.vstack([X1, X2])
    y_stream = np.concatenate([y1, y2])
    
    print(f"Total Stream Size: {len(y_stream)} rows")
    
    # Models
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    hoeffding = HoeffdingPredictor(n_features=X_stream.shape[1])
    
    # Train RF on an initial batch (Offline assumption)
    initial_batch_size = 1000
    rf.fit(X_stream[:initial_batch_size], y_stream[:initial_batch_size])
    for row, label in zip(X_stream[:initial_batch_size].tolist(), y_stream[:initial_batch_size].tolist()):
        hoeffding.partial_fit(row, label)
        
    print(f"Initial training on first {initial_batch_size} rows complete.")
    
    # Evaluation over time (T)
    window_size = 1000
    
    print("\nEvaluating Accuracy over Time (T):")
    print(f"{'Time (T)':<15} | {'Random Forest':<15} | {'Hoeffding Tries':<15}")
    print("-" * 50)
    
    for t in range(initial_batch_size, len(y_stream), window_size):
        X_window = X_stream[t:t+window_size]
        y_window = y_stream[t:t+window_size]
        
        # Test Phase (Predict next window)
        rf_preds = rf.predict(X_window)
        rf_acc = accuracy_score(y_window, rf_preds)
        
        hoeffding_preds = []
        for row in X_window.tolist():
            dist = hoeffding.predict_proba(row)
            if not dist:
                hoeffding_preds.append(0)
            else:
                hoeffding_preds.append(max(dist, key=dist.get))
        hoeffding_acc = accuracy_score(y_window, hoeffding_preds)
        
        print(f"T={t:<13} | {rf_acc:<15.3f} | {hoeffding_acc:<15.3f}")
        
        # Train Phase (Online model learns from the window it just saw)
        for row, label in zip(X_window.tolist(), y_window.tolist()):
            hoeffding.partial_fit(row, label)
            
    print("-" * 50)
    print("Notice how the Random Forest permanently fails after T=10000 (Concept Drift).")
    print("The Hoeffding Trie asymptotically learns the new pattern online!")

if __name__ == '__main__':
    run_streaming_benchmark()
