import time
import math
from uchi.predictor import UniversalPredictor
from uchi.forest import PredictorForest
from uchi_datasets import load_math_corpus

def test_math_simulation():
    print("Loading math corpus...")
    math_stream = load_math_corpus(n_chars=20_000, seed=42)
    vocab = set(math_stream)
    print(f"Math stream length: {len(math_stream)} characters")
    print(f"Vocab size: {len(vocab)} symbols: {sorted(vocab)}")
    
    # We will test both the single predictor and the forest.
    pred = UniversalPredictor(context_length=6, learning_rate=0.1)
    forest = PredictorForest(context_length=4, n_trees=5, voting='adaptive')
    
    print("\nStarting Online Streaming...")
    
    start_time = time.time()
    
    pred_log_loss = 0.0
    forest_log_loss = 0.0
    
    # Measure in chunks of 5000 tokens
    chunk_size = 5000
    
    for i, token in enumerate(math_stream):
        # Predict
        pred.predict()
        forest.predict()
        
        # Calculate log loss (bits per symbol)
        p_prob = pred._last_distribution.get(token, 1e-12) if pred._last_distribution else 1.0 / len(vocab)
        pred_log_loss += -math.log2(max(p_prob, 1e-12))
        
        pred.observe(token)
        forest.observe(token)
        
        pred.feedback(token)
        forest.feedback(token)
        
        if (i + 1) % chunk_size == 0:
            avg_loss = pred_log_loss / chunk_size
            print(f"Step {i+1:5d} | Predictor Bits/Char: {avg_loss:.4f} | Nodes: {pred.node_stats()['total_nodes']}")
            pred_log_loss = 0.0
            
    print(f"\nSimulation complete in {time.time() - start_time:.2f}s")
    print(f"Final Tree Stats:")
    print(f"Single Predictor Nodes: {pred.node_stats()['total_nodes']}")
    print(f"Forest Trees: {forest.node_stats()['n_active']}")

if __name__ == "__main__":
    run_math_simulation()
