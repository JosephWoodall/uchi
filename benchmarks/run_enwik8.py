import sys
import time

# We must add the parent directory to sys.path so we can import uchi
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from uchi.generative import SequenceGenerator

def main():
    print("==================================================")
    print("   UCHI ENWIK8 COMPRESSION BENCHMARK")
    print("==================================================")
    
    file_path = "enwik8_subset.txt"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
        
    print(f"Loading dataset...")
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
        
    # We will test on 500,000 characters to show the curve without taking 24 hours
    n_chars = min(len(text), 500_000)
    text = text[:n_chars]
    
    print(f"Loaded {n_chars} characters.")
    print("Initializing Infinite Context SequenceGenerator (Pure CTW/PPM-Star)...")
    
    # Initialize with infinite context window
    generator = SequenceGenerator(context_length=None)
    
    # Force fitting flag so we can use partial_fit/score manually
    generator.is_fitted_ = True
    
    from uchi.tabular import _make_predictor
    # Create the core predictor (we do it directly for speed and manual tracking)
    pred = _make_predictor(k=None, lr=0.08, cred_max=6.05, lp=0.65)
    
    import math
    total_entropy = 0.0
    interval = 50_000
    
    print("\nStarting Streaming Evaluation:")
    print("Characters Seen | Avg Bits/Char")
    print("-" * 35)
    
    start_time = time.time()
    
    for i, char in enumerate(text):
        # 1. Predict next character
        pred.predict()
        
        # 2. Score the actual character
        prob = pred._last_distribution.get(char, 1e-12)
        total_entropy += -math.log2(max(prob, 1e-12))
        
        # 3. Observe the character
        pred.observe(char)
        pred.feedback(char)
        
        # Periodically flush history to avoid infinite array growth (simulate sliding window of 256 for speed)
        if len(pred.history) > 256:
            pred.history = pred.history[-256:]
            
        if (i + 1) % interval == 0:
            avg_bpc = total_entropy / (i + 1)
            print(f"{i + 1:<15} | {avg_bpc:.4f}")
            
    end_time = time.time()
    print("-" * 35)
    print(f"Final Score: {total_entropy / n_chars:.4f} bits/character")
    print(f"Time Taken:  {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
