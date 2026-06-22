"""
Phase 5: Associative Memory (Zero-Shot Fractal Attention)
=========================================================
Provides a Non-Parametric geometric buffer for zero-shot Question Answering.
Implements Dynamic Fractal Attention: calculates global co-occurrence weights 
to mathematically mimic Multi-Headed Self-Attention, allowing Uchi to bridge 
long-range dependencies beyond the fixed window.
"""
from collections import defaultdict

class AssociativeMemory:
    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: The local context window size.
        """
        self.window_size = window_size
        self.buffer = []
        # Global co-occurrence graph for Fractal Attention
        self.co_occurrence = defaultdict(lambda: defaultdict(float))
        
    def stream_context(self, sequence: list):
        """
        Populate the geometric memory buffer and build the Fractal Attention graph.
        """
        if len(sequence) <= self.window_size:
            return
            
        for i in range(len(sequence)):
            start = max(0, i - self.window_size)
            end = min(len(sequence), i + self.window_size + 1)
            
            window = sequence[start:i] + sequence[i+1:end]
            target = sequence[i]
            
            # Store the local window for exact matching
            self.buffer.append((set(window), target))
            
            # Update the global Fractal Attention graph
            for w in window:
                self.co_occurrence[target][w] += 1.0
                self.co_occurrence[w][target] += 1.0 # Symmetric relationship
                
    def _expand_query(self, q_sequence: list, top_k: int = 3) -> dict:
        """
        Fractal Attention: Expands the query sequence using global co-occurrence weights.
        """
        q_weights = {q: 1.0 for q in q_sequence}
        
        for q in q_sequence:
            if q in self.co_occurrence:
                # Get the strongest co-occurring concepts
                strongest = sorted(self.co_occurrence[q].items(), key=lambda x: x[1], reverse=True)[:top_k]
                for related_concept, weight in strongest:
                    if related_concept not in q_weights:
                        # Apply a fractional attention weight
                        q_weights[related_concept] = 0.5 
                        
        return q_weights
            
    def query(self, q_sequence: list) -> str:
        """
        Calculates the mathematical overlap using Fractal Attention weights.
        """
        if not self.buffer:
            return None
            
        # 1. Expand the query dynamically using global attention weights
        q_weights = self._expand_query(q_sequence)
        
        best_score = -1
        best_target = None
        
        # 2. Score against the buffer
        for window_set, target in self.buffer:
            score = 0.0
            for w in window_set:
                if w in q_weights:
                    score += q_weights[w]
                    
            if score > best_score:
                best_score = score
                best_target = target
                
        return best_target

