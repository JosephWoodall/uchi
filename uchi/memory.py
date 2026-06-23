"""
Phase 5: Associative Memory (Zero-Shot Fractal Attention)
=========================================================
Provides a Non-Parametric geometric buffer for zero-shot Question Answering.
Implements Dynamic Fractal Attention: calculates global co-occurrence weights 
to mathematically mimic Multi-Headed Self-Attention, allowing Uchi to bridge 
long-range dependencies beyond the fixed window.
"""
from collections import defaultdict
import torch
from .cpu_memory import CPUVectorMemory

def _default_dict_float():
    return defaultdict(float)

class AssociativeMemory:
    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: The local context window size.
        """
        self.window_size = window_size
        self.cpu_mem = CPUVectorMemory()
        # Global co-occurrence graph for Fractal Attention
        self.co_occurrence = defaultdict(_default_dict_float)

    def __setstate__(self, state):
        """Forward-compatibility: initialize attributes added in newer versions after unpickling."""
        self.__dict__.update(state)
        if not hasattr(self, 'cpu_mem'):
            self.cpu_mem = CPUVectorMemory()
        if not hasattr(self, 'co_occurrence'):
            self.co_occurrence = defaultdict(_default_dict_float)
        # Old versions stored buffer list — no longer used, safe to drop
        if hasattr(self, 'buffer'):
            del self.buffer
        
    def stream_context(self, sequence: list):
        """
        Populate the geometric memory buffer and build the Fractal Attention graph.
        """
        if len(sequence) <= self.window_size:
            return
            
        from uchi.neuro_symbolic import get_ssm
        ssm = get_ssm()
            
        # We store the full sequence as the retrieved target to provide complete context.
        # This solves the issue where single-word targets like 'understand' would tie with the actual answer.
        full_sentence = ' '.join([str(s) for s in sequence if not (isinstance(s, str) and s.startswith('<|'))])
        
        # Get semantic state vector
        state_vec = ssm.get_state(sequence)
        
        self.cpu_mem.add_memory(full_sentence, state_vec.detach().cpu().numpy())
        
        for i in range(len(sequence)):
            start = max(0, i - self.window_size)
            end = min(len(sequence), i + self.window_size + 1)
            
            window = sequence[start:i] + sequence[i+1:end]
            target = sequence[i]
            
            # Update the global Fractal Attention graph
            for w in window:
                if not (isinstance(w, str) and w.startswith("<|") and w.endswith("|>")):
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
            
    def query(self, q_sequence: list) -> tuple:
        """
        Retrieves best matching memory using SSM cosine similarity.
        Returns (text, cosine_score) where score is in [-1, 1].
        A score >= 0.5 is a confident match; caller should threshold accordingly.
        """
        from uchi.neuro_symbolic import get_ssm

        if not self.cpu_mem.records:
            return None, None

        ssm = get_ssm()
        q_state = ssm.get_state(q_sequence)

        results = self.cpu_mem.retrieve_with_scores(q_state.detach().cpu().numpy(), top_k=1)
        if results:
            text, score = results[0]
            return text, score
        return None, None
