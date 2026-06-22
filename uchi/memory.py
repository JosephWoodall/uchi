"""
Phase 5: Associative Memory (Zero-Shot Attention)
=================================================
Provides a Non-Parametric geometric buffer for zero-shot Question Answering.
When combined with the SemanticTokenizer, it mathematically mimics LLM
attention heads by binding ad-hoc variables directly to the generative output.
"""

class AssociativeMemory:
    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: The number of tokens in the context window.
        """
        self.window_size = window_size
        self.buffer = []
        
    def stream_context(self, sequence: list):
        """
        Populate the geometric memory buffer.
        Extracts sliding windows of `window_size` and their subsequent target token.
        """
        if len(sequence) <= self.window_size:
            return
            
        for i in range(len(sequence)):
            # Symmetric window: window_size tokens before, window_size tokens after
            start = max(0, i - self.window_size)
            end = min(len(sequence), i + self.window_size + 1)
            
            window = sequence[start:i] + sequence[i+1:end]
            target = sequence[i]
            
            # We store the window as a set for O(1) intersection testing later
            self.buffer.append((set(window), target))
            
    def query(self, q_sequence: list) -> str:
        """
        Calculates the mathematical overlap (dot-product / Jaccard similarity)
        between the query sequence and the memory buffer.
        Returns the target token of the maximally overlapping context.
        """
        if not self.buffer:
            return None
            
        q_set = set(q_sequence)
        best_score = -1
        best_target = None
        
        for window_set, target in self.buffer:
            score = len(q_set.intersection(window_set))
            if score > best_score:
                best_score = score
                best_target = target
                
        return best_target
