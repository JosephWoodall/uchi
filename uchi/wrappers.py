"""
Predictor Wrappers
==================
Wrappers that seamlessly extend the capabilities of the core engine
without polluting the O(k) mathematical purity of the base UniversalPredictor.
"""

from typing import Any
from .online_tokenizer import OnlineTokenizer
from .predictor import UniversalPredictor

class InfiniteContextPredictor:
    """
    Wrapper that transparently adds Infinite Context (Phase 5) to any core predictor.
    It buffers raw observations, compresses them on the fly using the OnlineTokenizer,
    and feeds the compressed stream to the underlying engine.

    This breaks the O(V^k) and fixed 'k' limits natively.
    """

    def __init__(self, predictor: UniversalPredictor, tokenizer: OnlineTokenizer = None, max_merges: int = 64, buffer_size: int = 500):
        self.predictor = predictor
        self.tokenizer = tokenizer or OnlineTokenizer(max_merges=max_merges)
        self.raw_history = []
        self.buffer_size = buffer_size

    def observe(self, value: Any) -> 'InfiniteContextPredictor':
        """Buffer raw values."""
        self.raw_history.append(value)
        if len(self.raw_history) > self.buffer_size:
            self.raw_history.pop(0)
        return self

    def predict(self) -> tuple[Any, float]:
        """Temporarily inject the compressed history and predict."""
        if not self.raw_history:
            return self.predictor.predict()
            
        compressed = self.tokenizer.tokenize(self.raw_history)
        saved = self.predictor.history
        self.predictor.history = compressed
        
        pred, conf = self.predictor.predict()
        
        self.predictor.history = saved
        return pred, conf

    def feedback(self, actual: Any) -> None:
        """Update the underlying trie and the tokenizer's merge rules."""
        if not self.raw_history:
            return

        compressed = self.tokenizer.tokenize(self.raw_history)
        
        # The 'actual' token might have been merged. 
        # The last token in the compressed history represents the current state.
        compressed_actual = compressed[-1]

        saved = self.predictor.history
        self.predictor.history = compressed
        
        self.predictor.feedback(compressed_actual)
        
        self.predictor.history = saved
        
        # Update the tokenizer with the accuracy of the last prediction
        abstained = getattr(self.predictor, '_last_abstained', False)
        last_pred = getattr(self.predictor, '_last_prediction', None)
        
        # Determine if the predictor was right (if it abstained, it gets 0 accuracy for this step)
        correct = 1.0 if (not abstained and last_pred == compressed_actual) else 0.0
        
        # Only evaluate tokenizer merges if we have a decent window
        if len(self.raw_history) >= 10:
            self.tokenizer.update(self.raw_history[-100:], predictor_accuracy=correct)

    @property
    def _last_distribution(self):
        return self.predictor._last_distribution

    @property
    def _last_prediction(self):
        return self.predictor._last_prediction

    @property
    def history(self):
        return self.raw_history
        
    @history.setter
    def history(self, val):
        self.raw_history = val

    @property
    def k(self):
        return self.predictor.k

    @property
    def _vocab(self):
        return self.predictor._vocab
