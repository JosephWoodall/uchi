"""Prompt Cache — LRU cache for SSM hidden states keyed on tokenized prefix hash.
Eliminates redundant prefill computation for repeated system prompts and common prefixes."""

import hashlib
from collections import OrderedDict
from typing import Optional, Tuple, List
import torch


class PromptCache:
    """LRU cache storing SSM hidden states at prefix boundaries.
    
    Key: SHA-256 hash of token ID sequence
    Value: (hidden_states_per_layer, last_position_index)
    """

    def __init__(self, max_entries: int = 64):
        self.max_entries = max_entries
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _key(self, token_ids: List[int]) -> str:
        return hashlib.sha256(str(token_ids).encode()).hexdigest()

    def get(self, token_ids: List[int]) -> Optional[dict]:
        """Look up cached hidden states for a token prefix.
        
        Returns dict with 'hidden_states' and 'position' if found, else None.
        Tries progressively shorter prefixes for partial hits.
        """
        # Try exact match first
        key = self._key(token_ids)
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]

        # Try longest matching prefix (powers of 2 for efficiency)
        for length in [len(token_ids) // 2, len(token_ids) // 4]:
            if length < 10:
                break
            prefix_key = self._key(token_ids[:length])
            if prefix_key in self._cache:
                self._cache.move_to_end(prefix_key)
                self.hits += 1
                return self._cache[prefix_key]

        self.misses += 1
        return None

    def put(self, token_ids: List[int], hidden_states: list,
            position: int):
        """Cache hidden states for a token prefix."""
        key = self._key(token_ids)
        entry = {
            "hidden_states": [h.clone() for h in hidden_states],
            "position": position,
        }
        self._cache[key] = entry
        self._cache.move_to_end(key)

        # Evict oldest if over capacity
        while len(self._cache) > self.max_entries:
            evicted_key, evicted_val = self._cache.popitem(last=False)
            # Free tensors
            for h in evicted_val["hidden_states"]:
                del h

    def clear(self):
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "entries": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / max(total, 1),
        }
