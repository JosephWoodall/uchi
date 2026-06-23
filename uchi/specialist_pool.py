"""
specialist_pool.py
==================
Phase 2: MoE SpecialistPool — three specialized brains.

ProceduralMemory already routes queries to intent categories.
SpecialistPool maps those categories to separate brain files trained on
domain-specific corpora. Each specialist is lazy-loaded on first use.

Brain files:
  brain_code.uchi  — Python stdlib patterns, algorithm skeletons
  brain_math.uchi  — arithmetic, physics formulas, numeric procedures
  brain_convo.uchi — persona, world knowledge, conversation patterns

Bootstrap: run scripts/bootstrap_specialist.py to build each brain.
"""

import os
from typing import Optional, Dict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPECIALIST_BRAINS: Dict[str, str] = {
    "code":  os.path.join(_PROJECT_ROOT, "brain_code.uchi"),
    "math":  os.path.join(_PROJECT_ROOT, "brain_math.uchi"),
    "convo": os.path.join(_PROJECT_ROOT, "brain_convo.uchi"),
}

_INTENT_TO_SPECIALIST: Dict[str, str] = {
    "code":     "code",
    "physics":  "math",
    "math":     "math",
    "search":   "convo",
    "document": "convo",
}


class SpecialistPool:
    """
    Lazy-loads specialist OmniRouter instances from separate brain files.
    Falls back to the default router when no specialist brain is available.
    """

    def __init__(self, default_router):
        self.default = default_router
        self._pool: Dict[str, object] = {}

    def route(self, intent_key: str):
        """Return specialist router for this intent, or the default router."""
        specialist_key = _INTENT_TO_SPECIALIST.get(intent_key.lower(), "convo")
        brain_path = SPECIALIST_BRAINS.get(specialist_key, "")

        if not brain_path or not os.path.exists(brain_path):
            return self.default

        if specialist_key not in self._pool:
            loaded = self._load(brain_path)
            self._pool[specialist_key] = loaded if loaded is not None else self.default

        return self._pool[specialist_key]

    def _load(self, brain_path: str) -> Optional[object]:
        from uchi.cli import load_brain
        try:
            return load_brain(brain_path)
        except Exception:
            return None

    def has_specialist(self, intent_key: str) -> bool:
        key = _INTENT_TO_SPECIALIST.get(intent_key.lower())
        if key is None:
            return False
        return os.path.exists(SPECIALIST_BRAINS.get(key, ""))

    def get_predictor(self, intent_key: str):
        """Return the trie predictor for this intent's specialist (or default)."""
        specialist = self.route(intent_key)
        return specialist.predictor
