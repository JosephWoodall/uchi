import json
import os
from typing import Optional

class ProceduralMemory:
    """JSON-backed store of task-type → procedure step lists."""

    _DEFAULTS = {
        "physics":    ["identify relevant formula", "substitute values and compute"],
        "code":       ["write the function or script", "verify syntax if possible"],
        "math":       ["break into arithmetic steps", "compute each step"],
        "search":     ["use web_search to retrieve relevant knowledge", "synthesize retrieved context into answer"],
        "document":   ["ingest document", "retrieve relevant sections"],
    }

    def __init__(self, path: str = "uchi_procedural_memory.json"):
        self.path = path
        if os.path.exists(path):
            with open(path) as f:
                self._store = json.load(f)
        else:
            self._store = dict(self._DEFAULTS)
            self._save()

    # Synonym map: query terms → store key
    _SYNONYMS = {
        "python": "code", "function": "code", "script": "code",
        "class": "code", "debug": "code", "program": "code",
        "formula": "physics", "force": "physics", "energy": "physics",
        "velocity": "physics", "kinetic": "physics", "momentum": "physics",
        "calculate": "math", "equation": "math", "compute": "math",
        "retrieve": "search", "find": "search", "look up": "search",
        "document": "document", "pdf": "document", "file": "document",
    }

    def retrieve(self, query: str) -> Optional[str]:
        q = query.lower()
        # Direct key match
        for key, steps in self._store.items():
            if key in q:
                return f"Procedure ({key}): " + " → ".join(steps)
        # Synonym match
        for term, key in self._SYNONYMS.items():
            if term in q and key in self._store:
                return f"Procedure ({key}): " + " → ".join(self._store[key])
        return None

    def update(self, task_type: str, step: str):
        if task_type not in self._store:
            self._store[task_type] = [step]
        elif step not in self._store[task_type]:
            self._store[task_type].append(step)
        self._save()

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self._store, f, indent=2)
