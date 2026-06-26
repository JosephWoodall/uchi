import json
from pathlib import Path

# Fallback bootstrap if JSON doesn't exist yet
try:
    from .ontology import ONTOLOGY as BOOTSTRAP_ONTOLOGY
except ImportError:
    BOOTSTRAP_ONTOLOGY = {}

class OntologyManager:
    """
    Manages the real-time dynamic semantic ontology for Uchi.
    Reads/writes to `ontology.json` to perpetually learn new slang and typos.
    """
    def __init__(self, filepath: str | Path = "ontology.json"):
        self.filepath = Path(filepath)
        self.mapping: dict[str, str] = {}
        self._load()

    def _load(self):
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.mapping = json.load(f)
            except Exception:
                self.mapping = BOOTSTRAP_ONTOLOGY.copy()
        else:
            self.mapping = BOOTSTRAP_ONTOLOGY.copy()
            self._save()

    def _save(self):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.mapping, f, indent=4)
        except Exception:
            pass

    def get(self, word: str) -> str | None:
        """Returns the canonical mapping if it exists, else None."""
        return self.mapping.get(word.lower())

    def add_mapping(self, slang: str, canonical: str):
        """Adds a new slang -> canonical mapping in real-time."""
        slang = slang.lower().strip(",.")
        canonical = canonical.lower().strip(",.")
        if slang not in self.mapping or self.mapping[slang] != canonical:
            self.mapping[slang] = canonical
            self._save()
