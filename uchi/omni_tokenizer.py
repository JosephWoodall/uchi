"""
Omni Tokenizer (Multi-Modal Cortex)
===================================
Abstracts multiple data modalities (Text, Math, Code, Images, Audio) 
into a universal mathematical Concept ID stream.
"""
import ast
import difflib
import numpy as np
from .process import OntologicalState, OntologicalAction

class UnknownConcept:
    """Deprecated: OOV words are now shattered via BPE fallback in OmniTokenizer."""
    def __init__(self, raw_word: str):
        self.raw_word = raw_word

    def __repr__(self):
        return f"[UNKNOWN:{self.raw_word}]"

class OmniTokenizer:
    def __init__(self, use_wordnet: bool = True):
        self.use_wordnet = use_wordnet
        self._cache = {}
        self._wn = None
        self._known_concepts = set()
        
        from .ontology_manager import OntologyManager
        self.ontology = OntologyManager("ontology.json")
        
        if self.use_wordnet:
            try:
                import nltk
                from nltk.corpus import wordnet
                self._wn = wordnet
                try:
                    self._wn.synsets('dog')
                except LookupError:
                    nltk.download('wordnet', quiet=True)
            except ImportError:
                self.use_wordnet = False

    @staticmethod
    def _bpe_fallback(word: str) -> list:
        """
        Shatter an OOV word into known subwords, guaranteeing the TokenEmbedder
        always receives valid, hashable string tokens.

        Strategy:
          1. Try tiktoken (cl100k_base) — returns byte-pair subwords as strings.
          2. Fall back to overlapping character bigrams, which always cover the word.

        The returned tokens are non-empty strings safe for the hashing embedder.
        """
        # Attempt tiktoken BPE
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            ids = enc.encode(word)
            subwords = [enc.decode([tid]) for tid in ids]
            clean = [s for s in subwords if s.strip()]
            if clean:
                return clean
        except Exception:
            pass

        # Character bigram fallback: always produces at least one token
        if len(word) <= 2:
            return [word]
        return [word[i : i + 2] for i in range(len(word) - 1)]

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_wn'] = None
        state['ontology'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        from .ontology_manager import OntologyManager
        self.ontology = OntologyManager("ontology.json")
        if self.use_wordnet:
            try:
                from nltk.corpus import wordnet
                self._wn = wordnet
            except ImportError:
                self.use_wordnet = False

    def _cluster_image(self, path: str) -> str:
        try:
            import torch
            from transformers import CLIPModel
            return f"[IMG_EMBEDDING_HASH]"
        except ImportError:
            return f"[IMG_CONCEPT_{hash(path) % 1000}]"

    def _cluster_audio(self, path: str) -> str:
        try:
            import librosa
            return f"[AUDIO_EMBEDDING_HASH]"
        except ImportError:
            return f"[AUDIO_CONCEPT_{hash(path) % 1000}]"

    def _hash_ast(self, code_str: str) -> list[str]:
        try:
            tree = ast.parse(code_str)
            concepts = []
            for node in ast.walk(tree):
                node_type = type(node).__name__
                if isinstance(node, ast.Name):
                    concepts.append(f"[AST_VAR]")
                elif isinstance(node, ast.FunctionDef):
                    concepts.append(f"[AST_FUNC_DEF]")
                elif isinstance(node, ast.For):
                    concepts.append(f"[AST_FOR_LOOP]")
                elif isinstance(node, ast.If):
                    concepts.append(f"[AST_IF_COND]")
                elif isinstance(node, ast.Assign):
                    concepts.append(f"[AST_ASSIGN]")
                else:
                    concepts.append(f"[AST_NODE_{node_type}]")
            return concepts
        except SyntaxError:
            return []

    def tokenize(self, data, is_inference: bool = False, _in_code_block: bool = False) -> list:
        if isinstance(data, list):
            out = []
            in_code = _in_code_block
            for item in data:
                if isinstance(item, str) and item.strip() == "```python":
                    in_code = True
                    out.append("```python")
                    continue
                elif isinstance(item, str) and item.strip() == "```" and in_code:
                    in_code = False
                    out.append("```")
                    continue
                out.extend(self.tokenize(item, is_inference, _in_code_block=in_code))
            return out

        if isinstance(data, str) and data.lower().endswith((".png", ".jpg", ".jpeg")):
            return [self._cluster_image(data)]
            
        if isinstance(data, str) and data.lower().endswith((".wav", ".mp3", ".flac")):
            return [self._cluster_audio(data)]
            
        if isinstance(data, OntologicalState):
            return [f"[STATE_{data.name}]"]
        if isinstance(data, OntologicalAction):
            if hasattr(data, 'target') and getattr(data, 'target') is not None:
                return [f"[ACTION_{data.name}_{data.target}]"]
            return [f"[ACTION_{data.name}]"]
            
        if isinstance(data, str):
            # Special tokens pass through immediately
            if data.startswith("<|") or data.startswith("[SYS_"):
                self._known_concepts.add(data)
                return [data]
                
            if data.replace('.', '', 1).isdigit():
                self._known_concepts.add(data)
                return [data]

            if data in self._cache and not _in_code_block:
                return [self._cache[data]]
                
            # ── CODE MODE: strict literal pass-through ──
            if _in_code_block:
                self._known_concepts.add(data)
                return [data]
            
            # ── CONVERSATION MODE: semantic mapping active ──
            concept = data.lower().strip(".,!?")
            
            # Still bypass WordNet for Python keywords even in conversation
            _CODE_KEYWORDS = {
                "def", "class", "return", "import", "from", "self",
                "while", "for", "if", "else", "elif", "try", "except",
                "finally", "with", "as", "yield", "lambda", "pass",
                "break", "continue", "and", "or", "not", "in", "is",
                "none", "true", "false", "print", "len", "range",
                "async", "await", "raise", "assert", "del", "global",
                "nonlocal", "```python", "```",
                "a", "b", "c", "d", "e", "f", "x", "y", "z", "n", "s",
                "add", "subtract", "multiply", "divide", "max", "min", "sum",
                "function", "integer", "string", "boolean", "returns", "takes", "called",
                "reverse", "factorial", "even", "odd"
            }
            if concept in _CODE_KEYWORDS or any(c in concept for c in "()[]{}:="):
                self._known_concepts.add(concept)
                return [concept]
                
            canonical = self.ontology.get(concept)
            if canonical:
                self._cache[data] = canonical
                self._known_concepts.add(canonical)
                return [canonical]
                
            mapped = False
            
            # Common stop words should bypass WordNet to prevent "i" -> "iodine"
            _STOP_WORDS = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
                           "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", 
                           "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
                           "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
                           "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
                           "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", 
                           "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", 
                           "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", 
                           "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", 
                           "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", 
                           "can", "will", "just", "don", "should", "now"}
                           
            if self.use_wordnet and self._wn and concept not in _STOP_WORDS:
                synsets = self._wn.synsets(concept)
                if synsets:
                    concept = synsets[0].name()
                    mapped = True
                    
            if not mapped:
                if self._known_concepts:
                    closest = difflib.get_close_matches(concept, self._known_concepts, n=1, cutoff=0.9)
                    if closest:
                        canonical = closest[0]
                        self.ontology.add_mapping(concept, canonical)
                        concept = canonical
                        mapped = True
                        
            if not mapped and is_inference and concept not in self._known_concepts:
                # BPE fallback: shatter into subwords so TokenEmbedder always
                # receives valid, non-zero vectors (no more OOV annihilation).
                subwords = self._bpe_fallback(concept)
                for sw in subwords:
                    self._known_concepts.add(sw)
                try:
                    import uchi.telemetry as _tel
                    _tel.append("tokenizer", "bpe_splits",
                                {"word": concept, "subwords": subwords})
                    _tel.increment("tokenizer", "bpe_fallback_count")
                except Exception:
                    pass
                return subwords

            self._cache[data] = concept
            self._known_concepts.add(concept)
            try:
                import uchi.telemetry as _tel
                _tel.increment("tokenizer", "exact_dictionary_count")
            except Exception:
                pass
            return [concept]
            
        return [str(data)]

    def detokenize(self, concepts: list) -> list[str]:
        out = []
        for c in concepts:
            if c and isinstance(c, str):
                parts = c.split('.')
                if len(parts) == 3 and parts[1] in ['n', 'v', 'a', 'r', 's'] and parts[2].isdigit():
                    out.append(parts[0].replace('_', ' '))
                else:
                    out.append(c)
            elif isinstance(c, UnknownConcept):
                out.append(c.raw_word)
            else:
                out.append(str(c))
        return out
