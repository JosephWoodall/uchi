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
    """Wrapper for out-of-vocabulary words that need user clarification."""
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

    def tokenize(self, data, is_inference: bool = False) -> list:
        if isinstance(data, list):
            out = []
            for item in data:
                out.extend(self.tokenize(item, is_inference))
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
            
        if isinstance(data, str) and ("def " in data or "import " in data):
            ast_concepts = self._hash_ast(data)
            if ast_concepts:
                return ast_concepts
            
        if isinstance(data, str):
            if data.startswith("<|") or data.startswith("[SYS_"):
                self._known_concepts.add(data)
                return [data]
                
            if data.replace('.', '', 1).isdigit():
                self._known_concepts.add(data)
                return [data]

            if data in self._cache:
                return [self._cache[data]]
                
            concept = data.lower().strip(",.")
            
            canonical = self.ontology.get(concept)
            if canonical:
                self._cache[data] = canonical
                self._known_concepts.add(canonical)
                return [canonical]
                
            mapped = False
            if self.use_wordnet and self._wn:
                synsets = self._wn.synsets(concept)
                if synsets:
                    concept = synsets[0].name()
                    mapped = True
                    
            if not mapped:
                if self._known_concepts:
                    closest = difflib.get_close_matches(concept, self._known_concepts, n=1, cutoff=0.8)
                    if closest:
                        canonical = closest[0]
                        self.ontology.add_mapping(concept, canonical)
                        concept = canonical
                        mapped = True
                        
            if not mapped and is_inference and concept not in self._known_concepts:
                return [UnknownConcept(concept)]
            
            self._cache[data] = concept
            self._known_concepts.add(concept)
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
