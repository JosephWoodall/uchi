"""
Omni Tokenizer (Multi-Modal Cortex)
===================================
Abstracts multiple data modalities (Text, Math, Code, Images, Audio) 
into a universal mathematical Concept ID stream.
"""
import ast
import difflib
from .process import OntologicalState, OntologicalAction

class OmniTokenizer:
    def __init__(self, use_wordnet: bool = True):
        self.use_wordnet = use_wordnet
        self._cache = {}
        self._wn = None
        self._known_concepts = set()
        
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

    def _cluster_image(self, path: str) -> str:
        """Plugin Architecture: Hooks into lightweight Vision encoders if available."""
        try:
            import torch
            from transformers import CLIPModel
            # In a real plugin, this hashes the ResNet/CLIP embedding vector
            return f"[IMG_EMBEDDING_HASH]"
        except ImportError:
            return f"[IMG_CONCEPT_{hash(path) % 1000}]"

    def _cluster_audio(self, path: str) -> str:
        """Plugin Architecture: Hooks into lightweight Audio encoders if available."""
        try:
            import librosa
            # In a real plugin, this clusters MFCC frames
            return f"[AUDIO_EMBEDDING_HASH]"
        except ImportError:
            return f"[AUDIO_CONCEPT_{hash(path) % 1000}]"

    def _hash_ast(self, code_str: str) -> list[str]:
        """
        Native Coding Superpowers: Parses Python code into an Abstract Syntax Tree (AST) 
        and extracts geometric structural patterns rather than English text.
        """
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
            return [] # Fallback to standard text if not valid Python

    def tokenize(self, data) -> list[str]:
        """
        Universal router that casts any modality into a sequence of geometric concept IDs.
        Always returns a list of concepts.
        """
        if isinstance(data, list):
            # If a list is passed, recursively tokenize
            out = []
            for item in data:
                out.extend(self.tokenize(item))
            return out

        # 1. Image Modality
        if isinstance(data, str) and data.lower().endswith((".png", ".jpg", ".jpeg")):
            return [self._cluster_image(data)]
            
        # 2. Audio Modality
        if isinstance(data, str) and data.lower().endswith((".wav", ".mp3", ".flac")):
            return [self._cluster_audio(data)]
            
        # 3. Python Object Modality (Ontological Agents)
        if isinstance(data, OntologicalState):
            return [f"[STATE_{data.name}]"]
        if isinstance(data, OntologicalAction):
            if hasattr(data, 'target') and getattr(data, 'target') is not None:
                return [f"[ACTION_{data.name}_{data.target}]"]
            return [f"[ACTION_{data.name}]"]
            
        # 4. Code Modality (AST Hashing)
        if isinstance(data, str) and ("def " in data or "import " in data):
            ast_concepts = self._hash_ast(data)
            if ast_concepts:
                return ast_concepts
            
        # 5. Text and Math Modality
        if isinstance(data, str):
            if data in self._cache:
                return [self._cache[data]]
                
            concept = data.lower().strip(",.")
            
            # WordNet Semantic Mapping
            if self.use_wordnet and self._wn:
                synsets = self._wn.synsets(concept)
                if synsets:
                    concept = synsets[0].name()
                else:
                    # Levenshtein Subword Fallback for OOV slang/typos
                    if self._known_concepts:
                        closest = difflib.get_close_matches(concept, self._known_concepts, n=1, cutoff=0.8)
                        if closest:
                            concept = closest[0]
            
            self._known_concepts.add(concept)
            self._cache[data] = concept
            return [concept]
            
        return [str(data)]

