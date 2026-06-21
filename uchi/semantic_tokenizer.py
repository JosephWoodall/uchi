"""
Semantic Tokenizer
==================
Replaces raw exact string matching with a lightweight semantic concept mapping.
Uses NLTK WordNet (if available) to hash similar words to the same synset ID.
"""

class SemanticTokenizer:
    def __init__(self, use_wordnet: bool = True):
        self.use_wordnet = use_wordnet
        self._cache = {}
        self._wn = None
        
        if self.use_wordnet:
            try:
                import nltk
                from nltk.corpus import wordnet
                self._wn = wordnet
                
                # Check if data is downloaded
                try:
                    self._wn.synsets('dog')
                except LookupError:
                    nltk.download('wordnet', quiet=True)
            except ImportError:
                self.use_wordnet = False
                
    def tokenize(self, token: str) -> str:
        """
        Maps a string token to its core concept ID if possible.
        If it's a known concept, returns the synset name. 
        Otherwise returns the lowered token.
        """
        if not isinstance(token, str):
            return token
            
        if token in self._cache:
            return self._cache[token]
            
        concept = token.lower()
        if self.use_wordnet and self._wn:
            synsets = self._wn.synsets(concept)
            if synsets:
                # Use the most common synset as the abstract concept ID
                concept = synsets[0].name()
                
        self._cache[token] = concept
        return concept
