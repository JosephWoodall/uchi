"""
Omni Tokenizer (Multi-Modal Cortex)
===================================
Abstracts multiple data modalities (Text, Math, Code, Images, Audio) 
into a universal mathematical Concept ID stream.
"""

from .process import OntologicalState, OntologicalAction

class OmniTokenizer:
    def __init__(self, use_wordnet: bool = True):
        self.use_wordnet = use_wordnet
        self._cache = {}
        self._wn = None
        
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
        """
        Abstract hook for Image feature extraction.
        In production, passes the image array through a lightweight CNN (e.g. ResNet)
        or CLIP model to extract visual geometry clusters.
        """
        # Mock geometric hashing for the v0.2.0 framework
        return f"[IMG_CONCEPT_{hash(path) % 1000}]"

    def _cluster_audio(self, path: str) -> str:
        """
        Abstract hook for Audio feature extraction.
        In production, extracts MFCC frames and clusters acoustic features.
        """
        # Mock acoustic hashing
        return f"[AUDIO_CONCEPT_{hash(path) % 1000}]"

    def tokenize(self, data) -> str:
        """
        Universal router that casts any modality into a geometric concept ID.
        """
        # 1. Image Modality
        if isinstance(data, str) and data.lower().endswith((".png", ".jpg", ".jpeg")):
            return self._cluster_image(data)
            
        # 2. Audio Modality
        if isinstance(data, str) and data.lower().endswith((".wav", ".mp3", ".flac")):
            return self._cluster_audio(data)
            
        # 3. Python Object Modality (Ontological Agents)
        if isinstance(data, OntologicalState):
            return f"[STATE_{data.name}]"
        if isinstance(data, OntologicalAction):
            if hasattr(data, 'target') and getattr(data, 'target') is not None:
                return f"[ACTION_{data.name}_{data.target}]"
            return f"[ACTION_{data.name}]"
            
        # 4. Text and Math Modality
        if isinstance(data, str):
            if data in self._cache:
                return self._cache[data]
                
            concept = data.lower().strip(",.")
            if self.use_wordnet and self._wn:
                synsets = self._wn.synsets(concept)
                if synsets:
                    concept = synsets[0].name()
                    
            self._cache[data] = concept
            return concept
            
        return str(data)
