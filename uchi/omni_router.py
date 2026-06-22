"""
Omni Router (The Deterministic LLM Master Controller)
=====================================================
Unifies Phase 1 (Math), Phase 2 (Process), Phase 3 (Plural Simulation), 
Phase 4 (Infinite Context), and Phase 5 (Associative Memory) into a 
single, coherent Omni-Modal architecture.
"""

from typing import List, Any
from .omni_tokenizer import OmniTokenizer
from .online_tokenizer import OnlineTokenizer
from .memory import AssociativeMemory
from .generative import SequenceGenerator

class OmniRouter:
    """
    The Prefrontal Cortex of Uchi v0.2.0.
    Ingests mixed-modality streams (Text, Audio, Images, Math, Code), 
    compresses them infinitely, stores them in geometric buffers, 
    and predicts parallel futures.
    """
    def __init__(self, use_bpe: bool = True, memory_window: int = 5, context_length: int = 10):
        self.use_bpe = use_bpe
        # Phase 5 (Frontend): Omni-Modal Abstraction
        self.tokenizer = OmniTokenizer(use_wordnet=True)
        # Phase 4: Infinite Context Compression
        self.bpe = OnlineTokenizer(vocab_limit=50000) if use_bpe else None
        # Phase 5 (Backend): Zero-Shot Associative Memory
        self.memory = AssociativeMemory(window_size=memory_window)
        # Phase 3: Plural Sequence Generation
        self.predictor = SequenceGenerator(context_length=context_length) 
        
    def stream(self, data_stream: List[Any]):
        """
        Ingest a mixed-modality stream, abstract it, compress it, 
        store it geometrically, and train the deterministic predictor.
        """
        # 1. Abstract modalities into [CONCEPT_IDs]
        concepts = [self.tokenizer.tokenize(d) for d in data_stream]
        
        # 2. Compress the stream via BPE
        if self.bpe:
            concepts = list(self.bpe.encode(concepts))
            
        # 3. Store in the Zero-Shot Associative Memory buffer
        self.memory.stream_context(concepts)
        
        # 4. Train the deterministic trie
        self.predictor.fit(concepts)
        
    def query(self, prompt: List[Any]) -> str:
        """
        Zero-Shot Omni-Retrieval. Cross-modally retrieve an answer 
        using geometric non-parametric attention.
        """
        concepts = [self.tokenizer.tokenize(p) for p in prompt]
        if self.bpe:
            concepts = list(self.bpe.encode(concepts))
            
        return self.memory.query(concepts)
        
    def predict_future(self, prompt: List[Any], steps: int = 1, temperature: float = 1.0) -> List[str]:
        """
        Simulate the parallel future of the multimodal stream.
        """
        concepts = [self.tokenizer.tokenize(p) for p in prompt]
        if self.bpe:
            concepts = list(self.bpe.encode(concepts))
            
        predicted = self.predictor.generate(concepts, steps=steps, temperature=temperature)
        return predicted
