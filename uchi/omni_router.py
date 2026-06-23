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
    def __init__(self, use_bpe: bool = False, memory_window: int = 5, context_length: int = 10):
        self.use_bpe = use_bpe
        # Phase 5 (Frontend): Omni-Modal Abstraction
        self.tokenizer = OmniTokenizer(use_wordnet=False)
        # Phase 4: Infinite Context Compression
        self.bpe = OnlineTokenizer(max_merges=50000) if use_bpe else None
        # Phase 5 (Backend): Zero-Shot Associative Memory
        self.memory = AssociativeMemory(window_size=memory_window)
        # Phase 3: Plural Sequence Generation
        self.predictor = SequenceGenerator(context_length=context_length) 
        
        self._bootstrap_persona()
        
    def _bootstrap_persona(self):
        """
        Natively instruction-tunes the geometric trie so ODUSP understands conversational interactions 
        without external RAG or hardcoded heuristics.
        
        Loads conversation turns from persona.txt and trains each turn as an
        isolated sequence so the trie learns tight <|user|> X <|assistant|> Y associations.
        """
        import os
        persona_path = os.path.join(os.path.dirname(__file__), "persona.txt")
        
        turns = []
        try:
            with open(persona_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "<|user|>" in line and "<|assistant|>" in line:
                        turns.append(line)
        except FileNotFoundError:
            # Fallback minimal persona if file is missing
            turns = [
                "<|user|> hello <|assistant|> hello there how can i help you today",
                "<|user|> who are you <|assistant|> i am uchi a deterministic sequence predictor",
            ]
        
        # Simulate Massive Positive Reinforcement RL during cold start
        for _ in range(25):
            for turn in turns:
                tokens = turn.split()
                self.stream(tokens)
        
    def stream(self, concepts: list):
        """
        Ingest a raw stream of multi-modal objects.
        """
        concepts = self.tokenizer.tokenize(concepts)
        
        # 2. Compress the stream via BPE
        if self.bpe:
            self.bpe.update(concepts, predictor_accuracy=1.0)
            concepts = list(self.bpe.tokenize(concepts))
            
        # 3. Store in the Zero-Shot Associative Memory buffer
        self.memory.stream_context(concepts)
        
        # 4. Train the deterministic trie
        self.predictor.partial_fit(concepts)
        
    def retrieve_context(self, sequence: list) -> str:
        """
        Retrieves the most likely long-term memory association.
        """
        try:
            concepts = self.tokenizer.tokenize(sequence, is_inference=True)
            if self.bpe:
                concepts = list(self.bpe.tokenize(concepts))
            
            ans_concept, raw_score = self.memory.query(concepts)
            
            # Normalize score by query length to prevent long conversational 
            # queries from artificially inflating the score via cumulative common words.
            normalized_score = raw_score / max(1, len(concepts)) if raw_score is not None else 0.0
            
            if ans_concept and normalized_score >= 1.5:
                return str(ans_concept)
        except Exception:
            pass
        return "[Unknown Context]"
        
    def query(self, prompt: list) -> str:
        """
        Zero-Shot Question Answering against the memory buffer.
        """
        query_sequence = prompt
        concept_query = self.tokenizer.tokenize(query_sequence)
        if self.bpe:
            concept_query = list(self.bpe.tokenize(concept_query))
        
        ans_concept, raw_score = self.memory.query(concept_query)
        
        # Normalize score by query length to prevent long conversational 
        # queries from artificially inflating the score via cumulative common words.
        normalized_score = raw_score / max(1, len(concept_query)) if raw_score is not None else 0.0
        
        if ans_concept and normalized_score >= 1.5:
            return str(ans_concept)
            
        # If it's a known sequence in the trie (raw_score > 0) but didn't meet the 
        # RAG threshold, it's just a conversational turn. Skip web sourcing.
        if raw_score is not None and raw_score > 0:
            return "[Unknown Context]"
        
        # Autonomous Web Sourcing Hook
        try:
            from uchi.plugins.web import fetch_web_context
            # If memory failed, try searching the web
            raw_query = " ".join(query_sequence)
            web_context = fetch_web_context(raw_query)
            if web_context:
                # Stream the new structural truth
                self.stream(web_context.split())
                # Re-query the memory
                ans_concept2, ans_score2 = self.memory.query(concept_query)
                if ans_concept2 and ans_score2 >= 1.5:
                    return str(ans_concept2)
        except Exception:
            pass
            
        return "[Unknown Context]"
        
    def predict_future(self, prompt: list, steps: int = 1, temperature: float = 0.0, creativity: float = 0.0) -> list:
        """
        Simulates the causal future of the sequence.
        If creativity > 0.0, Stochastic Context Mutation is applied.
        """
        import random
        concepts = self.tokenizer.tokenize(prompt, is_inference=True)
        if self.bpe:
            concepts = list(self.bpe.tokenize(concepts))
            
        # Stochastic Context Mutation
        if creativity > 0.0 and concepts:
            mutated_concepts = []
            vocab = list(self.memory.G.nodes) if hasattr(self.memory, 'G') and len(self.memory.G.nodes) > 0 else []
            for c in concepts:
                # creativity maps 0.0 to 1.0. At 1.0, 50% chance to drop or swap.
                roll = random.random()
                if roll < (creativity * 0.25):
                    continue # drop token
                elif roll < (creativity * 0.5) and vocab:
                    mutated_concepts.append(random.choice(vocab)) # swap token
                else:
                    mutated_concepts.append(c)
            concepts = mutated_concepts
            
        predicted = self.predictor.generate(n_tokens=steps, seed=concepts, temperature=temperature)
        if self.bpe:
            predicted = self.bpe.detokenize(predicted)
        
        # Translate WordNet concepts back to readable strings
        predicted = self.tokenizer.detokenize(predicted)
        return [str(p) for p in predicted]
