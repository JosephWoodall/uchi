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
from .grpo import AgenticBaseline
from .procedural_memory import ProceduralMemory
import torch

class OmniRouter:
    """
    The Prefrontal Cortex of Uchi v0.2.0.
    Ingests mixed-modality streams (Text, Audio, Images, Math, Code), 
    compresses them infinitely, stores them in geometric buffers, 
    and predicts parallel futures.
    """
    def __init__(self, use_bpe: bool = False, memory_window: int = 5, context_length: int = 10, progress_callback=None):
        self.use_bpe = use_bpe
        # Phase 5 (Frontend): Omni-Modal Abstraction
        self.tokenizer = OmniTokenizer(use_wordnet=True)
        # Phase 4: Infinite Context Compression
        self.bpe = OnlineTokenizer(max_merges=50000) if use_bpe else None
        # Phase 5 (Backend): Zero-Shot Associative Memory
        self.memory = AssociativeMemory(window_size=memory_window)
        # Phase 3: Plural Sequence Generation
        self.predictor = SequenceGenerator(
            context_length=context_length,
            min_context_length=0
        ) 
        self.baseline = AgenticBaseline()
        self.procedural = ProceduralMemory()

        # SSM optimizer persisted on self so Adam's momentum accumulates across turns
        from uchi.neuro_symbolic import get_ssm
        _ssm = get_ssm()
        self.ssm_optimizer = torch.optim.Adam(_ssm.parameters(), lr=1e-3)
        self._knowledge_bootstrapped = False

        self._bootstrap_persona(progress_callback)
        self._bootstrap_knowledge(progress_callback)

    def __setstate__(self, state):
        """Forward-compatibility: initialize attributes added in newer versions after unpickling."""
        self.__dict__.update(state)
        if not hasattr(self, 'procedural'):
            self.procedural = ProceduralMemory()
        if not hasattr(self, 'baseline'):
            self.baseline = AgenticBaseline()
        if not hasattr(self, 'ssm_optimizer'):
            from uchi.neuro_symbolic import get_ssm
            self.ssm_optimizer = torch.optim.Adam(get_ssm().parameters(), lr=1e-3)
        if not hasattr(self, '_knowledge_bootstrapped'):
            self._knowledge_bootstrapped = True  # don't re-bootstrap existing brains

    def _bootstrap_persona(self, progress_callback=None):
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
                content = f.read()
                # Split by <|user|> but keep the token
                blocks = content.split("<|user|>")
                for block in blocks:
                    block = block.strip()
                    if not block or block.startswith("#"):
                        continue
                    if "<|assistant|>" in block:
                        turns.append("<|user|> " + block)
        except FileNotFoundError:
            # Fallback minimal persona if file is missing
            turns = [
                "<|user|> hello <|assistant|> hello there how can i help you today",
                "<|user|> who are you <|assistant|> i am uchi a deterministic sequence predictor",
            ]
        
        # Simulate Massive Positive Reinforcement RL during cold start
        total_steps = 5 * len(turns)
        current_step = 0
        from uchi.neuro_symbolic import get_ssm
        ssm = get_ssm()
        ssm.train()
        for _ in range(5):
            for turn in turns:
                tokens = turn.split()
                self.stream(tokens)

                # Jumpstart the value head so it doesn't randomly output < 0.0
                self.ssm_optimizer.zero_grad()
                v_loss = ssm.update_value(tokens, reward=1.0)
                d_loss = ssm.train_dynamics(tokens)
                (v_loss + d_loss).backward()
                self.ssm_optimizer.step()
                
                current_step += 1
                if progress_callback:
                    progress_callback(current_step, total_steps)

    def _bootstrap_knowledge(self, progress_callback=None):
        """
        Runs once on cold start (when brain.uchi doesn't exist).
        Streams stdlib function patterns and Wikipedia fact triples into the trie.
        Guarded by self._knowledge_bootstrapped so pickle restores skip it.
        """
        if getattr(self, '_knowledge_bootstrapped', False):
            return

        import os, sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

        try:
            from bootstrap_code import run as run_code
            run_code(self, progress_callback=progress_callback)
        except Exception:
            pass

        try:
            from bootstrap_wikidata import run as run_wiki
            run_wiki(self, progress_callback=progress_callback)
        except Exception:
            pass

        # Persist SSM weights so they survive across restarts
        try:
            import torch
            from uchi.neuro_symbolic import get_ssm
            torch.save(get_ssm().state_dict(), "ssm_dynamics.pt")
        except Exception:
            pass

        self._knowledge_bootstrapped = True

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
        
        ans_concept, cosine_score = self.memory.query(concept_query)

        # Cosine similarity is already in [-1, 1] — no length normalization needed.
        # 0.5 = confident semantic match; below that the SSM states are dissimilar.
        if ans_concept and cosine_score is not None and cosine_score >= 0.5:
            return str(ans_concept)

        # Known token sequence in trie but no strong memory match — conversational turn,
        # skip web sourcing.
        if cosine_score is not None and cosine_score > 0:
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
        
    def predict_future(self, prompt: list, steps: int = 1, temperature: float = 0.0, creativity: float = 0.0, bias_context: str = None) -> list:
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
            
        predicted = self.predictor.generate(n_tokens=steps, seed=concepts, temperature=temperature, use_mcts=True, bias_context=bias_context)
        if self.bpe:
            predicted = self.bpe.detokenize(predicted)
        
        return predicted

    def chat(self, message: str, callback=None) -> str:
        """
        High-level Conversational API that integrates RL, Sentiment, and Pruning.
        """
        from uchi.omni_tokenizer import UnknownConcept
        
        # 1. Tamed Active Learning & Intent Extraction
        query_tokens = message.split()
        
        # Root 1 Fix: Procedural intent routing
        procedure = self.procedural.retrieve(message)
        if procedure:
            query_tokens = procedure.split() + ["|"] + query_tokens
            
        concepts = self.tokenizer.tokenize(query_tokens, is_inference=True)
        unknowns = [c.raw_word for c in concepts if isinstance(c, UnknownConcept)]
        
        # Only interrupt if the user explicitly asks to learn, or if we want strict active learning.
        # For a seamless experience, we just let UnknownConcepts pass through as their raw strings
        # unless they ask "define X" or something similar. For now, we'll gracefully bypass.
        
        # 2. Continuous Sentiment & Pruning
        positive_words = {"good", "great", "awesome", "correct", "yes", "amazing", "thanks", "thank"}
        negative_words = {"bad", "wrong", "incorrect", "no", "stop", "terrible", "awful"}
        
        msg_lower = message.lower()
        score_val = 0
        for w in query_tokens:
            w_clean = w.lower().strip(".,!?")
            if w_clean in positive_words: score_val += 1
            if w_clean in negative_words: score_val -= 1
            
        if hasattr(self, 'last_sequence') and self.last_sequence:
            reward = 0.0
            if score_val > 0:
                # Positive Reinforcement
                if callback: callback("reinforce", "Positive Momentum: Reinforcing previous sequence!")
                reward = 1.0
            elif score_val < 0:
                # Synaptic Pruning
                if callback: callback("prune", "Synaptic Pruning: Eradicating previous hallucination!")
                self.predictor.unlearn(self.last_sequence)
                reward = -1.0
                
            # Root 2 Fix: GRPO SSM Value Head Training
            advantage = self.baseline.advantage(reward)
            self.baseline.update(reward)

            from uchi.neuro_symbolic import get_ssm
            ssm = get_ssm()
            ssm.train()
            self.ssm_optimizer.zero_grad()
            v_loss = ssm.update_value(self.last_sequence, reward=advantage)
            d_loss = ssm.train_dynamics(self.last_sequence)
            (v_loss + d_loss).backward()
            self.ssm_optimizer.step()

        # 3. Query Memory
        retrieved_context_str = self.query(concepts)
        
        tokens = ["<|user|>"] + concepts + ["<|assistant|>"]
        # Root 3 Fix: Stream moved to the end of interaction loop
        
        # 4. Predict
        bias = retrieved_context_str if retrieved_context_str != "[Unknown Context]" else None
        prompt_entropy = self.predictor.score(tokens)
        pred = self.predict_future(tokens, steps=60, temperature=0.0, creativity=0.0, bias_context=bias)
        
        # 5. Hallucination Check (Seamless UX)
        # Gate strategy: use entropy until GRPO has enough samples (baseline.mean != 0),
        # then trust the trained value head. This prevents cold-start false rejections.
        if pred:
            ssm_trained = abs(self.baseline.mean) > 0.01 or self.baseline.var < 0.95
            if ssm_trained:
                from uchi.neuro_symbolic import get_ssm
                ssm = get_ssm()
                state_vec = ssm.get_state(tokens)
                value_conf = ssm.value(state_vec).item()
                should_reject = value_conf < -0.5
            else:
                value_conf = 0.0
                should_reject = prompt_entropy > 12.0

            if should_reject:
                if callback: callback("hallucination", f"Unknown context (entropy={prompt_entropy:.1f}, value={value_conf:.2f})")
                fallback = "I do not know the answer to that, even after searching the web."
                self.last_sequence = ["<|user|>"] + query_tokens + ["<|assistant|>"] + fallback.split()
                return fallback

        # Store walk data for debugging
        self.last_walk_data = {
            "prediction_depth": self.predictor._pred._last_prediction_depth,
            "contributions": self.predictor._pred._last_contributions,
            "max_sim": self.predictor._pred._last_max_sim
        }
        
        # 6. Topologically walk the graph
        print(f"DEBUG PRED: {pred}")
        reply = []
        recording = True 
            
        for p in pred:
            if recording and p in ("<|user|>", "<|assistant|>", "<|end|>"):
                break
            if recording:
                reply.append(p)
                
        if reply:
            self.last_sequence = ["<|user|>"] + query_tokens + ["<|assistant|>"] + reply
        else:
            self.last_sequence = None
            
        # Translate WordNet concepts back to readable strings
        predicted = self.tokenizer.detokenize(reply)
        reply_str = ' '.join([str(p) for p in predicted])
        
        if not reply_str.strip():
            if callback: callback("hallucination", "Empty prediction path.")
            fallback = "I do not know the answer to that, even after searching the web."
            self.last_sequence = None
            return fallback
        
        # RL Autonomous Code Sandbox
        if "```python" in reply_str and "```" in reply_str.split("```python")[1]:
            code_block = reply_str.split("```python")[1].split("```")[0].strip()
            # Extremely simple sandbox (do not use in production without real sandboxing)
            import subprocess
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code_block)
                temp_path = f.name
                
            try:
                # Compile to check syntax
                result = subprocess.run(["python", "-m", "py_compile", temp_path], capture_output=True, text=True)
                code_reward = 0.0
                if result.returncode != 0:
                    if callback: callback("prune", "SyntaxError in predicted code! Pruning the hallucination...")
                    if self.last_sequence:
                        self.predictor.unlearn(self.last_sequence)
                    reply_str += "\n\n[RL ENGINE: I detected a SyntaxError in my prediction and autonomously pruned it.]"
                    code_reward = -1.0
                else:
                    if callback: callback("reinforce", "Code compiled successfully! Double-reinforcing syntax path.")
                    code_reward = 1.0
                    
                if self.last_sequence:
                    advantage = self.baseline.advantage(code_reward)
                    self.baseline.update(code_reward)
                    from uchi.neuro_symbolic import get_ssm
                    ssm = get_ssm()
                    ssm.train()
                    self.ssm_optimizer.zero_grad()
                    v_loss = ssm.update_value(self.last_sequence, reward=advantage)
                    d_loss = ssm.train_dynamics(self.last_sequence)
                    (v_loss + d_loss).backward()
                    self.ssm_optimizer.step()
                    
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        # Root 3 Fix: Stream the full interaction sequence ONLY after generation is complete
        if self.last_sequence:
            self.stream(self.last_sequence)
            
        return reply_str

