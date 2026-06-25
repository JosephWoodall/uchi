"""
Omni Router (The Deterministic LLM Master Controller)
=====================================================
Unifies Phase 1 (Math), Phase 2 (Process), Phase 3 (Plural Simulation), 
Phase 4 (Infinite Context), and Phase 5 (Associative Memory) into a 
single, coherent Omni-Modal architecture.
"""

import threading
from typing import List, Any
from .omni_tokenizer import OmniTokenizer
from .online_tokenizer import OnlineTokenizer
from .memory import AssociativeMemory
from .generative import SequenceGenerator
from .grpo import AgenticBaseline
from .procedural_memory import ProceduralMemory
from .code_engine import CodeEngine
from .specialist_pool import SpecialistPool
from .skill_registry import SkillRegistry
from .convergent_engine import ConvergentEngine
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
        self.ssm_lock = threading.Lock()
        self._knowledge_bootstrapped = False

        # Phase 1+3: Parallel MCTS code generation with REPL oracle
        self.code_engine = CodeEngine(self.predictor, n_workers=4)
        # Phase 2: MoE SpecialistPool (lazy-loads specialist brains on first use)
        self.specialist_pool = SpecialistPool(self)
        # Skill toolkit: markdown-defined /commands
        self.skills = SkillRegistry(self)
        # Convergent MCTS engine for deliberative response generation
        self.convergent = ConvergentEngine(self)

        self._conversation_history: list = []  # [(q_tokens, r_tokens), ...]
        self._context_window: int = 3           # prior turns to prepend to seed

        self._bootstrap_persona(progress_callback)
        self._bootstrap_knowledge(progress_callback)

    def __setstate__(self, state):
        """Forward-compatibility: initialize attributes added in newer versions after unpickling."""
        self.__dict__.update(state)
        if not hasattr(self, 'procedural'):
            self.procedural = ProceduralMemory()
        if not hasattr(self, 'baseline'):
            self.baseline = AgenticBaseline()
        # Always re-bind optimizer to the current SSM instance (architecture may have changed)
        from uchi.neuro_symbolic import get_ssm
        self.ssm_optimizer = torch.optim.Adam(get_ssm().parameters(), lr=1e-3)
        if not hasattr(self, '_knowledge_bootstrapped'):
            self._knowledge_bootstrapped = True
        if not hasattr(self, 'code_engine'):
            self.code_engine = CodeEngine(self.predictor, n_workers=4)
        if not hasattr(self, 'specialist_pool'):
            self.specialist_pool = SpecialistPool(self)
        if not hasattr(self, 'skills'):
            self.skills = SkillRegistry(self)
        if not hasattr(self, 'convergent'):
            self.convergent = ConvergentEngine(self)
        if not hasattr(self, 'ssm_lock'):
            self.ssm_lock = threading.Lock()
        if not hasattr(self, '_background_started'):
            self._background_started = False
        if not hasattr(self, '_daemon_procs'):
            self._daemon_procs = []
        if not hasattr(self, '_conversation_history'):
            self._conversation_history = []
        if not hasattr(self, '_context_window'):
            self._context_window = 3

    def __getstate__(self):
        # Exclude non-serialisable or always-recreated transient attributes.
        # __setstate__ rebuilds ssm_lock, ssm_optimizer, and _background_started
        # on every load, so persisting them is both wasteful and broken
        # (threading.Lock cannot be pickled).
        skip = {"ssm_lock", "ssm_optimizer", "_background_started", "_daemon_procs"}
        return {k: v for k, v in self.__dict__.items() if k not in skip}

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

                # Jumpstart the value head so it doesn't randomly output < 0.0.
                # Single compute_loss call avoids running two GRU forward passes
                # before .backward() — double-pass invalidates gradient graphs.
                self.ssm_optimizer.zero_grad()
                loss = ssm.compute_loss(tokens, reward=1.0)
                loss.backward()
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

        # Code knowledge: HumanEval + MBPP → trie via ```python``` code-block tokens
        # Gives the trie real Python token sequences so pass@1 lifts from 0%.
        try:
            from bootstrap_humaneval import run as run_humaneval
            run_humaneval(self, progress_callback=progress_callback)
        except Exception:
            pass

        # Conversational knowledge: SQuAD Q&A → AssociativeMemory + trie.
        # Capped at 2000 pairs on cold start to keep boot time reasonable;
        # run scripts/bootstrap_convo.py manually for the full 87k corpus.
        try:
            from bootstrap_convo import _load_squad, run as run_convo
            squad_pairs = _load_squad(limit=2000)
            run_convo(self, pairs=squad_pairs or None, progress_callback=progress_callback)
        except Exception:
            pass

        # Bootstrap specialist brains automatically on cold start
        try:
            import sys as _sys
            import subprocess as _sp
            import os as _os
            _script = _os.path.join(_os.path.dirname(__file__), '..', 'scripts', 'bootstrap_specialist.py')
            _script = _os.path.normpath(_script)
            if _os.path.exists(_script) and not _os.path.exists("brain_code.uchi"):
                _sp.Popen(
                    [_sys.executable, _script, "--domain", "all"],
                    stdout=_sp.DEVNULL,
                    stderr=_sp.DEVNULL,
                )
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

    def start_background_jobs(self):
        """
        Spawn persistent background processes. Safe to call multiple times —
        guards prevent double-spawn. Called by both TUI and API on startup.
        Handles are stored in self._daemon_procs so stop_background_jobs()
        can terminate them cleanly on quit.
        """
        if getattr(self, '_background_started', False):
            return
        self._background_started = True
        if not hasattr(self, '_daemon_procs'):
            self._daemon_procs = []

        import sys, subprocess, os
        scripts_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', 'scripts')
        )

        # Legacy RL daemon (predict_future + sandbox)
        rl_script = os.path.join(scripts_dir, 'background_rl.py')
        if os.path.exists(rl_script):
            proc = subprocess.Popen(
                [sys.executable, rl_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._daemon_procs.append(proc)

        # Convergent dream daemon: ConvergentEngine self-play → SSM alignment
        # Runs forever (--daemon), periodically saving ssm_dynamics.pt.
        dream_script = os.path.join(scripts_dir, 'offline_dream.py')
        if os.path.exists(dream_script):
            proc = subprocess.Popen(
                [sys.executable, dream_script, '--daemon'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._daemon_procs.append(proc)

    def stop_background_jobs(self):
        """Terminate all daemons spawned by start_background_jobs().

        Sends SIGTERM so offline_dream.py can save its SSM checkpoint
        before exiting. Called by the TUI on quit and by the API lifespan
        on shutdown.
        """
        for proc in getattr(self, '_daemon_procs', []):
            if proc.poll() is None:
                proc.terminate()
        self._daemon_procs = []
        self._background_started = False

    def _fire_contrastive_update(
        self,
        query_tokens: list,
        response_tokens: list,
        reward: float,
    ) -> None:
        """
        Enqueue a contrastive SSM update on a daemon thread.

        The lock prevents concurrent optimizer.step() calls from colliding
        gradients or corrupting Adam's moment buffers.  Daemon=True ensures
        the thread does not block interpreter shutdown.
        """
        from uchi.vector_oracle import contrastive_update as _cu
        opt = self.ssm_optimizer
        lock = self.ssm_lock
        q = list(query_tokens)
        r = list(response_tokens)
        rew = float(reward)

        def _bg():
            with lock:
                _cu(q, r, rew, opt)

        threading.Thread(target=_bg, daemon=True).start()

    def _train_ssm(self, sequence: list, reward: float):
        """Train SSM value head + dynamics on one sequence/reward pair."""
        from uchi.neuro_symbolic import get_ssm
        ssm = get_ssm()
        ssm.train()
        self.ssm_optimizer.zero_grad()
        v_loss = ssm.update_value(sequence, reward=reward)
        d_loss = ssm.train_dynamics(sequence)
        (v_loss + d_loss).backward()
        self.ssm_optimizer.step()

    def _compute_response_reward(self, query_tokens: list, reply_tokens: list) -> float:
        """Coherence reward: bonus for appropriately-sized, non-repetitive responses."""
        n = len(reply_tokens)
        reward = 0.0
        if 5 <= n <= 50:
            reward += 0.1
        if n > 0:
            overlap = sum(1 for t in reply_tokens if t in set(query_tokens))
            if overlap / n > 0.35:
                reward -= 0.15
        return reward

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
        
    def query(self, concepts: list) -> str:
        """
        Zero-Shot Question Answering against the memory buffer.
        """
        concept_query = list(concepts)
        if self.bpe:
            concept_query = list(self.bpe.tokenize(concept_query))
        
        ans_concept, cosine_score = self.memory.query(concept_query)

        # Cosine similarity is in [-1, 1] on the unit hypersphere.
        # 0.5 = confident semantic match (per memory.query docstring).
        # Additional keyword check: on an undertrained SSM, unrelated sentences can
        # score > 0.5. Require at least one content word from the query to appear in
        # the stored text to reject false positives (e.g. "do you dream" for "thermodynamics").
        if ans_concept and cosine_score is not None and cosine_score >= 0.5:
            _stop = {"the", "a", "of", "in", "is", "that", "to", "and", "or", "it", "i"}
            _q_words = {str(c).split(".")[0].lower() for c in concept_query} - _stop
            _stored_words = {w.split(".")[0].lower() for w in str(ans_concept).split()}
            if _q_words & _stored_words:  # at least one content word overlaps
                return str(ans_concept)

        # Autonomous Web Sourcing Hook
        try:
            from uchi.web_search import perform_web_search
            # Strip OmniTokenizer lemma IDs (e.g. "capital.n.01" → "capital") so the
            # web search receives plain English rather than lemmatised synset tokens.
            raw_query = " ".join(str(c).split(".")[0] for c in concept_query)
            print(f"\n[!] Knowledge gap detected for '{raw_query}'. Engaging Autonomous Web Sourcing...")
            web_context = perform_web_search(raw_query)
            if web_context:
                print(f"[+] Sourced {len(web_context)} bytes of structural truth from the web.")
                # Stream raw content first (fills trie vocabulary)
                self.stream(web_context.split())
                # Also stream as a QA pair so <|assistant|> → answer path exists in trie.
                # Without this, MCTS can't select the web tokens as a response.
                raw_q_words = raw_query.split()
                qa_seq = ["<|user|>"] + raw_q_words + ["<|assistant|>"] + web_context[:250].split()
                self.stream(qa_seq)
                ans_concept2, ans_score2 = self.memory.query(concept_query)
                if ans_concept2 and ans_score2 is not None and ans_score2 >= 0.5:
                    return str(ans_concept2)
                # Cosine memory gate failed (cold SSM — embedding space untrained).
                # Tag the return so callers can distinguish web content from memory results
                # and return it directly when MCTS fails to use it as generation guidance.
                return "\x00WEB\x00" + web_context[:250]
        except Exception as e:
            print(f"[-] Web Sourcing error: {e}")

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
        High-level Conversational API with GRPO RL, MoE routing, and REPL-oracle code.
        """
        from uchi.omni_tokenizer import UnknownConcept

        # 1. Intent detection + procedural context prepend
        query_tokens = message.split()
        intent_key = self.procedural.get_intent_key(message)
        procedure = self.procedural.retrieve(message)
        if procedure:
            query_tokens = procedure.split() + ["|"] + query_tokens

        concepts = self.tokenizer.tokenize(query_tokens, is_inference=True)

        # 2. Sentiment-based RL on previous turn
        positive_words = {"good", "great", "awesome", "correct", "yes", "amazing", "thanks", "thank"}
        negative_words = {"bad", "wrong", "incorrect", "no", "stop", "terrible", "awful"}
        score_val = sum(
            (1 if w.lower().strip(".,!?") in positive_words else
             -1 if w.lower().strip(".,!?") in negative_words else 0)
            for w in message.split()
        )

        if hasattr(self, 'last_sequence') and self.last_sequence:
            reward = 0.0
            if score_val > 0:
                if callback: callback("reinforce", "Positive Momentum: Reinforcing previous sequence!")
                reward = 1.0
            elif score_val < 0:
                if callback: callback("prune", "Synaptic Pruning: Eradicating previous hallucination!")
                self.predictor.unlearn(self.last_sequence)
                reward = -1.0

            if reward != 0.0:
                advantage = self.baseline.advantage(reward)
                self.baseline.update(reward)
                self._train_ssm(self.last_sequence, advantage)

        _WEB_TAG = "\x00WEB\x00"

        # 3a. Convergent MCTS: unified vector routing + generation
        #     Retrieves memory bias first so candidates are grounded in context.
        #     Keyword intent (steps 3b/3c) runs AFTER as a more specific override.
        if intent_key is None:
            try:
                retrieved = self.query(concepts)
                # Web-sourced content is tagged with _WEB_TAG. Return it directly:
                # MCTS cannot reliably steer toward a sparsely-streamed trie path,
                # so we surface the retrieval result rather than hallucinating over it.
                if retrieved.startswith(_WEB_TAG):
                    web_answer = retrieved[len(_WEB_TAG):]
                    self._conversation_history.append((concepts, web_answer.split()))
                    return web_answer
                bias = retrieved if retrieved != "[Unknown Context]" else None
                kind, payload, reward_hint = self.convergent.generate(
                    concepts,
                    bootstrapped=self._knowledge_bootstrapped,
                    bias_context=bias,
                    context=self._conversation_history[-self._context_window:],
                    callback=callback,
                )
                if kind == "tool" and payload and self.skills.has(payload):
                    skill = self.skills._skills.get(payload)
                    if skill:
                        desc_toks = (skill.name + " " + skill.description).lower().split()
                        self._fire_contrastive_update(concepts, desc_toks, reward_hint)
                    return self.skills.dispatch(payload, message, callback)

                elif kind == "uncertain" and not payload and bias:
                    # MCTS produced no valid candidates but retrieval found grounding context.
                    # Return the retrieved content directly rather than the fallback string.
                    self._conversation_history.append((concepts, bias.split()))
                    return bias

                elif kind in ("text", "uncertain") and payload:
                    reply = [t for t in payload if t not in
                             ("<|user|>", "<|assistant|>", "<|end|>")]
                    if reply:
                        self.last_sequence = (
                            ["<|user|>"] + concepts + ["<|assistant|>"] + reply
                        )
                        self._fire_contrastive_update(concepts, reply, reward_hint)
                        self.stream(self.last_sequence)
                        self.last_walk_data = {
                            "prediction_depth": getattr(
                                self.predictor._pred, "_last_prediction_depth", 0
                            ),
                            "contributions": getattr(
                                self.predictor._pred, "_last_contributions", {}
                            ),
                            "max_sim": getattr(
                                self.predictor._pred, "_last_max_sim", 0.0
                            ),
                        }
                        predicted = self.tokenizer.detokenize(reply)
                        reply_str = " ".join(str(p) for p in predicted)
                        if reply_str.strip():
                            if kind == "uncertain":
                                # Memory fallback: associative memory may know
                                # what the MCTS could not generate from trie alone.
                                mem_result = self.query(concepts)
                                if mem_result.startswith(_WEB_TAG):
                                    mem_result = mem_result[len(_WEB_TAG):]
                                if mem_result != "[Unknown Context]":
                                    self._conversation_history.append((concepts, mem_result.split()))
                                    return mem_result
                                self._conversation_history.append((concepts, reply))
                                return f"[Uncertain] {reply_str}"
                            self._conversation_history.append((concepts, reply))
                            return reply_str
            except Exception:
                pass

        # 3b. Analytical intents: auto-dispatch when a data file path is present
        _ANALYTICAL = {"classify", "regress", "anomaly", "forecast", "tsclassify"}
        if intent_key in _ANALYTICAL:
            import os as _os
            from uchi.data_loader import find_path as _fp
            data_path = _fp(message)
            if data_path and _os.path.exists(data_path):
                return self.skills.dispatch(intent_key, message, callback)
            elif data_path:
                return f"File not found: {data_path}"
            else:
                return (
                    f"I can run {intent_key} analysis. "
                    f"Please provide a data file path, e.g.:\n"
                    f"  /{intent_key} yourdata.csv"
                )

        # 3c. Phase 1+2: Route CODE intent through CodeEngine + SpecialistPool
        if intent_key == "code":
            return self._handle_code_intent(message, query_tokens, concepts, callback)

        # 4. Conversation path: memory retrieval + MCTS prediction
        retrieved_context_str = self.query(concepts)
        # Strip web tag if present (specialist path also handles web-sourced content)
        if retrieved_context_str.startswith(_WEB_TAG):
            web_answer = retrieved_context_str[len(_WEB_TAG):]
            self._conversation_history.append((concepts, web_answer.split()))
            return web_answer
        # Prepend prior turns so specialist predictor sees rolling context
        history_prefix = []
        for q_toks, r_toks in self._conversation_history[-self._context_window:]:
            history_prefix += ["<|user|>"] + q_toks[:6] + ["<|assistant|>"] + r_toks[:6]
        tokens = history_prefix + ["<|user|>"] + concepts + ["<|assistant|>"]

        bias = retrieved_context_str if retrieved_context_str != "[Unknown Context]" else None
        prompt_entropy = self.predictor.score(tokens)

        # Route to specialist if available (Phase 2)
        specialist = self.specialist_pool.route(intent_key or "convo")
        # Stream grounding context to specialist trie so its MCTS can select those tokens.
        # Web/memory content is only in the main trie; specialist has its own trie.
        # Frame as a proper QA pair so <|assistant|> → answer path exists in trie.
        if bias and specialist is not self:
            raw_q_words = [str(c).split(".")[0] for c in concepts]
            qa_seq = ["<|user|>"] + raw_q_words + ["<|assistant|>"] + bias.split()
            specialist.stream(qa_seq)
        pred = specialist.predict_future(tokens, steps=60, temperature=0.0, creativity=0.0, bias_context=bias)

        # 5. Hallucination gate — bypassed when retrieval provides grounding context.
        # If we have bias (memory match or web snippet), trust the grounded generation.
        if pred and bias is None:
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

        self.last_walk_data = {
            "prediction_depth": self.predictor._pred._last_prediction_depth,
            "contributions": self.predictor._pred._last_contributions,
            "max_sim": self.predictor._pred._last_max_sim,
        }

        # 6. Extract reply tokens
        reply = []
        for p in pred:
            if p in ("<|user|>", "<|assistant|>", "<|end|>"):
                break
            reply.append(p)

        if reply:
            self.last_sequence = ["<|user|>"] + query_tokens + ["<|assistant|>"] + reply
        else:
            self.last_sequence = None

        predicted = self.tokenizer.detokenize(reply)
        reply_str = " ".join(str(p) for p in predicted)

        if not reply_str.strip():
            if callback: callback("hallucination", "Empty prediction path.")
            self.last_sequence = None
            return "I do not know the answer to that, even after searching the web."

        # 7. Coherence reward: length + repetition signal
        if self.last_sequence:
            coh_reward = self._compute_response_reward(query_tokens, reply)
            if coh_reward != 0.0:
                self.baseline.update(coh_reward)
                self._train_ssm(self.last_sequence, coh_reward)

        # 8. Stream full interaction AFTER generation (Root 3 fix)
        if self.last_sequence:
            self.stream(self.last_sequence)

        if reply:
            self._conversation_history.append((concepts, reply))

        return reply_str

    def _handle_code_intent(self, message: str, query_tokens: list, concepts: list, callback) -> str:
        """
        Phase 1+2: Parallel MCTS code generation with REPL oracle.
        Uses specialist brain predictor if available.
        """
        # Use specialist predictor when its brain is bootstrapped
        if self.specialist_pool.has_specialist("code"):
            active_predictor = self.specialist_pool.get_predictor("code")
            engine = CodeEngine(active_predictor, n_workers=4)
        else:
            engine = self.code_engine

        seed_tokens = ["<|user|>"] + concepts + ["<|assistant|>"]
        code_str, reward, passed = engine.generate_code(seed_tokens, max_tokens=80)

        if passed:
            if callback: callback("reinforce", "REPL oracle: code compiled successfully!")
        else:
            if callback: callback("prune", "REPL oracle: no candidate compiled — showing best attempt.")

        # GRPO: train SSM on code result
        self.last_sequence = ["<|user|>"] + query_tokens + ["<|assistant|>"] + code_str.split()
        advantage = self.baseline.advantage(reward)
        self.baseline.update(reward)
        self._train_ssm(self.last_sequence, advantage)

        if not passed and reward < 0:
            self.predictor.unlearn(self.last_sequence)

        self.stream(self.last_sequence)
        return code_str

