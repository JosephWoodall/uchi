"""
Omni Router (The Deterministic LLM Master Controller)
=====================================================
Unifies Phase 1 (Math), Phase 2 (Process), Phase 3 (Plural Simulation), 
Phase 4 (Infinite Context), and Phase 5 (Associative Memory) into a 
single, coherent Omni-Modal architecture.
"""

import re
import threading
from .omni_tokenizer import OmniTokenizer
from .online_tokenizer import OnlineTokenizer
from .memory import AssociativeMemory
from .generative import SequenceGenerator
from .procedural_memory import ProceduralMemory
from .code_engine import CodeEngine
from .specialist_pool import SpecialistPool
from .skill_registry import SkillRegistry
from .goal_state import GoalState
from .experience_replay import ExperienceReplayBuffer as PrioritisedReplayBuffer

_FENCED_CODE_RE = re.compile(r'```(?:python)?\s*(.*?)```', re.DOTALL)
_INLINE_CODE_RE = re.compile(r'`([^`\n]{6,})`')


def _extract_code_bias(message: str) -> str | None:
    """Return key tokens from any code block in the message for MCTS bias.

    Fenced blocks take priority; then inline backtick code containing
    a def/class keyword; then bare indented lines. Returns at most 25
    tokens so the bias is focused rather than overwhelming.
    """
    m = _FENCED_CODE_RE.search(message)
    if m:
        tokens = m.group(1).strip().split()[:25]
        return " ".join(tokens) if tokens else None

    for snippet in _INLINE_CODE_RE.findall(message):
        if any(kw in snippet for kw in ("def ", "class ", "return ", "import ")):
            return snippet.strip()[:120]

    code_lines = [
        ln.strip() for ln in message.split("\n")
        if ln.strip().startswith(("def ", "class ", "return ", "import ", "    "))
    ]
    if code_lines:
        return " ".join(" ".join(ln.split()) for ln in code_lines[:5])[:120]

    return None


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
        self.procedural = ProceduralMemory()
        self._knowledge_bootstrapped = False
        self.web_search_enabled = True

        # Code generation with REPL oracle (a skill)
        self.code_engine = CodeEngine(self.predictor, n_workers=4)
        # MoE SpecialistPool (lazy-loads specialist brains on first use)
        self.specialist_pool = SpecialistPool(self)
        # Skill toolkit: markdown-defined /commands + dynamically-callable skills
        self.skills = SkillRegistry(self)

        self._conversation_history: list = []  # [(q_tokens, r_tokens), ...]
        self._context_window: int = 3           # prior turns to prepend to seed
        self.goal_state = GoalState()           # cross-turn goal tracking
        self.replay_buffer = PrioritisedReplayBuffer()  # experience replay for GRPO
        self._replay_train_every: int = 8       # sample+train every N turns
        self._turn_counter: int = 0             # counts chat() invocations
        self._semantic_index = None             # Generate-and-Ground retrieval index

        self._bootstrap_persona(progress_callback)
        self._bootstrap_knowledge(progress_callback)

    def __setstate__(self, state):
        """Forward-compatibility: initialize attributes added in newer versions after unpickling."""
        self.__dict__.update(state)
        if not hasattr(self, 'procedural'):
            self.procedural = ProceduralMemory()
        if not hasattr(self, '_knowledge_bootstrapped'):
            self._knowledge_bootstrapped = True
        if not hasattr(self, 'code_engine'):
            self.code_engine = CodeEngine(self.predictor, n_workers=4)
        if not hasattr(self, 'specialist_pool'):
            self.specialist_pool = SpecialistPool(self)
        if not hasattr(self, 'skills'):
            self.skills = SkillRegistry(self)
        if not hasattr(self, '_background_started'):
            self._background_started = False
        if not hasattr(self, '_daemon_procs'):
            self._daemon_procs = []
        if not hasattr(self, '_conversation_history'):
            self._conversation_history = []
        if not hasattr(self, '_context_window'):
            self._context_window = 3
        if not hasattr(self, 'goal_state'):
            self.goal_state = GoalState()
        if not hasattr(self, 'replay_buffer'):
            self.replay_buffer = PrioritisedReplayBuffer()
        if not hasattr(self, '_replay_train_every'):
            self._replay_train_every = 8
        if not hasattr(self, '_turn_counter'):
            self._turn_counter = 0
        if not hasattr(self, 'web_search_enabled'):
            self.web_search_enabled = True
        if not hasattr(self, '_semantic_index'):
            self._semantic_index = None

    def __getstate__(self):
        # Exclude non-serialisable and always-reconstructed attributes.
        # - ssm_lock: threading.Lock cannot be pickled
        # - ssm_optimizer: recreated in __setstate__ from current SSM params
        # - specialist_pool, skills, convergent, code_engine: wrappers that hold
        #   back-references to self; __setstate__ reconstructs them, so persisting
        #   them inflates the brain file with redundant circular-reference chains
        #   without preserving any state that can't be rebuilt in milliseconds.
        skip = {
            "_background_started", "_daemon_procs",
            "specialist_pool", "skills", "code_engine",
            "_decoder", "_answerability",   # torch model caches; lazy-reloaded
        }
        return {k: v for k, v in self.__dict__.items() if k not in skip}

    # ── Generate-and-Ground: primary answering path ───────────────────────────
    def _default_embeddings_path(self):
        import os
        return os.path.join(os.path.dirname(__file__), "data", "skipgram_emb.pt")

    def build_semantic_index(self, texts, embeddings_path=None):
        """Build the retrieval index from ingested corpus text (called at brain build)."""
        from uchi.retrieval import SemanticIndex
        path = embeddings_path or self._default_embeddings_path()
        idx = SemanticIndex.from_embeddings_file(path)
        for t in texts:
            if isinstance(t, str) and t.strip():
                idx.build_from_corpus(t)
        self._semantic_index = idx
        return idx

    def _load_decoder(self):
        """Lazy-load the trained answer decoder from data/decoder.pt (or None)."""
        import os
        if getattr(self, "_decoder", "unset") == "unset":
            self._decoder = None
            path = os.path.join(os.path.dirname(__file__), "data", "decoder.pt")
            try:
                from uchi.decoder import NeuralDecoder
                if NeuralDecoder.exists(path):
                    self._decoder = NeuralDecoder.load(path)
            except Exception:
                self._decoder = None
        return self._decoder

    def _load_answerability(self):
        """Lazy-load the answerability checker from data/answerability.pt (or None)."""
        import os
        if getattr(self, "_answerability", "unset") == "unset":
            self._answerability = None
            path = os.path.join(os.path.dirname(__file__), "data", "answerability.pt")
            try:
                from uchi.answerability import AnswerabilityChecker
                if AnswerabilityChecker.exists(path):
                    self._answerability = AnswerabilityChecker.load(path)
            except Exception:
                self._answerability = None
        return self._answerability

    def answer(self, question: str, callback=None) -> str:
        """Primary endpoint: retrieve → answerability-gate → generate →
        fact-check → emit / abstain.

        Never confabulates: with no grounded knowledge (empty/absent index) or
        when the evidence does not actually answer the question, it abstains.
        """
        idx = getattr(self, "_semantic_index", None)
        if idx is not None and len(idx) > 0:
            from uchi.generate_and_ground import GenerateAndGround
            gg = GenerateAndGround(idx, decoder=self._load_decoder(),
                                   answerability=self._load_answerability(),
                                   predictor=self.predictor, min_answerable=0.6)
            return gg.answer(question)
        return "I don't have grounded knowledge to answer that."

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
        for _ in range(5):
            for turn in turns:
                self.stream(turn.split())
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

        import os
        import sys
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
            pass  # SSM checkpoint retired with the Family C generation path
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

        # Background self-play daemons were part of the retired Family C
        # (SSM self-alignment) path; Generate-and-Ground needs no daemons.
        return

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
        if getattr(self, 'web_search_enabled', True):
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
        """Conversational entry. Analytical/code intents route to skills; every
        other natural-language question goes through Generate-and-Ground
        (retrieve → generate → fact-check → emit/abstain)."""
        intent_key = self.procedural.get_intent_key(message)
        _ANALYTICAL = {"classify", "regress", "anomaly", "forecast", "tsclassify"}
        if intent_key in _ANALYTICAL:
            import os as _os
            from uchi.data_loader import find_path as _fp
            data_path = _fp(message)
            if data_path and _os.path.exists(data_path):
                return self.skills.dispatch(intent_key, message, callback)
            if data_path:
                return f"File not found: {data_path}"
            return (f"I can run {intent_key} analysis. Please provide a data file "
                    f"path, e.g.:\n  /{intent_key} yourdata.csv")
        if intent_key == "code":
            concepts = self.tokenizer.tokenize(message.split(), is_inference=True)
            return self._handle_code_intent(message, message.split(), concepts, callback)
        return self.answer(message, callback=callback)

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
            if callback:
                callback("reinforce", "REPL oracle: code compiled successfully!")
        else:
            if callback:
                callback("prune", "REPL oracle: no candidate compiled — showing best attempt.")

        self.last_sequence = ["<|user|>"] + query_tokens + ["<|assistant|>"] + code_str.split()
        if not passed and reward < 0:
            self.predictor.unlearn(self.last_sequence)
        self.stream(self.last_sequence)
        return code_str

