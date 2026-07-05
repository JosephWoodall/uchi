"""Core — the raw single-instance Uchi engine.

``Core`` is the lightweight, single-instance engine: the generative brain, the
sequence predictor, all analytical tools, and persistent brain state, with no
orchestration layered on top. Most users should import ``Uchi`` from the top
of the package instead (``from uchi import Uchi``), which wraps ``Core`` in
the ``MetaUchi`` facade (see ``uchi/meta.py``). Import ``Core`` directly only
if you want the raw, un-orchestrated node.

    from uchi import Core

    u = Core()
    u.learn("Q3 revenue was $4.2M, up 23% YoY.")
    print(u.ask("What was Q3 revenue growth?"))

Compounding analysis — the core value proposition
--------------------------------------------------
``ask()`` always returns a plain string.
``learn()`` always accepts a plain string.
This means the output of any analysis is immediately learnable by any other
``Core`` instance. Knowledge compounds across instances without any glue code:

    # Instance 1: run classification on your dataset
    u1 = Core()
    report = u1.ask("/classify", X=X_train, y=y_train)

    # Instance 2: treat that report as learned knowledge
    u2 = Core()
    u2.learn(report)
    u2.ask("What accuracy did we achieve and what does it imply for Q4?")

Each ``ask()`` result can feed the next ``learn()``. Pipelines of Core
instances build compounding analytical context without any external
orchestration layer.
"""

from __future__ import annotations

import gzip
import os
import pickle
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .goal_state import GoalState


class Core:
    """The raw, single-instance Uchi engine.

    One import. Everything discoverable. Outputs always compound. This is
    the unorchestrated node ``MetaUchi`` wraps by default — use it directly
    only when you want lightweight, raw trie access with no facade on top.

    Parameters
    ----------
    brain_path : str, optional
        Path to a ``brain.uchi`` file. Falls back to the pre-packaged brain
        shipped with the library. Pass ``None`` to use the default.
    web_search : bool
        Enable autonomous web sourcing when the brain has a knowledge gap.
        Default ``False`` — the brain runs fully offline.

    Examples
    --------
    Knowledge & Q&A:

        u = Core()
        u.learn("The boiling point of water is 100°C at sea level.")
        u.ask("At what temperature does water boil?")

    Ingest files and directories:

        u.ingest("knowledge_base/")          # walks directory, all text/md/py/json/csv
        u.ingest("report.pdf")               # requires pip install pdfminer.six
        u.ingest("events.csv", col="notes")  # specific CSV column
        # chainable
        u = Core().ingest("docs/").ingest("data.csv").ingest("report.md")

    Analytical tools via slash commands:

        result = u.ask("/classify", X=X_train, y=y_train)
        result = u.ask("/regress",  X=X_train, y=y_train)
        result = u.ask("/anomaly",  X=sensor_matrix)
        result = u.ask("/forecast", X=time_series, steps=20)

    Sequence generation via the underlying predictor:

        u.predictor.fit([["a", "b", "c", "d"]])
        u.predictor.generate(n=10, seed=["a", "b"])
        u.predictor.train(["a", "b", "c", "d"])
        u.predictor.predict_next(["b", "c"])   # → "d"

    Escape hatch for power users:

        u.pipeline   # the underlying GenerateAndGround pipeline
    """

    _DEFAULT_BRAIN = os.path.join(os.path.dirname(__file__), "data", "brain.uchi")

    def __init__(
        self,
        brain_path: Optional[str] = None,
        web_search: bool = False,
    ) -> None:
        from .retrieval import SemanticIndex
        from .oracle import FactCheckOracle
        from .proposer import FluxProposer
        from .generate_and_ground import GenerateAndGround
        from .skill_registry import SkillRegistry

        self.web_search_enabled = web_search
        
        # New Uchi Architecture Components
        from .brain_fetch import get_embeddings_path
        embed_path = get_embeddings_path()   # bundled -> cached -> download -> None
        if embed_path is not None:
            self.index = SemanticIndex.from_embeddings_file(embed_path)
        else:
            import numpy as np
            self.index = SemanticIndex({}, np.zeros((1, 1), dtype=np.float32))
            
        self.oracle = FactCheckOracle()
        
        # Load the trained FLUX checkpoint (the Proposer). flux_best.pt is the
        # canonical artifact produced by scripts/train_all.sh; the per-phase
        # checkpoints are fallbacks in pipeline order. If none exist, the proposer
        # degrades to None and the verifier falls back to grounded extraction /
        # honest abstention — Uchi still works, just without FLUX's generation.
        checkpoint_dir = os.path.join(os.path.dirname(__file__), "flux", "checkpoints")
        best_ckpt = next(
            (p for p in (os.path.join(checkpoint_dir, n) for n in
                         ("flux_best.pt", "qat_best.pt", "cot_best.pt", "sft_best.pt"))
             if os.path.exists(p)),
            None,
        )
        self.proposer = FluxProposer.load(checkpoint=best_ckpt) if best_ckpt else None
        
        self.pipeline = GenerateAndGround(
            index=self.index,
            oracle=self.oracle,
            proposer=self.proposer
        )
        
        # New: Episodic Memory
        from .episodic_memory import EpisodicMemory
        self.episodic_memory = EpisodicMemory()
        
        # New: Procedural Memory (Autonomous Tool Creation)
        from .procedural_memory import ProceduralMemory
        self.procedural_memory = ProceduralMemory(proposer=self.proposer)
        
        # Skill Registry (legacy adapter)
        self.skills = SkillRegistry(self)
        
        # New: Swarm Synthesizer (Default Map-Reduce behavior)
        from .swarm import SwarmSynthesizer
        self.swarm = SwarmSynthesizer(self.pipeline)

        # New: Tool Calling (0.4.0 Item 4) — filesystem ops + Python scratchpad
        # + (0.4.0 Item 11) web search, gated by the web_search flag above.
        from .tool_calling import default_registry
        self.tools = default_registry(enable_web_search=self.web_search_enabled)

        # New: Goal State (0.4.0 Item 5) — set via start_goal(); inert by
        # default so single-shot ask() calls are unaffected.
        self.goal_state = None

        # New: HitL Yielding (0.4.0 Item 10) — set when ask() pauses on a
        # <|yield_to_user|> or an auto-escalated loop-guard block; the next
        # ask() call is treated as the human's answer to it.
        self.pending_yield = None

        # Advanced SDK sequence predictor (trie), constructed lazily on first use.
        self._predictor = None

    def start_goal(self, goal: str) -> "GoalState":
        """Begin tracking a multi-step task under *goal*.

        While active, every tool call dispatched during ``ask()`` is
        recorded into the returned ``GoalState`` and compacted once the
        raw log grows past the threshold; ``goal_state.context_string()``
        is folded into subsequent ``ask()`` calls so the task never loses
        the original intent, however many steps it takes.
        """
        from .goal_state import GoalState
        self.goal_state = GoalState(goal=goal)
        return self.goal_state

    def end_goal(self) -> None:
        """Stop tracking the active goal (subsequent ask() calls stop
        injecting goal context / recording tool calls into it)."""
        self.goal_state = None

    def learn_tools(self, path: str) -> list:
        """Parse *path* and register every top-level Python function as a
        tool callable via ``<|tool_call|>`` — no source changes to Uchi
        itself required. Each function's docstring and signature are also
        ingested into the knowledge index, so they're discoverable the
        same way any other learned fact is.
        """
        from .tool_learning import learn_tools as _learn_tools
        learned = _learn_tools(path, self.tools)
        for t in learned:
            self.learn(t.as_knowledge())
        return learned

    @property
    def predictor(self):
        """Underlying sequence predictor (CTW-style trie) for power users.

        Exposes ``fit`` / ``train`` / ``predict_next`` / ``generate`` for raw
        sequence modelling. Built on first access so it costs nothing unless used.
        """
        if self._predictor is None:
            from .predictor import UniversalPredictor
            self._predictor = UniversalPredictor(context_length=8)
        return self._predictor

    # ── Legacy adapters for SkillRegistry ──────────────────────────────────────
    def chat(self, msg: str, callback=None) -> str:
        return self.ask(msg)
        
    def stream(self, tokens: list[str]) -> None:
        self.learn(" ".join(tokens))
        
    def query(self, tokens: list[str]) -> str:
        return self.ask(" ".join(tokens))

    # ── brain interface ───────────────────────────────────────────────────────

    def learn(self, text: str) -> None:
        """Stream text into the brain's knowledge semantic index.

        Accepts any string — a sentence, a document, or the string output of
        a previous ``ask()`` call. That last case is the compounding mechanism:
        the analysis produced by one ``Uchi`` instance becomes learnable
        knowledge for another, with no serialisation or schema required.
        """
        try:
            self.index.build_from_corpus(text)
        except Exception as e:
            print(f"[-] Failed to learn: {e}")

    def ask(self, question: str, callback=None, **data: Any) -> str:
        """Ask the brain a question or invoke a tool skill.

        Natural-language questions route through the FLUX + Uchi verifier pipeline.

        Slash commands with ``**data`` keyword arguments invoke the
        corresponding analytical skill directly.
        """
        from .response_normalizer import normalize
        if question.startswith("/") and data:
            parts = question.lstrip("/").split(None, 1)
            cmd = parts[0].lower()
            extra_args = parts[1] if len(parts) > 1 else ""
            raw = self.skills.dispatch(cmd, extra_args, data_kwargs=data, callback=callback) or ""
        elif question.startswith("/"):
            parts = question.lstrip("/").split(None, 1)
            cmd = parts[0].lower()
            extra_args = parts[1] if len(parts) > 1 else ""
            raw = self.skills.dispatch(cmd, extra_args, callback=callback) or ""
        else:
            # Inject episodic memory context, plus the active goal state's
            # context (goal + compacted notes) if start_goal() is active.
            # If a HitL yield is pending (Item 10), this question IS the
            # human's answer to it — fold it in and clear the pending yield.
            context = self.episodic_memory.get_context_string(n_turns=3)
            goal_context = self.goal_state.context_string() if self.goal_state else ""
            pending_context = ""
            if self.pending_yield:
                pending_context = f"Uchi previously asked: {self.pending_yield!r}\nHuman answered: {question}"
                self.pending_yield = None
            combined_context = "\n\n".join(c for c in (pending_context, goal_context, context) if c)
            augmented_question = f"{combined_context}\n\nQuestion: {question}" if combined_context else question

            # Route to Swarm by default
            raw = self.swarm.answer(augmented_question, callback=callback) or ""

            # Dispatch any <|tool_call_async|> calls concurrently first
            # (Item 7), then any remaining sequential <|tool_call|> calls
            # (Item 4), splicing results back before it's saved/returned.
            # Recorded into the active goal state (if any) so long tasks
            # compact instead of losing the plot.
            from .tool_calling import run_async_tool_calls, run_with_tools
            log_before = len(self.tools.log)
            raw = run_async_tool_calls(raw, self.tools, goal_state=self.goal_state)
            raw = run_with_tools(raw, self.tools, goal_state=self.goal_state)
            new_entries = self.tools.log[log_before:]

            # HitL Yielding (Item 10): an explicit <|yield_to_user|> marker,
            # or a tool call auto-escalated because the loop guard (Item 6)
            # blocked an exact repeat of a prior failure, pauses the
            # response instead of returning it as a normal answer.
            from .hitl import format_yield, is_blocked_by_loop_guard, parse_yield
            explicit_yield = parse_yield(raw)
            blocked = next((e for e in new_entries if is_blocked_by_loop_guard(e)), None)
            if explicit_yield is not None:
                self.pending_yield = explicit_yield.question
                raw = format_yield(explicit_yield.question)
            elif blocked is not None:
                clarifying = (
                    f"The '{blocked.name}' tool keeps failing with the same arguments "
                    f"({blocked.args}). How would you like me to proceed?"
                )
                self.pending_yield = clarifying
                raw = format_yield(clarifying)

            # Save to episodic memory
            self.episodic_memory.add_interaction(question, raw)

        return normalize(raw)

    def ingest(self, path: str, col: Optional[str] = None) -> "Core":
        """Load files or directories into the brain.

        Walks *path* recursively if it is a directory. Each file is read,
        converted to text, and streamed through ``learn()``. Returns ``self``
        so calls can be chained::

            u = Core().ingest("docs/").ingest("reports/").ingest("events.csv")

        Supported formats
        -----------------
        - Plain text: ``.txt`` ``.md`` ``.rst`` ``.py`` ``.yaml`` ``.yml``
          ``.toml`` ``.ini`` ``.cfg`` ``.sh``
        - ``.csv`` — all text cells, or a single column when *col* is given
        - ``.json`` — all string values extracted recursively
        - ``.pdf`` — requires ``pip install pdfminer.six``; skipped with a
          warning when the package is absent

        Unrecognised extensions and unreadable files are silently skipped so
        that an entire project directory can be ingested safely.

        Parameters
        ----------
        path : str
            File or directory to ingest.
        col : str, optional
            For CSV files: the column name whose values are fed into the
            brain. When *None* every text-valued cell is concatenated.
        """
        import os
        path = os.path.expanduser(str(path))
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for fname in sorted(files):
                    self._ingest_file(os.path.join(root, fname), col=col)
        else:
            self._ingest_file(path, col=col)
        return self

    def _ingest_file(self, path: str, col: Optional[str] = None) -> None:
        import os
        ext = os.path.splitext(path)[1].lower()
        _TEXT_EXTS = {
            ".txt", ".md", ".rst", ".py", ".yaml", ".yml",
            ".toml", ".ini", ".cfg", ".sh",
        }
        try:
            if ext in _TEXT_EXTS:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    self.learn(fh.read())
            elif ext == ".csv":
                self._ingest_csv(path, col=col)
            elif ext == ".json":
                self._ingest_json(path)
            elif ext == ".pdf":
                self._ingest_pdf(path)
        except Exception:
            pass  # skip unreadable files; don't abort a directory walk

    def _ingest_csv(self, path: str, col: Optional[str] = None) -> None:
        import csv
        with open(path, encoding="utf-8", errors="ignore", newline="") as fh:
            for row in csv.DictReader(fh):
                if col is not None:
                    text = str(row.get(col, "") or "").strip()
                else:
                    text = " ".join(
                        str(v) for v in row.values() if v and str(v).strip()
                    )
                if text:
                    self.learn(text)

    def _ingest_json(self, path: str) -> None:
        import json
        with open(path, encoding="utf-8", errors="ignore") as fh:
            data = json.load(fh)
        text = self._extract_strings(data)
        if text:
            self.learn(text)

    def _extract_strings(self, obj: Any) -> str:
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            return " ".join(self._extract_strings(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return " ".join(self._extract_strings(item) for item in obj)
        return ""

    def _ingest_pdf(self, path: str) -> None:
        try:
            import pdfminer.high_level as _pdf  # type: ignore[import]
            text = _pdf.extract_text(path) or ""
            if text.strip():
                self.learn(text)
        except ImportError:
            import warnings
            warnings.warn(
                f"PDF ingestion requires pdfminer.six: pip install pdfminer.six "
                f"— skipping {path}",
                stacklevel=4,
            )

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist the current semantic index state to *path*."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with gzip.open(path, "wb") as f:
            pickle.dump(self.index, f)
