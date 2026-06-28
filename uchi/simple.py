"""Uchi — single import, everything discoverable.

The ``Uchi`` class is the canonical public interface for the Uchi library.
Import once, access everything: the generative brain, the sequence predictor,
all analytical tools, and persistent brain state.

    from uchi import Uchi

    u = Uchi()
    u.learn("Q3 revenue was $4.2M, up 23% YoY.")
    print(u.ask("What was Q3 revenue growth?"))

Compounding analysis — the core value proposition
--------------------------------------------------
``ask()`` always returns a plain string.
``learn()`` always accepts a plain string.
This means the output of any analysis is immediately learnable by any other
``Uchi`` instance. Knowledge compounds across instances without any glue code:

    # Instance 1: run classification on your dataset
    u1 = Uchi()
    report = u1.ask("/classify", X=X_train, y=y_train)

    # Instance 2: treat that report as learned knowledge
    u2 = Uchi()
    u2.learn(report)
    u2.ask("What accuracy did we achieve and what does it imply for Q4?")

Each ``ask()`` result can feed the next ``learn()``. Pipelines of Uchi
instances build compounding analytical context without any external
orchestration layer.
"""

from __future__ import annotations

import gzip
import os
import pickle
from typing import Any, Optional


class Uchi:
    """Single entry-point for the Uchi toolkit.

    One import. Everything discoverable. Outputs always compound.

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

        u = Uchi()
        u.learn("The boiling point of water is 100°C at sea level.")
        u.ask("At what temperature does water boil?")

    Ingest files and directories:

        u.ingest("knowledge_base/")          # walks directory, all text/md/py/json/csv
        u.ingest("report.pdf")               # requires pip install pdfminer.six
        u.ingest("events.csv", col="notes")  # specific CSV column
        # chainable
        u = Uchi().ingest("docs/").ingest("data.csv").ingest("report.md")

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

    Toggle web search at any time:

        u.web_search = True   # live web sourcing on knowledge gaps
        u.web_search = False  # back to fully offline

    Escape hatch for power users:

        u.router   # the underlying OmniRouter
    """

    _DEFAULT_BRAIN = os.path.join(os.path.dirname(__file__), "data", "brain.uchi")

    def __init__(
        self,
        brain_path: Optional[str] = None,
        web_search: bool = False,
    ) -> None:
        from .omni_router import OmniRouter

        resolved = brain_path or self._DEFAULT_BRAIN
        router: Optional[OmniRouter] = None

        if os.path.exists(resolved):
            try:
                with gzip.open(resolved, "rb") as f:
                    router = pickle.load(f)
            except Exception:
                try:
                    with open(resolved, "rb") as f:
                        router = pickle.load(f)
                except Exception:
                    pass

        if router is None:
            OmniRouter._bootstrap_knowledge = lambda self, *a, **kw: None
            OmniRouter._bootstrap_persona = lambda self, *a, **kw: None
            router = OmniRouter(use_bpe=False)

        router.web_search_enabled = web_search
        self._router = router

    # ── brain interface ───────────────────────────────────────────────────────

    def learn(self, text: str) -> None:
        """Stream text into the brain's knowledge trie.

        Accepts any string — a sentence, a document, or the string output of
        a previous ``ask()`` call. That last case is the compounding mechanism:
        the analysis produced by one ``Uchi`` instance becomes learnable
        knowledge for another, with no serialisation or schema required.
        """
        self._router.stream(text.split())

    def ask(self, question: str, **data: Any) -> str:
        """Ask the brain a question or invoke a tool skill.

        Natural-language questions route through the full convergent engine
        (trie + MCTS + SSM + HRR fallback).

        Slash commands with ``**data`` keyword arguments invoke the
        corresponding analytical skill directly, bypassing string parsing:

            u.ask("/classify", X=X_train, y=y_train)
            u.ask("/regress",  X=X_train, y=y_train)
            u.ask("/anomaly",  X=X)
            u.ask("/forecast", X=time_series, steps=20)
            u.ask("/tsclassify", X=windows, y=labels)

        All forms return a plain string normalised for human readability —
        synset tokens, internal control markers, and tokeniser artefacts are
        stripped before the result is returned.  This guarantees that
        ``u2.learn(u1.ask(...))`` chains always feed clean English into the
        trie rather than raw trie vocabulary, creating a self-reinforcing
        quality loop.
        """
        from .response_normalizer import normalize
        if question.startswith("/") and data:
            parts = question.lstrip("/").split(None, 1)
            cmd = parts[0].lower()
            extra_args = parts[1] if len(parts) > 1 else ""
            raw = self._router.skills.dispatch(cmd, extra_args, data_kwargs=data) or ""
        else:
            raw = self._router.chat(question) or ""
        return normalize(raw)

    def ingest(self, path: str, col: Optional[str] = None) -> "Uchi":
        """Load files or directories into the brain.

        Walks *path* recursively if it is a directory. Each file is read,
        converted to text, and streamed through ``learn()``. Returns ``self``
        so calls can be chained::

            u = Uchi().ingest("docs/").ingest("reports/").ingest("events.csv")

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

    def stream(self, tokens: list) -> None:
        """Feed a raw token sequence directly into the trie (low-level path)."""
        self._router.stream(tokens)

    # ── sequence predictor ────────────────────────────────────────────────────

    @property
    def predictor(self):
        """The SequenceGenerator powering the brain's trie.

        Full sklearn-compatible sequence API:

            u.predictor.fit(sequences)              # train on list of sequences
            u.predictor.partial_fit(sequences)      # online update
            u.predictor.generate(n=10, seed=["x"]) # sample n tokens
            u.predictor.train(sequence)             # single online sequence
            u.predictor.predict_next(context)       # argmax next token
            u.predictor.score(sequence)             # bits-per-token
        """
        return self._router.predictor

    # ── config ────────────────────────────────────────────────────────────────

    @property
    def web_search(self) -> bool:
        """Whether autonomous web sourcing is enabled on knowledge gaps."""
        return getattr(self._router, "web_search_enabled", False)

    @web_search.setter
    def web_search(self, value: bool) -> None:
        self._router.web_search_enabled = bool(value)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist the current brain state to *path*."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with gzip.open(path, "wb") as f:
            pickle.dump(self._router, f)

    # ── escape hatch ─────────────────────────────────────────────────────────

    @property
    def router(self):
        """Direct access to the underlying OmniRouter for advanced use."""
        return self._router
