"""code_retrieval.py — repo-scale code retrieval for SWE-bench-shaped tasks
(0.5.0 Item 3.5).

Distinct from ``uchi/retrieval.py``: that module indexes free-text brain
passages (skip-gram word vectors trained on natural-language prose, English
stopword filtering, sentence splitting) for Generate-and-Ground QA. This
module indexes an arbitrary, unfamiliar *source repo* at file/function
granularity so a context-limited model can retrieve the relevant slice of a
large codebase instead of needing the whole repo in context. Both share the
same two-stage "skip-gram cosine, then deterministic rerank" shape and the
same non-negotiable constraint (no pretrained model, anywhere) — but the
mechanics differ because code and prose are different substrates:

  - **Tokenizer**: identifiers are split on snake_case/camelCase boundaries
    and Python keywords/``self``/``cls`` are dropped as the code-corpus
    equivalent of ``retrieval.py``'s English stopword list — an English
    stopword filter is meaningless for code, but code has its own filler
    words (``def``, ``return``, ``self``) that appear in nearly every chunk
    and carry no topical signal.
  - **Embeddings are trained per-repo, from scratch, at index-build time**
    (see ``_train_skipgram``) rather than loaded from a pre-shipped brain
    corpus — an unfamiliar repo's identifier vocabulary ("_handler",
    "sync_config") shares ~nothing with the brain's natural-language
    co-occurrence statistics, so reusing ``retrieval.py``'s embeddings would
    not carry code semantics. Training is hand-written NumPy (SGNS: Mikolov
    et al. 2013), not autograd — auditable end to end, and fast enough at
    repo scale (thousands, not billions, of tokens) that torch is not
    needed anywhere in this module, only at inference like ``retrieval.py``.
  - **Rerank replaces lexical-overlap-only with lexical overlap *and*
    import-graph proximity.** ``retrieval.py``'s hybrid rerank has no
    notion of "this passage is related because it lives next to the one I
    already know matters" — code does: a chunk in a file the current
    context imports (or is imported by) is a real cross-file navigation
    signal a flat lexical rerank can't see on its own.
  - **Index structure**: plain NumPy cosine (like ``retrieval.py``), not
    ``hnswlib``. A per-task repo index is bounded by one repo's chunk count
    — thousands, not the millions ``hnswlib``'s approximate search exists
    for. Brute-force cosine over a few thousand rows of a few dozen
    dimensions is microseconds; approximate search would trade recall for
    scale that does not exist at this problem size. Reconsider only if this
    is ever pointed at a monorepo with 100k+ chunks.

Chunking uses Python's ``ast`` module (v1 scope: ``.py`` files only, matching
SWE-bench's primarily-Python task shape — see ``uchi/code_engine.py`` for the
existing ``ast.parse`` usage pattern this follows). A file that fails to
parse (invalid syntax, non-UTF-8, vendored Python 2, etc.) is skipped and
recorded in ``skipped_files`` rather than aborting the whole index — this is
best-effort corpus ingestion, the same discipline ``retrieval.py.add()``
uses when a passage can't be embedded, not the "no fallback masking a real
failure" discipline that governs ``execution_sandbox.py``'s patch-apply/test
execution (there, a swallowed failure would corrupt a pass/fail verdict;
here, one unparseable file just means fewer retrievable chunks from it).

Import-graph edges are v1 scope: ``import x`` / ``from x import y`` per
file, resolved to files inside the repo. Symbol-level call-graph tracking
(which function calls which) is a nice-to-have deferred past v1 per the
0.5.0 plan — the edges here answer "what file is this file coupled to",
which is what makes "the bug is in file A, the fix is in file B it
imports" queryable via ``neighbor_chunks``.
"""
from __future__ import annotations

import ast
import keyword
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

_SKIP_DIRS = frozenset({
    "__pycache__", "node_modules", "build", "dist", ".git", ".tox",
    ".mypy_cache", ".pytest_cache", ".eggs", ".hg",
})

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_CODE_STOPWORDS = frozenset(keyword.kwlist) | {"self", "cls", "true", "false", "none"}


def _tokenize_code(text: str) -> list[str]:
    """Identifier-aware tokenizer: split snake_case/camelCase, lowercase,
    drop Python keywords + self/cls. Applied to raw chunk source, so
    comments and docstrings (often natural-language) contribute tokens too
    — the same regex catches both, deliberately.
    """
    tokens: list[str] = []
    for ident in _IDENT.findall(text):
        for part in ident.split("_"):
            if not part:
                continue
            for sub in _CAMEL_BOUNDARY.split(part):
                sub = sub.lower()
                if sub and len(sub) > 1 and sub not in _CODE_STOPWORDS:
                    tokens.append(sub)
    return tokens


@dataclass(frozen=True)
class CodeChunk:
    """One retrievable unit: a top-level function, a method, a class, or
    the leftover module-level source (imports, constants, ``if __name__``)
    of a single ``.py`` file.
    """
    chunk_id: int
    file: str          # repo-relative, posix-style
    qualname: str       # "<module>" | "foo" | "Bar" | "Bar.method"
    kind: str           # "module" | "function" | "method" | "class"
    start_line: int
    end_line: int
    source: str


# ── chunking ────────────────────────────────────────────────────────────────

def _node_start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        return min(node.lineno, min(d.lineno for d in decorators))
    return node.lineno


def _line_source(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[start - 1:end])


def _chunk_module(rel_path: str, source: str, tree: ast.Module) -> list[CodeChunk]:
    lines = source.splitlines()
    chunks: list[CodeChunk] = []
    covered = [False] * (len(lines) + 1)  # 1-indexed

    def mark(a: int, b: int) -> None:
        for ln in range(max(a, 1), min(b, len(lines)) + 1):
            covered[ln] = True

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = _node_start(node)
            chunks.append(CodeChunk(
                0, rel_path, node.name, "function", start, node.end_lineno,
                _line_source(lines, start, node.end_lineno),
            ))
            mark(start, node.end_lineno)
        elif isinstance(node, ast.ClassDef):
            start = _node_start(node)
            chunks.append(CodeChunk(
                0, rel_path, node.name, "class", start, node.end_lineno,
                _line_source(lines, start, node.end_lineno),
            ))
            mark(start, node.end_lineno)
            # Methods get their own chunk too — deliberate duplication with
            # the class chunk above, trading index size for two useful
            # retrieval granularities ("the whole class" vs "exactly this
            # method"). Nested functions/classes inside a method are left
            # embedded in that method's chunk (v1 scope).
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    m_start = _node_start(sub)
                    chunks.append(CodeChunk(
                        0, rel_path, f"{node.name}.{sub.name}", "method",
                        m_start, sub.end_lineno,
                        _line_source(lines, m_start, sub.end_lineno),
                    ))

    leftover = [ln for ln in range(1, len(lines) + 1)
                if not covered[ln] and lines[ln - 1].strip()]
    if leftover:
        chunks.append(CodeChunk(
            0, rel_path, "<module>", "module", leftover[0], leftover[-1],
            "\n".join(lines[ln - 1] for ln in leftover),
        ))
    return chunks


def _read_source(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def _iter_python_files(repo_root: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if name.endswith(".py"):
                found.append(Path(dirpath) / name)
    return sorted(found)


# ── import graph ─────────────────────────────────────────────────────────────

def _resolve_dotted(root: Path, module: str) -> Optional[str]:
    base = root.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return str(candidate.relative_to(root)).replace(os.sep, "/")
    return None


def _resolve_importfrom(root: Path, current_file: Path, node: ast.ImportFrom) -> set[str]:
    if node.level and node.level > 0:
        base_dir = current_file.parent
        for _ in range(node.level - 1):
            base_dir = base_dir.parent
    else:
        base_dir = root
    package_dir = base_dir.joinpath(*node.module.split(".")) if node.module else base_dir

    found: set[str] = set()
    # "from pkg.sub import name" — the target is pkg/sub.py itself.
    for candidate in (package_dir.with_suffix(".py"), package_dir / "__init__.py"):
        if candidate.is_file():
            found.add(str(candidate.relative_to(root)).replace(os.sep, "/"))
    # "from pkg import submodule" — submodule may itself be a file, not an
    # attribute of pkg/__init__.py; try each imported name as a file too.
    for alias in node.names:
        sub = package_dir / alias.name
        for candidate in (sub.with_suffix(".py"), sub / "__init__.py"):
            if candidate.is_file():
                found.add(str(candidate.relative_to(root)).replace(os.sep, "/"))
    return found


def _build_import_graph(
    root: Path, file_trees: dict[str, ast.Module]
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    py_files = set(file_trees)
    imports: dict[str, set[str]] = {f: set() for f in py_files}
    imported_by: dict[str, set[str]] = {f: set() for f in py_files}
    for rel, tree in file_trees.items():
        current_file = root / rel
        for node in ast.walk(tree):
            targets: set[str] = set()
            if isinstance(node, ast.Import):
                for alias in node.names:
                    t = _resolve_dotted(root, alias.name)
                    if t:
                        targets.add(t)
            elif isinstance(node, ast.ImportFrom):
                targets |= _resolve_importfrom(root, current_file, node)
            for t in targets:
                if t in py_files and t != rel:
                    imports[rel].add(t)
                    imported_by[t].add(rel)
    return imports, imported_by


# ── from-scratch skip-gram (SGNS) ────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


def _train_skipgram(
    stream: np.ndarray, vocab_size: int, dim: int, window: int,
    negatives: int, epochs: int, seed: int, lr: float = 0.05,
) -> np.ndarray:
    """From-scratch skip-gram with negative sampling (Mikolov et al. 2013),
    trained with hand-written NumPy gradient steps rather than an autograd
    framework — keeps the one from-scratch mechanism auditable end to end,
    and per-repo corpora (thousands, not billions, of tokens) train in
    seconds this way with no torch dependency in this module at all.

    Chunk boundaries are not preserved in the training stream (context
    windows may occasionally span two adjacent chunks) — the same
    simplification ``experiments/skipgram_probe.py`` makes when flattening
    per-sentence token lists into one stream; acceptable noise at this
    corpus scale.
    """
    rng = np.random.default_rng(seed)
    L = stream.shape[0]
    if vocab_size == 0 or L < 3:
        return np.zeros((max(vocab_size, 1), dim), dtype=np.float32)

    # Clamp window so tiny repos (few dozen tokens) still get a valid,
    # non-empty range of trainable center positions.
    window = max(1, min(window, (L - 1) // 2))
    W_in = (rng.uniform(-0.5, 0.5, size=(vocab_size, dim)) / dim).astype(np.float32)
    W_out = np.zeros((vocab_size, dim), dtype=np.float32)
    freq = np.bincount(stream, minlength=vocab_size).astype(np.float64) ** 0.75
    neg_p = freq / freq.sum()

    positions = np.arange(window, L - window)
    if positions.size == 0:
        return _l2_normalize_rows(W_in)
    batch = min(1024, positions.size)

    for _ in range(epochs):
        rng.shuffle(positions)
        for start in range(0, positions.size, batch):
            idx = positions[start:start + batch]
            centers = stream[idx]
            offsets = rng.integers(1, window + 1, size=idx.size) * rng.choice(
                np.array([-1, 1]), size=idx.size
            )
            contexts = stream[idx + offsets]
            negs = rng.choice(vocab_size, size=(idx.size, negatives), p=neg_p)

            vc = W_in[centers]
            vo = W_out[contexts]
            vn = W_out[negs]

            pos_score = _sigmoid((vc * vo).sum(-1))
            neg_score = _sigmoid((vn * vc[:, None, :]).sum(-1))

            grad_vc = (pos_score - 1.0)[:, None] * vo + (neg_score[:, :, None] * vn).sum(1)
            grad_vo = (pos_score - 1.0)[:, None] * vc
            grad_vn = neg_score[:, :, None] * vc[:, None, :]

            np.add.at(W_in, centers, -lr * grad_vc)
            np.add.at(W_out, contexts, -lr * grad_vo)
            np.add.at(W_out, negs.reshape(-1), -lr * grad_vn.reshape(-1, dim))

    return _l2_normalize_rows(W_in)


# ── index ─────────────────────────────────────────────────────────────────

class CodeIndex:
    """Repo-scale code chunk index: skip-gram cosine retrieval plus a
    deterministic code-aware rerank, over ``.py`` chunks extracted with
    ``ast``, with an import graph for cross-file navigation.
    """

    def __init__(self, dim: int = 64, seed: int = 0) -> None:
        self.dim = dim
        self.seed = seed
        self.chunks: list[CodeChunk] = []
        self.skipped_files: list[str] = []
        self.w2i: dict[str, int] = {}
        self.E: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._P: Optional[np.ndarray] = None
        self._imports: dict[str, set[str]] = {}
        self._imported_by: dict[str, set[str]] = {}

    @classmethod
    def build(
        cls, repo_root: str | os.PathLike, dim: int = 64, min_count: int = 1,
        window: int = 4, negatives: int = 5, epochs: int = 8, seed: int = 0,
    ) -> "CodeIndex":
        idx = cls(dim=dim, seed=seed)
        idx.index_repo(repo_root, min_count=min_count, window=window,
                        negatives=negatives, epochs=epochs)
        return idx

    def index_repo(
        self, repo_root: str | os.PathLike, min_count: int = 1,
        window: int = 4, negatives: int = 5, epochs: int = 8,
    ) -> None:
        """Chunk every ``.py`` file, build the import graph, then train
        embeddings and embed every chunk. Call once per task repo.
        """
        root = Path(repo_root).resolve()
        chunks: list[CodeChunk] = []
        file_trees: dict[str, ast.Module] = {}
        self.skipped_files = []

        for path in _iter_python_files(root):
            rel = str(path.relative_to(root)).replace(os.sep, "/")
            source = _read_source(path)
            if source is None:
                self.skipped_files.append(rel)
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                self.skipped_files.append(rel)
                continue
            file_trees[rel] = tree
            chunks.extend(_chunk_module(rel, source, tree))

        self.chunks = [replace(c, chunk_id=i) for i, c in enumerate(chunks)]
        self._imports, self._imported_by = _build_import_graph(root, file_trees)
        self._fit_embeddings(min_count=min_count, window=window,
                              negatives=negatives, epochs=epochs)

    def _fit_embeddings(self, min_count: int, window: int, negatives: int, epochs: int) -> None:
        token_lists = [_tokenize_code(c.source) for c in self.chunks]
        counts: dict[str, int] = {}
        for toks in token_lists:
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
        vocab = [w for w, c in counts.items() if c >= min_count]
        self.w2i = {w: i for i, w in enumerate(vocab)}

        stream = np.array(
            [self.w2i[t] for toks in token_lists for t in toks if t in self.w2i],
            dtype=np.int64,
        )
        self.E = _train_skipgram(stream, len(vocab), self.dim, window,
                                  negatives, epochs, self.seed)

        vecs = np.zeros((len(self.chunks), self.dim), dtype=np.float32)
        for i, toks in enumerate(token_lists):
            ids = [self.w2i[t] for t in toks if t in self.w2i]
            if not ids:
                continue
            v = self.E[ids].mean(0)
            n = np.linalg.norm(v)
            if n > 0:
                vecs[i] = v / n
        self._P = vecs

    def _vec(self, text: str) -> Optional[np.ndarray]:
        ids = [self.w2i[t] for t in _tokenize_code(text) if t in self.w2i]
        if not ids:
            return None
        v = self.E[ids].mean(0)
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 0 else None

    # ── retrieval ─────────────────────────────────────────────────────────

    def retrieve(
        self, query: str, k: int = 10, context_chunk_ids: Optional[Iterable[int]] = None,
        lex_weight: float = 0.5, graph_weight: float = 0.3,
    ) -> list[tuple[CodeChunk, float]]:
        """Return up to k (chunk, cosine) pairs, hybrid-reranked.

        Mirrors ``retrieval.py.retrieve``'s two-stage shape: semantic cosine
        picks the candidate pool, a deterministic rerank reorders it. The
        rerank here adds two code-specific terms in place of the
        cross-encoder the original plan called for: shared-identifier
        overlap (same spirit as ``retrieval.py``'s lexical overlap) and
        import-graph proximity to *context_chunk_ids* — chunks already
        known to matter (e.g. the chunk containing the failing test) — so a
        candidate in a file that context imports, or is imported by, is
        boosted. The returned score is the semantic cosine; ordering is
        hybrid.
        """
        if self._P is None or not self.chunks:
            return []
        qv = self._vec(query)
        if qv is None:
            return []
        sims = self._P @ qv
        pool = min(max(k * 5, k), len(self.chunks))
        cand = np.argpartition(-sims, pool - 1)[:pool]

        qtoks = set(_tokenize_code(query))
        neighbor_files: set[str] = set()
        if context_chunk_ids:
            for cid in context_chunk_ids:
                if 0 <= cid < len(self.chunks):
                    f = self.chunks[cid].file
                    neighbor_files |= self._imports.get(f, set())
                    neighbor_files |= self._imported_by.get(f, set())

        def hybrid(i: int) -> float:
            score = float(sims[i])
            chunk = self.chunks[i]
            if qtoks:
                ctoks = set(_tokenize_code(chunk.source))
                score += lex_weight * (len(qtoks & ctoks) / len(qtoks))
            if chunk.file in neighbor_files:
                score += graph_weight
            return score

        top = sorted(cand.tolist(), key=hybrid, reverse=True)[:k]
        return [(self.chunks[i], float(sims[i])) for i in top]

    # ── graph navigation ──────────────────────────────────────────────────

    def imports_of(self, file: str) -> set[str]:
        """Repo-relative files that *file* imports."""
        return set(self._imports.get(file, set()))

    def imported_by(self, file: str) -> set[str]:
        """Repo-relative files that import *file*."""
        return set(self._imported_by.get(file, set()))

    def neighbor_chunks(self, chunk_id: int, direction: str = "both") -> list[CodeChunk]:
        """Cross-file navigation: chunks belonging to files that *chunk_id*'s
        file imports, is imported by, or both. This is the mechanism behind
        "the bug is in file A, but the fix is in file B it imports".
        """
        if not (0 <= chunk_id < len(self.chunks)):
            raise IndexError(f"no chunk id {chunk_id}")
        if direction not in ("both", "imports", "imported_by"):
            raise ValueError(f"invalid direction: {direction!r}")
        file = self.chunks[chunk_id].file
        neighbor_files: set[str] = set()
        if direction in ("both", "imports"):
            neighbor_files |= self._imports.get(file, set())
        if direction in ("both", "imported_by"):
            neighbor_files |= self._imported_by.get(file, set())
        return [c for c in self.chunks if c.file in neighbor_files]

    def __len__(self) -> int:
        return len(self.chunks)
