"""tool_learning.py — Dynamic Tool Registration (0.4.0 Item 8).

Hardcoding tools limits what Uchi can do. This lets a user hand it a plain
Python file (``u.learn_tools("database_client.py")``) and have every
top-level function in it become a callable tool immediately, with no
changes to Uchi's own source. Reuses the same name -> callable pattern
already proven by ``tool_calling.ToolRegistry`` (Item 4) rather than
inventing a second registry mechanism.

Docstrings and signatures are ingested as knowledge (not just wired up for
execution) so the trie can discover and reason about what a tool does, not
only how to call it.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import uuid
from dataclasses import dataclass
from typing import List

from .tool_calling import ToolRegistry


@dataclass
class LearnedTool:
    name: str
    signature: str
    docstring: str
    source_path: str

    def as_knowledge(self) -> str:
        """Render as a plain-text fact suitable for ``Core.learn()``."""
        doc = f" — {self.docstring}" if self.docstring else ""
        return f"Tool available: {self.name}{self.signature}{doc}"


def _load_module(path: str):
    path = os.path.abspath(path)
    mod_name = f"_uchi_learned_tools_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _top_level_function_names(path: str) -> List[str]:
    """Parse *path* with ``ast`` (read-only) to find functions actually
    *defined* at module scope in this file — not names merely imported
    into its namespace, and not methods nested inside a class."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def learn_tools(path: str, registry: ToolRegistry) -> List[LearnedTool]:
    """Parse *path*, register every top-level function into *registry*,
    and return what was learned (name, signature, docstring) so the
    caller can also ingest it as knowledge."""
    names = _top_level_function_names(path)
    if not names:
        return []
    module = _load_module(path)
    learned = []
    for name in names:
        fn = getattr(module, name, None)
        if fn is None or not callable(fn):
            continue
        registry.register(name, fn)
        try:
            sig = str(inspect.signature(fn))
        except (TypeError, ValueError):
            sig = "(...)"
        learned.append(
            LearnedTool(name=name, signature=sig, docstring=inspect.getdoc(fn) or "", source_path=path)
        )
    return learned
