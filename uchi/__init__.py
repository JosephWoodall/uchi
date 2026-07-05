"""
uchi
===============
A grounded, no-LLM assistant that verifies factual answers and abstains rather than confabulate.

Quick start
-----------
    from uchi import Uchi

    u = Uchi()
    u.learn("Q3 revenue was $4.2M, up 23% YoY.")
    print(u.ask("What was Q3 revenue growth?"))

``Uchi`` is the ``MetaUchi`` orchestrator facade. Import ``Core`` instead for
the raw, un-orchestrated single-instance engine with nothing wrapping it.
"""

__version__ = "0.3.0"

from .meta import MetaUchi as Uchi
from .simple import Core

__all__ = ["Uchi", "Core"]
