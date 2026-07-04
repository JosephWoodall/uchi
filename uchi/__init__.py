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
"""

__version__ = "0.3.0"

from .simple import Uchi

__all__ = ["Uchi"]
