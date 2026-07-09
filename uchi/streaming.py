"""streaming.py — Observable Monologue / Thought-vs-Speech streaming (0.4.0 Item 16.9).

``Core.ask()`` already reports progress via a ``callback(stage, message)``
mechanism used throughout the pipeline (swarm decomposition, oracle
verification, empirical synthesis, etc.) — "thinking"/"reinforce"/"prune"
events. This module turns that into a real-time stream: ``ask()`` runs in
a background thread, each callback event is pushed onto a queue as it
actually happens, and the generator yields it immediately as a
``{"type": "thought", ...}`` event, followed by one final
``{"type": "speech", "content": ...}`` event once ``ask()`` returns.

This is genuine Thought-vs-Speech separation — not a line drawn after the
fact — and genuine streaming: events are yielded as the pipeline produces
them, not chunks of an already-finished string.
"""
from __future__ import annotations

import queue
import threading
from typing import Any, Dict, Generator

_SENTINEL = object()


def stream_ask(core: Any, question: str, **data: Any) -> Generator[Dict[str, Any], None, None]:
    """Yield ``{"type": "thought", "stage": ..., "message": ...}`` events
    as ``core.ask(question, **data)`` runs, then a final
    ``{"type": "speech", "content": ...}`` once it returns.
    """
    q: "queue.Queue" = queue.Queue()
    result: Dict[str, Any] = {}

    def callback(stage: str, message: str) -> None:
        q.put({"type": "thought", "stage": stage, "message": message})

    def run() -> None:
        try:
            result["answer"] = core.ask(question, callback=callback, **data)
        except Exception as e:
            result["error"] = str(e)
        finally:
            q.put(_SENTINEL)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    while True:
        item = q.get()
        if item is _SENTINEL:
            break
        yield item

    thread.join()
    if "error" in result:
        yield {"type": "speech", "content": "", "error": result["error"]}
    else:
        yield {"type": "speech", "content": result.get("answer", "")}
