"""MetaUchi — the default facade.

``from uchi import Uchi`` returns this class. Today it wraps a single
``Core`` node and forwards every call untouched — a stub. Tool calling
(0.4.0 Item 4), the Python scratchpad (Item 3), and multi-node delegation
(Item 9) attach to this facade in later items without changing this public
surface: callers of ``Uchi()`` see no difference between now and once those
land.

Use ``uchi.Core`` directly for the raw, un-orchestrated single-instance
engine with nothing wrapping it.
"""
from __future__ import annotations

from typing import Any, Optional

from .simple import Core


class MetaUchi:
    """Default facade. Wraps a single ``Core`` node.

    Every attribute access not defined here forwards to the wrapped
    ``Core`` instance, so today ``MetaUchi`` behaves identically to
    using ``Core`` directly.
    """

    def __init__(self, brain_path: Optional[str] = None, web_search: bool = False) -> None:
        self.core = Core(brain_path=brain_path, web_search=web_search)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.core, name)

    def ingest(self, path: str, col: Optional[str] = None) -> "MetaUchi":
        """Forward to ``Core.ingest`` but return ``self`` so chaining works
        on the facade rather than unwrapping to the inner ``Core``."""
        self.core.ingest(path, col=col)
        return self
