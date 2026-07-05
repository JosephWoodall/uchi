"""data_silo.py — Enterprise Data Silos (0.4.0 Item 16.5).

Configures a ``Core`` instance so it can ingest ``./finance`` but is
logically barred from ``./hr`` — an application-layer allow/deny list
enforced on ``ingest()``, checked before any file is read.

Scoped honestly: this is *not* OS-level permission enforcement (chmod,
containers, a separate OS user) — that's a deployment concern outside
what a Python library can itself grant or restrict. What it gives you is
a configurable, auditable boundary at the one place all external
ingestion already funnels through, useful as a guardrail against
accidental cross-silo ingestion in a single process, not as a substitute
for real OS/container-level isolation in an actual multi-tenant deployment.
"""
from __future__ import annotations

import os
from typing import List, Optional


class DataSiloViolation(Exception):
    """Raised when a path is denied, or (with an allow-list configured)
    not explicitly allowed."""


class DataSilo:
    """Application-layer allow/deny list for ``Core.ingest()`` paths."""

    def __init__(
        self,
        allowed_paths: Optional[List[str]] = None,
        denied_paths: Optional[List[str]] = None,
    ) -> None:
        self.allowed = [os.path.realpath(p) for p in (allowed_paths or [])]
        self.denied = [os.path.realpath(p) for p in (denied_paths or [])]

    @staticmethod
    def _under(path: str, root: str) -> bool:
        return path == root or path.startswith(root + os.sep)

    def check(self, path: str) -> None:
        """Raise ``DataSiloViolation`` if *path* isn't allowed."""
        real = os.path.realpath(path)

        for root in self.denied:
            if self._under(real, root):
                raise DataSiloViolation(f"path {path!r} is inside a denied silo: {root!r}")

        if self.allowed and not any(self._under(real, root) for root in self.allowed):
            raise DataSiloViolation(
                f"path {path!r} is outside all allowed silos: {self.allowed!r}"
            )
