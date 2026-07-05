"""workspace.py — filesystem path guard for autonomous tool calls (0.4.0 Item 2).

Resolves every path a tool call touches against a fixed sandbox root and
refuses anything that would escape it. This is a guard rail, not a new
subsystem: `uchi/code_engine.py`'s `REPLOracle` already writes to
`tempfile.NamedTemporaryFile` and runs `subprocess.run` safely without a
dedicated chroot — this module gives that same pattern a fixed root and a
single place to enforce "stay inside the sandbox" for every tool call.

Follows the same convention as `uchi/telemetry.py`'s `.uchi/telemetry`:
a dot-directory relative to the current working directory, not a path
inside the installed package (which may not be writable).
"""
from __future__ import annotations

import os

DEFAULT_ROOT = os.path.join(".uchi", "workspace")


class WorkspaceViolation(Exception):
    """Raised when a tool call tries to touch a path outside the workspace root."""


def _ensure_root(root: str) -> str:
    os.makedirs(root, exist_ok=True)
    return os.path.realpath(root)


def resolve_path(path: str, root: str = DEFAULT_ROOT) -> str:
    """Resolve *path* against *root*, raising WorkspaceViolation if it escapes.

    Relative paths are joined to *root*. Absolute paths are still required to
    resolve inside *root* — an absolute path pointing elsewhere is rejected
    rather than silently honoured. Resolution goes through ``os.path.realpath``,
    so a symlink inside the sandbox that points outside it is caught too.
    """
    real_root = _ensure_root(root)
    candidate = path if os.path.isabs(path) else os.path.join(real_root, path)
    resolved = os.path.realpath(candidate)
    if resolved != real_root and not resolved.startswith(real_root + os.sep):
        raise WorkspaceViolation(f"path escapes workspace root {real_root!r}: {path!r}")
    return resolved


def read_file(path: str, root: str = DEFAULT_ROOT) -> str:
    """Read a text file from within the workspace sandbox."""
    resolved = resolve_path(path, root=root)
    with open(resolved, encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def write_file(path: str, content: str, root: str = DEFAULT_ROOT) -> str:
    """Write a text file within the workspace sandbox. Returns the resolved path."""
    resolved = resolve_path(path, root=root)
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as fh:
        fh.write(content)
    return resolved


def list_files(path: str = ".", root: str = DEFAULT_ROOT) -> list[str]:
    """List entries under *path* within the workspace sandbox."""
    resolved = resolve_path(path, root=root)
    return sorted(os.listdir(resolved))


def delete_file(path: str, root: str = DEFAULT_ROOT) -> None:
    """Delete a file within the workspace sandbox."""
    resolved = resolve_path(path, root=root)
    os.remove(resolved)
