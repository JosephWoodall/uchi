"""sandbox_isolation.py — 0.5.0 Item 5 Stage 2: container/VM-level
isolation for running model-generated code, on top of Stage 1's bare
subprocess execution (`execution_sandbox.py`).

`execution_sandbox.py`'s own docstring is explicit that Stage 1
(subprocess + timeout + explicit test-id list) is "only safe at the
volume and trust level this stage is scoped for" — fine for the curated,
low-volume evaluation this release has run so far, not safe once Item 6's
self-play starts running many model-generated patches' test suites.

Uses `bwrap` (bubblewrap) — the only sandboxing primitive actually present
in this environment (checked directly, not assumed: no `docker`, `runsc`
(gVisor), `firejail`, or `nsjail` on this host; `bwrap` is real and
unprivileged user namespaces work here). Bubblewrap is the same mechanism
Flatpak uses for unprivileged application sandboxing — mount/PID/network
namespaces, no root or daemon required per invocation.

Validated by hand before writing this: a real bwrap-wrapped `pytest`
invocation against a throwaway repo correctly reports a passing test as
passing and a failing test's real assertion text + exit code 1 unchanged,
and a test that tries to open a real TCP socket confirms the connection
is genuinely refused under `--unshare-net` (not a no-op wrapper).

Graceful degradation, same contract as `proprioception.py`'s
`load()`/`FluxProposer.load()`: if `bwrap` isn't installed, `run_isolated`
transparently falls back to a plain `subprocess.run` rather than raising —
isolation is a hardening layer, not a hard new dependency this repo
requires everywhere it's imported.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

_BWRAP_PATH = shutil.which("bwrap")


def bwrap_available() -> bool:
    return _BWRAP_PATH is not None


def _venv_root() -> str:
    """Root of the running interpreter's venv (two dirs up from
    `sys.executable`, e.g. `.venv/bin/python` -> `.venv`). Deliberately
    NOT resolved through symlinks -- `sys.executable` already reports the
    venv's own bin path as invoked, and resolving it would follow the
    interpreter binary out to the system Python's location instead,
    missing the venv's real `lib/pythonX.Y/site-packages` directory that
    isn't a symlink."""
    return str(Path(sys.executable).parent.parent)


def _bwrap_cmd(cmd: list[str], cwd: Path, env: dict | None) -> list[str]:
    # bwrap resolves bind-source paths against its own process, not the
    # caller's -- a relative *cwd* (e.g. DEFAULT_ROOT-derived paths, which
    # are relative by construction) silently resolves to the wrong,
    # nonsensical nested path. Found live: a relative cwd made bwrap
    # report "Can't find source path <the same relative string>" even
    # though the directory genuinely existed from the caller's cwd.
    cwd = Path(cwd).resolve()
    venv_root = _venv_root()
    wrapped = [
        _BWRAP_PATH,
        "--unshare-net", "--unshare-pid", "--die-with-parent",
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/sbin", "/sbin",
        "--ro-bind", venv_root, venv_root,
        "--proc", "/proc",
        "--dev", "/dev",
        # --tmpfs /tmp BEFORE the --bind for *cwd*: bwrap applies mounts
        # in argument order, so a later, more specific bind shadows an
        # earlier, broader one at the same path -- not the reverse. Found
        # live: pytest's own `tmp_path` fixture (and plenty of real
        # callers) put *cwd* under the host's /tmp; binding *cwd* before
        # mounting the empty tmpfs let the tmpfs mount clobber it
        # afterward ("Can't chdir to <cwd>: No such file or directory"),
        # even though the directory genuinely existed.
        "--tmpfs", "/tmp",
        "--bind", str(cwd), str(cwd),
        "--chdir", str(cwd),
    ]
    for key, value in (env or {}).items():
        wrapped += ["--setenv", key, str(value)]
    wrapped.append("--")
    wrapped += cmd
    return wrapped


def run_isolated(
    cmd: list[str], cwd: Path, timeout: float, env: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run *cmd* in *cwd*, isolated (network + PID namespace unshared,
    filesystem restricted to `/usr` + the running venv, read-only, plus
    *cwd* itself, read-write) if `bwrap` is available, else a plain
    `subprocess.run` -- same interface either way, so callers don't need
    to branch on availability themselves.

    *env* is merged additively via `--setenv` when isolated -- `bwrap`
    inherits the invoking process's environment by default (no
    `--clearenv` here, same as validated by hand), so `--setenv` only
    needs to add/override the specific keys a caller cares about
    (e.g. `PYTHONDONTWRITEBYTECODE`), not restate the whole environment.
    When not isolated, *env* is merged onto `os.environ` via
    `subprocess.run`'s own *env*, matching existing non-isolated callers'
    expectations.
    """
    if bwrap_available():
        full_cmd = _bwrap_cmd(cmd, cwd, env)
        return subprocess.run(full_cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)

    run_env = dict(os.environ, **(env or {}))
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=run_env)
