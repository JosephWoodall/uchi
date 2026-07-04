"""
brain_fetch.py — locate or fetch the premade general-knowledge brain
(uchi/data/embeddings.pt, ~221MB).

That file cannot ship inside the PyPI wheel: PyPI hard-caps uploads at 100MB
per file, and embeddings.pt alone is over 2x that (confirmed — a real publish
attempt failed with "File too large. Limit for project 'uchi-python' is 100 MB").
`git clone` + `git lfs pull` still gets it directly (it's tracked in the repo),
but a plain `pip install` needs another way to end up with a working premade
brain — this module fetches it once, on first use, and caches it.

Resolution order:
    1. Already present next to the package (git clone + LFS pull case) — used
       as-is, no network.
    2. Already present in the user cache (~/.uchi/data/embeddings.pt) — used
       as-is, no network.
    3. Download from GitHub (the file is LFS-tracked in this same repo, so the
       LFS media endpoint serves the real bytes over plain HTTPS) into the
       cache, then use it.
    4. Any failure (offline, blocked network, HTTP error) — return None. The
       caller degrades to an empty index (learn()/ask() still work, just with
       no pre-loaded general knowledge) rather than raising.
"""
from __future__ import annotations

import os
from typing import Optional

_REPO_OWNER = "JosephWoodall"
_REPO_NAME = "uchi"
_REF = "main"  # branch/tag to fetch from
_LFS_MEDIA_URL = (
    f"https://media.githubusercontent.com/media/{_REPO_OWNER}/{_REPO_NAME}/{_REF}/"
    "uchi/data/embeddings.pt"
)


def _cache_path() -> str:
    return os.path.join(os.path.expanduser("~/.uchi/data"), "embeddings.pt")


def _bundled_path() -> str:
    return os.path.join(os.path.dirname(__file__), "data", "embeddings.pt")


def get_embeddings_path(download: bool = True, quiet: bool = False) -> Optional[str]:
    """Return a local path to embeddings.pt, fetching it once if necessary.

    Returns None (never raises) if it isn't available and can't be fetched —
    callers should treat that as "no premade brain", not a fatal error.
    """
    bundled = _bundled_path()
    if os.path.exists(bundled):
        return bundled

    cached = _cache_path()
    if os.path.exists(cached):
        return cached

    if not download or os.environ.get("UCHI_NO_BRAIN_DOWNLOAD"):
        return None

    try:
        import urllib.request

        os.makedirs(os.path.dirname(cached), exist_ok=True)
        tmp_path = cached + ".part"
        if not quiet:
            print("[*] Fetching the premade general-knowledge brain (~221MB, one-time)...")
        urllib.request.urlretrieve(_LFS_MEDIA_URL, tmp_path)
        os.replace(tmp_path, cached)
        if not quiet:
            print(f"[+] Saved to {cached} — cached for future runs.")
        return cached
    except Exception as e:
        if not quiet:
            print(f"[-] Could not fetch the premade brain ({type(e).__name__}: {e}). "
                  f"Continuing with an empty knowledge index — learn() still works.")
        # Clean up a partial download so a later retry doesn't see a corrupt file.
        try:
            if os.path.exists(cached + ".part"):
                os.remove(cached + ".part")
        except OSError:
            pass
        return None
