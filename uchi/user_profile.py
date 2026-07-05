"""user_profile.py — Infinite Session Memory (0.4.0 Item 16.4).

A hidden ``.uchi/user_profile.md`` file that accumulates user preferences
("I prefer Python 3.10") as plain text. Auto-ingested into the knowledge
index at the start of every ``Core`` session, this gives cross-session
memory without a vector database — the preference is just another
grounded fact the trie can retrieve, learned once and available forever.

Follows the same dot-directory convention already used by
``uchi/telemetry.py`` (``.uchi/telemetry``) and ``uchi/workspace.py``
(``.uchi/workspace``).
"""
from __future__ import annotations

import os
import time

DEFAULT_PATH = os.path.join(".uchi", "user_profile.md")


def remember_preference(text: str, path: str = DEFAULT_PATH) -> None:
    """Append *text* to the user profile file, creating it if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    stamp = time.strftime("%Y-%m-%d")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"- ({stamp}) {text}\n")


def load_profile(path: str = DEFAULT_PATH) -> str:
    """Read the current profile content, or ``""`` if none exists yet."""
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        return fh.read()
