"""
telemetry.py
============
Lightweight structured diagnostic telemetry for Uchi engine internals.

Thread-safe accumulator — every hook is fire-and-forget (errors silently
swallowed so telemetry never impacts engine correctness or latency).

Output:
  .uchi/telemetry/latest.json   — always overwritten with current session
  .uchi/telemetry/history.db    — append-only SQLite for trend analysis

Consumers:
  scripts/cognitive_debugger.py --telemetry
  Any agent that reads .uchi/telemetry/latest.json directly
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any

_TELEMETRY_DIR = os.path.join(".uchi", "telemetry")
_LOCK          = threading.Lock()
_SESSION: dict = {}


# ── write API ─────────────────────────────────────────────────────────────────

def record(section: str, key: str, value: Any) -> None:
    """Set _SESSION[section][key] = value."""
    try:
        with _LOCK:
            _SESSION.setdefault(section, {})[key] = value
    except Exception:
        pass


def append(section: str, key: str, item: Any) -> None:
    """Append *item* to the list at _SESSION[section][key]."""
    try:
        with _LOCK:
            _SESSION.setdefault(section, {}).setdefault(key, []).append(item)
    except Exception:
        pass


def increment(section: str, key: str, delta: int = 1) -> None:
    """Add *delta* to the integer counter at _SESSION[section][key]."""
    try:
        with _LOCK:
            bucket = _SESSION.setdefault(section, {})
            bucket[key] = bucket.get(key, 0) + delta
    except Exception:
        pass


# ── lifecycle ─────────────────────────────────────────────────────────────────

def reset() -> None:
    """Clear the current session (call at start of each query)."""
    try:
        with _LOCK:
            _SESSION.clear()
            _SESSION["_ts"] = time.time()
    except Exception:
        pass


def flush(telemetry_dir: str = _TELEMETRY_DIR) -> str:
    """
    Write current session to JSON + SQLite.  Returns the JSON path, or '' on failure.
    Safe to call from any thread at any time.
    """
    try:
        os.makedirs(telemetry_dir, exist_ok=True)
        with _LOCK:
            snapshot = dict(_SESSION)

        json_path = os.path.join(telemetry_dir, "latest.json")
        with open(json_path, "w") as fh:
            json.dump(snapshot, fh, indent=2, default=str)

        db_path = os.path.join(telemetry_dir, "history.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id   INTEGER PRIMARY KEY AUTOINCREMENT,
                ts   REAL,
                data TEXT
            )
        """)
        conn.execute(
            "INSERT INTO sessions (ts, data) VALUES (?,?)",
            (snapshot.get("_ts", time.time()), json.dumps(snapshot, default=str)),
        )
        conn.commit()
        conn.close()
        return json_path
    except Exception:
        return ""


def snapshot() -> dict:
    """Return a deep-copy of the current session (safe to mutate)."""
    try:
        with _LOCK:
            return json.loads(json.dumps(_SESSION, default=str))
    except Exception:
        return {}


# ── read API (for cognitive_debugger) ────────────────────────────────────────

def load_latest(telemetry_dir: str = _TELEMETRY_DIR) -> dict:
    """Read latest.json from disk.  Returns {} if not yet written."""
    try:
        path = os.path.join(telemetry_dir, "latest.json")
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def load_history(n: int = 20, telemetry_dir: str = _TELEMETRY_DIR) -> list[dict]:
    """Return the most recent *n* sessions from SQLite, newest first."""
    try:
        db_path = os.path.join(telemetry_dir, "history.db")
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT ts, data FROM sessions ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        conn.close()
        return [json.loads(row[1]) for row in rows]
    except Exception:
        return []
