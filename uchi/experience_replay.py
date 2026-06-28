"""
experience_replay.py
====================
SQLite-backed Prioritized Experience Replay Buffer for Uchi's offline dream loop.

Stores (query, positive, hard_negative, priority) tuples so the dreaming
daemon can re-visit the *hardest* memories — those where the SSM loss is
still high — rather than uniformly cycling through the seed concept pool.

Why SQLite (not RAM):
  - Survives Ctrl-C / restarts without losing experiences.
  - Thread-safe WAL mode lets inference writers coexist with the dream reader.
  - Zero-dependency: ships with Python's stdlib.

Priority semantics:
  - Higher priority → sampled more often.
  - Initial priority = provided loss value (or 1.0 default).
  - After each dream step the caller calls .update_priority(id, new_loss);
    if new_loss drops to zero the row's priority floors at a small epsilon
    so the daemon stops wasting compute on solved memories.
  - Sampling uses a weighted-random draw proportional to priority.

Usage
-----
    buf = ExperienceReplayBuffer("replay.db")
    buf.push(query_tokens, pos_tokens, neg_tokens, priority=2.5)
    batch = buf.sample(batch_size=8)
    for row in batch:
        loss = train_on(row)
        buf.update_priority(row["id"], loss)
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
import threading
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)

_PRIORITY_FLOOR = 1e-3   # never let priority drop to zero (still sampleable)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiences (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query       TEXT    NOT NULL,
    positive    TEXT    NOT NULL,
    negative    TEXT,
    priority    REAL    NOT NULL DEFAULT 1.0,
    created_at  INTEGER NOT NULL DEFAULT (strftime('%s','now')),
    updated_at  INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_priority ON experiences(priority DESC);
"""


class ExperienceReplayBuffer:
    """
    Thread-safe, disk-persistent prioritized replay buffer.

    All writes are async (fire-and-forget thread) so the hot inference path
    never blocks on disk I/O.  Reads (sample) are synchronous and tiny.
    """

    def __init__(self, db_path: str = "replay.db") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    # ── public API ────────────────────────────────────────────────────────────

    def push(
        self,
        query_tokens: List[str],
        positive_tokens: List[str],
        hard_negative_tokens: Optional[List[str]] = None,
        priority: float = 1.0,
    ) -> None:
        """
        Persist a new experience asynchronously.

        Args:
            query_tokens:        Tokenised query (includes <|user|> wrapper if present).
            positive_tokens:     Oracle-best response tokens.
            hard_negative_tokens: The candidate the SSM currently (wrongly) ranks closest;
                                  None if hard negative mining yielded nothing.
            priority:            Initial loss magnitude — higher = harder = sampled sooner.
        """
        q   = json.dumps(query_tokens)
        pos = json.dumps(positive_tokens)
        neg = json.dumps(hard_negative_tokens) if hard_negative_tokens is not None else None
        pri = max(float(priority), _PRIORITY_FLOOR)

        t = threading.Thread(target=self._write, args=(q, pos, neg, pri), daemon=True)
        t.start()

    def sample(self, batch_size: int = 8) -> List[Dict[str, Any]]:
        """
        Return up to *batch_size* experiences, weighted by priority.

        Returns a list of dicts with keys:
            id, query, positive, negative, priority
        where query/positive/negative are already decoded token lists.

        Returns [] when the buffer is empty.
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT id, query, positive, negative, priority FROM experiences"
                ).fetchall()
            finally:
                conn.close()

        if not rows:
            return []

        # Weighted random sample without replacement
        ids, weights = zip(*[(r[0], max(r[4], _PRIORITY_FLOOR)) for r in rows])
        total = sum(weights)
        probs = [w / total for w in weights]

        k = min(batch_size, len(rows))
        chosen_indices = random.choices(range(len(rows)), weights=probs, k=k)

        results = []
        seen = set()
        for idx in chosen_indices:
            row_id = ids[idx]
            if row_id in seen:
                continue
            seen.add(row_id)
            r = rows[idx]
            results.append({
                "id":       r[0],
                "query":    json.loads(r[1]),
                "positive": json.loads(r[2]),
                "negative": json.loads(r[3]) if r[3] is not None else None,
                "priority": r[4],
            })
        return results

    def update_priority(self, memory_id: int, new_priority: float) -> None:
        """Update the priority of a memory after training on it."""
        pri = max(float(new_priority), _PRIORITY_FLOOR)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE experiences SET priority=?, updated_at=strftime('%s','now') WHERE id=?",
                    (pri, memory_id),
                )
                conn.commit()
            finally:
                conn.close()

    def __len__(self) -> int:
        with self._lock:
            conn = self._connect()
            try:
                return conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
            finally:
                conn.close()

    # ── private helpers ───────────────────────────────────────────────────────

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _write(self, query: str, positive: str, negative: Optional[str], priority: float) -> None:
        try:
            with self._lock:
                conn = self._connect()
                try:
                    conn.execute(
                        "INSERT INTO experiences (query, positive, negative, priority) VALUES (?,?,?,?)",
                        (query, positive, negative, priority),
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception as exc:
            _log.debug("ExperienceReplayBuffer._write failed: %s", exc)
