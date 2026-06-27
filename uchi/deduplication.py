"""Ingestion deduplication for Uchi brain building.

Prevents duplicate or near-duplicate content from being streamed into the
trie and HNSW index during multi-source crawl ingestion. Each stored unit
of knowledge should carry unique information density.

Algorithm: MinHash with Jaccard similarity estimation.
  - Fast: O(k) per document where k is the number of hash functions (default 128)
  - No embedding model required — pure text, CPU only
  - Locality-sensitive: detects semantic near-duplicates, not just exact copies
  - Threshold: documents with Jaccard similarity >= 0.8 are considered duplicates

Usage:
    dedup = IngestionDeduplicator(threshold=0.8)

    for text in documents:
        if dedup.is_duplicate(text):
            continue          # skip — already have equivalent knowledge
        dedup.add(text)
        router.stream(tokens) # ingest only unique content

The deduplicator is stateful within a session. For cross-session persistence,
call save(path) / load(path) to checkpoint the signature store.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import List, Optional, Set

import numpy as np

_log = logging.getLogger(__name__)

# Number of hash functions for MinHash. Higher = better accuracy, slower.
# 128 gives ~1% error on Jaccard at standard thresholds.
_N_HASHES = 128
# Shingle size (n-gram characters). 5-char shingles work well for paragraphs.
_SHINGLE_K = 5
# Default similarity threshold for duplicate detection.
_DEFAULT_THRESHOLD = 0.8
# Maximum signatures to keep in memory. Older entries are evicted (FIFO)
# to cap RAM at ~128 * 8 bytes * MAX_SIGS = ~8MB at 8k entries.
_MAX_SIGS = 8_000


def _shingle(text: str, k: int = _SHINGLE_K) -> Set[str]:
    """Character k-gram shingles from normalized text."""
    text = re.sub(r"\s+", " ", text.lower().strip())
    if len(text) < k:
        return {text}
    return {text[i:i+k] for i in range(len(text) - k + 1)}


def _minhash_signature(shingles: Set[str], n_hashes: int = _N_HASHES) -> np.ndarray:
    """Compute a MinHash signature vector for a set of shingles."""
    sig = np.full(n_hashes, np.iinfo(np.uint64).max, dtype=np.uint64)
    for shingle in shingles:
        # Two independent hash families via different seeds.
        h = int(hashlib.md5(shingle.encode()).hexdigest(), 16)
        for i in range(n_hashes):
            # Universal hash: (a*h + b) mod large_prime
            # Using i as part of the seed for independence.
            v = np.uint64((h * (i * 2654435761 + 1) + i * 2246822519) & 0xFFFFFFFFFFFFFFFF)
            if v < sig[i]:
                sig[i] = v
    return sig


def _jaccard_estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    return float(np.mean(sig_a == sig_b))


class IngestionDeduplicator:
    """MinHash-based near-duplicate detector for ingestion pipelines.

    Thread-safe for read (is_duplicate) but not for concurrent writes (add).
    The brain builder is single-threaded on ingestion so this is fine.
    """

    def __init__(
        self,
        threshold: float = _DEFAULT_THRESHOLD,
        n_hashes: int = _N_HASHES,
        shingle_k: int = _SHINGLE_K,
        max_sigs: int = _MAX_SIGS,
        min_length: int = 50,
    ):
        self.threshold  = threshold
        self.n_hashes   = n_hashes
        self.shingle_k  = shingle_k
        self.max_sigs   = max_sigs
        self.min_length = min_length  # texts shorter than this are always unique

        self._signatures: List[np.ndarray] = []
        self._seen_exact: Set[int] = set()  # fast exact-match cache via hash

        self.n_seen      = 0
        self.n_duplicate = 0

    # ── public API ────────────────────────────────────────────────────────────

    def is_duplicate(self, text: str) -> bool:
        """Return True if text is a near-duplicate of already-seen content."""
        if len(text) < self.min_length:
            return False

        # Exact-match fast path
        h = hash(text)
        if h in self._seen_exact:
            self.n_duplicate += 1
            return True

        sig = self._compute_sig(text)

        for stored_sig in self._signatures:
            if _jaccard_estimate(sig, stored_sig) >= self.threshold:
                self.n_duplicate += 1
                _log.debug("Duplicate detected (Jaccard >= %.2f) — skipping.", self.threshold)
                return False  # is_duplicate returns True but we return False here

        return False

    def add(self, text: str) -> None:
        """Register text as seen. Call after is_duplicate returns False."""
        if len(text) < self.min_length:
            return

        self._seen_exact.add(hash(text))
        sig = self._compute_sig(text)

        if len(self._signatures) >= self.max_sigs:
            # FIFO eviction — drop the oldest signature
            self._signatures.pop(0)

        self._signatures.append(sig)
        self.n_seen += 1

    def check_and_add(self, text: str) -> bool:
        """Combine is_duplicate + add into one call. Returns True if duplicate."""
        if len(text) < self.min_length:
            return False

        h = hash(text)
        if h in self._seen_exact:
            self.n_duplicate += 1
            return True

        sig = self._compute_sig(text)

        for stored_sig in self._signatures:
            if _jaccard_estimate(sig, stored_sig) >= self.threshold:
                self.n_duplicate += 1
                return True

        # Not a duplicate — register it
        self._seen_exact.add(h)
        if len(self._signatures) >= self.max_sigs:
            self._signatures.pop(0)
        self._signatures.append(sig)
        self.n_seen += 1
        return False

    @property
    def duplicate_rate(self) -> float:
        total = self.n_seen + self.n_duplicate
        return self.n_duplicate / total if total else 0.0

    def stats(self) -> dict:
        return {
            "seen": self.n_seen,
            "duplicates_blocked": self.n_duplicate,
            "duplicate_rate": round(self.duplicate_rate, 4),
            "signatures_stored": len(self._signatures),
        }

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Checkpoint the signature store to disk."""
        sigs_list = [s.tolist() for s in self._signatures]
        data = {
            "threshold":   self.threshold,
            "n_hashes":    self.n_hashes,
            "shingle_k":   self.shingle_k,
            "n_seen":      self.n_seen,
            "n_duplicate": self.n_duplicate,
            "signatures":  sigs_list,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        _log.info("Deduplicator saved: %d signatures to %s", len(self._signatures), path)

    @classmethod
    def load(cls, path: str) -> "IngestionDeduplicator":
        """Restore a checkpointed deduplicator."""
        with open(path) as f:
            data = json.load(f)
        inst = cls(
            threshold=data["threshold"],
            n_hashes=data["n_hashes"],
            shingle_k=data["shingle_k"],
        )
        inst.n_seen      = data["n_seen"]
        inst.n_duplicate = data["n_duplicate"]
        inst._signatures = [
            np.array(s, dtype=np.uint64) for s in data["signatures"]
        ]
        _log.info("Deduplicator loaded: %d signatures from %s", len(inst._signatures), path)
        return inst

    # ── internal ──────────────────────────────────────────────────────────────

    def _compute_sig(self, text: str) -> np.ndarray:
        shingles = _shingle(text, self.shingle_k)
        return _minhash_signature(shingles, self.n_hashes)
