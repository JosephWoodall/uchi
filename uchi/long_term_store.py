"""
LongTermStore
=============
Persistent cross-sequence memory for the Universal Sequence Predictor.

Solves
------
  Problem 3 — Cold start         : warm prior available before token 1
  Problem 5 — Cross-seq memory   : patterns persist and strengthen across runs
  Problem 8 — Zero mass fallback : richer second layer before unigram floor

Consequence reasoning
---------------------
Stores P(t+n | ctx) for configurable n alongside the standard P(t+1 | ctx).
After replay, you can query what tends to happen 2 or 3 steps downstream
from any context the store has seen — useful for planning and lookahead.

Observability
-------------
Every replay call returns a stats dict and appends to run_history().
Watch accuracy climb across runs as the store warms up.

Persistence
-----------
Serialised with pickle (handles any token type) then gzip-compressed.
Portable: a single .lts file; load it anywhere with LongTermStore(path=...).
"""

import gzip
import os
import pickle
from typing import Any, Optional


class LongTermStore:
    """
    Persistent, slowly-updating trie that accumulates evidence across sequences.

    Parameters
    ----------
    path : str | None
        File path for persistence.  If the file exists it is loaded
        automatically on construction.  Saved after every replay().
    lr : float
        Learning rate for replay updates (default 0.01 — much slower than
        the short-term predictor's 0.08).
    replay_min_cred_ratio : float
        Only replay nodes whose credibility is at least this fraction of
        cred_max.  Filters out low-confidence short-term patterns.
    consequence_depth : int
        How many steps ahead to store consequence distributions.
        0 = disabled; 2 = store P(t+2|ctx) and P(t+3|ctx).
    """

    def __init__(
        self,
        path: Optional[str] = None,
        lr: float = 0.01,
        replay_min_cred_ratio: float = 0.6,
        consequence_depth: int = 2,
    ):
        self.path = path
        self.lr = lr
        self.replay_min_cred_ratio = replay_min_cred_ratio
        self.consequence_depth = consequence_depth

        # context_tuple → {token: accumulated_weight}
        self._dist: dict[tuple, dict] = {}
        # context_tuple → {offset: {token: weight}}
        self._conseq: dict[tuple, dict] = {}
        # running unigram across all replayed tokens
        self._unigram: dict[Any, float] = {}
        self._unigram_total: float = 0.0

        self._n_updates: int = 0
        self._run_history: list[dict] = []

        if path and os.path.exists(path):
            self.load(path)

    # ── prediction ────────────────────────────────────────────────────────────

    def predict(self, context: tuple) -> dict:
        """Normalised distribution for this exact context, or {} if unseen."""
        raw = self._dist.get(context)
        if not raw:
            return {}
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}

    def predict_consequence(self, context: tuple, offset: int = 2) -> dict:
        """
        Distribution over what tends to happen `offset` steps after context.
        Returns {} if the store has no consequence data for this context/offset.
        """
        node = self._conseq.get(context)
        if not node:
            return {}
        raw = node.get(offset, {})
        if not raw:
            return {}
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}

    def unigram(self) -> dict:
        """Normalised unigram distribution across all replayed tokens."""
        if not self._unigram_total:
            return {}
        return {k: v / self._unigram_total for k, v in self._unigram.items()}

    # ── blending ──────────────────────────────────────────────────────────────

    def blend(self, p_short: dict, context: tuple, vocab: set) -> dict:
        """
        Blend short-term distribution with long-term prior.

        λ_short → 1 when the short-term is confident (high max prob vs uniform).
        λ_short → 0 at cold start (short-term near-uniform).
        Falls through to unigram when the long-term store has no match either.
        """
        p_long = self.predict(context)

        # Estimate short-term confidence: how far above uniform is its peak?
        V = len(vocab) if vocab else 1
        uniform = 1.0 / V
        max_short = max(p_short.values()) if p_short else 0.0
        # Normalise to [0, 1]: 0 = completely random, 1 = completely certain
        lambda_s = min(1.0, max(0.0, (max_short - uniform) / max(1.0 - uniform, 1e-9)))

        if not p_long:
            # Fall back to unigram if long-term has nothing
            p_long = self.unigram()
        if not p_long:
            return p_short  # nothing to blend with

        lambda_l = 1.0 - lambda_s
        all_keys = set(p_short) | set(p_long)
        blended = {
            k: lambda_s * p_short.get(k, 0.0) + lambda_l * p_long.get(k, 0.0)
            for k in all_keys
        }
        total = sum(blended.values()) or 1.0
        return {k: v / total for k, v in blended.items()}

    # ── replay ────────────────────────────────────────────────────────────────

    def replay(self, short_predictor, sequence: list) -> dict:
        """
        Incorporate a completed sequence into the long-term store.

        Walks the short-term predictor's trie.  For each context node whose
        credibility exceeds replay_min_cred_ratio × cred_max, the observed
        token is replayed into the long-term store with weight proportional
        to how confident the short-term predictor was.

        Also records consequence chains (what happened offset steps later).

        Returns a stats dict and appends it to run_history().

        Parameters
        ----------
        short_predictor : UniversalPredictor
            The just-completed short-term predictor (before reset).
        sequence : list
            The complete token sequence that was just processed.
        """
        k = short_predictor.k
        cred_max = short_predictor._cred_max_base
        threshold = self.replay_min_cred_ratio * cred_max

        n_replayed = 0
        n_correct = 0

        for i in range(1, len(sequence)):
            actual = sequence[i]

            # Update unigram
            self._unigram[actual] = self._unigram.get(actual, 0.0) + 1.0
            self._unigram_total += 1.0

            for d in range(1, min(k, i) + 1):
                ctx = tuple(sequence[i - d:i])
                node = short_predictor._walk(ctx)
                if node is None or not node.succ_cred:
                    continue
                if node.node_cred < threshold:
                    continue

                # Weight = lr × how confident the node was (normalised)
                cred_ratio = min(1.0, node.node_cred / cred_max)
                weight = self.lr * cred_ratio

                # Update next-step distribution
                if ctx not in self._dist:
                    self._dist[ctx] = {}
                self._dist[ctx][actual] = self._dist[ctx].get(actual, 0.0) + weight
                n_replayed += 1

                # Track whether this node's top prediction was correct
                top = max(node.succ_cred, key=node.succ_cred.get)
                if top == actual:
                    n_correct += 1

                # Consequence chains
                if self.consequence_depth > 0:
                    if ctx not in self._conseq:
                        self._conseq[ctx] = {}
                    for offset in range(2, self.consequence_depth + 2):
                        future_idx = i + offset
                        if future_idx >= len(sequence):
                            break
                        future = sequence[future_idx]
                        decay = 0.7 ** (offset - 1)
                        if offset not in self._conseq[ctx]:
                            self._conseq[ctx][offset] = {}
                        self._conseq[ctx][offset][future] = (
                            self._conseq[ctx][offset].get(future, 0.0) + weight * decay
                        )

        self._n_updates += 1
        acc = n_correct / max(n_replayed, 1)
        stats = {
            'run': self._n_updates,
            'sequence_length': len(sequence),
            'n_replayed': n_replayed,
            'replay_accuracy': round(acc, 4),
            'total_contexts': len(self._dist),
            'unigram_vocab': len(self._unigram),
        }
        self._run_history.append(stats)

        if self.path:
            self.save()

        return stats

    # ── observability ─────────────────────────────────────────────────────────

    def run_history(self) -> list[dict]:
        """Per-run stats showing how the store has learned over time."""
        return list(self._run_history)

    def stats(self) -> dict:
        return {
            'total_contexts': len(self._dist),
            'total_runs': self._n_updates,
            'unigram_vocab': len(self._unigram),
            'consequence_contexts': len(self._conseq),
        }

    def learning_curve(self) -> list[float]:
        """Per-run replay accuracy as a plottable list. Watch it climb."""
        return [r['replay_accuracy'] for r in self._run_history]

    def top_consequences(self, context: tuple, offset: int = 2, n: int = 5) -> list[tuple]:
        """
        Most likely downstream outcomes `offset` steps after context.

        Returns list of (token, probability) sorted by probability descending.
        Useful for planning and lookahead: "what tends to happen 2 steps
        after seeing this context?"
        """
        dist = self.predict_consequence(context, offset)
        if not dist:
            return []
        items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        return items[:n]

    def coverage_report(self, sequence: list, k: int) -> dict:
        """
        Detailed breakdown of which contexts in sequence the store has seen.

        Returns
        -------
        dict with keys:
          coverage : float (fraction of k-grams matched)
          matched  : int   (number of k-grams with store data)
          total    : int   (total k-grams in sequence)
          novel    : list[tuple]  (first 10 unmatched contexts)
        """
        if len(sequence) < k + 1:
            return {'coverage': 0.0, 'matched': 0, 'total': 0, 'novel': []}
        matched = 0
        total_kgrams = 0
        novel = []
        for i in range(k, len(sequence)):
            ctx = tuple(sequence[i - k:i])
            total_kgrams += 1
            if ctx in self._dist:
                matched += 1
            elif len(novel) < 10:
                novel.append(ctx)
        return {
            'coverage': matched / total_kgrams if total_kgrams > 0 else 0.0,
            'matched': matched,
            'total': total_kgrams,
            'novel': novel,
        }

    def context_coverage(self, sequence: list, k: int) -> float:
        """Fraction of k-grams in sequence that the store has seen."""
        if len(sequence) < k + 1:
            return 0.0
        hits = sum(
            1 for i in range(k, len(sequence))
            if tuple(sequence[i - k:i]) in self._dist
        )
        return hits / (len(sequence) - k)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> None:
        """Gzip-pickle the store to disk."""
        p = path or self.path
        if not p:
            return
        data = {
            'dist': self._dist,
            'conseq': self._conseq,
            'unigram': self._unigram,
            'unigram_total': self._unigram_total,
            'n_updates': self._n_updates,
            'run_history': self._run_history,
            'lr': self.lr,
            'replay_min_cred_ratio': self.replay_min_cred_ratio,
            'consequence_depth': self.consequence_depth,
        }
        raw = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        with open(p, 'wb') as f:
            f.write(gzip.compress(raw, compresslevel=6))

    def load(self, path: str) -> None:
        """Load a previously saved store from disk."""
        with open(path, 'rb') as f:
            data = pickle.loads(gzip.decompress(f.read()))
        self._dist = data['dist']
        self._conseq = data.get('conseq', {})
        self._unigram = data.get('unigram', {})
        self._unigram_total = data.get('unigram_total', 0.0)
        self._n_updates = data.get('n_updates', 0)
        self._run_history = data.get('run_history', [])
        self.lr = data.get('lr', self.lr)
        self.replay_min_cred_ratio = data.get('replay_min_cred_ratio', self.replay_min_cred_ratio)
        self.consequence_depth = data.get('consequence_depth', self.consequence_depth)
