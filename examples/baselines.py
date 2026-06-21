"""
Online sequence prediction baselines.

All classes share the same interface as UniversalPredictor:
    predict()   -> (value, confidence)   called BEFORE observe()
    observe(v)  -> None                  appends v to history
    feedback(v) -> None                  updates model tables
"""
from collections import defaultdict
from typing import Any


class PersistencePredictor:
    """Always predict the most recently seen value."""

    def __init__(self):
        self.history: list = []

    def predict(self):
        if not self.history:
            return None, 0.0
        return self.history[-1], 1.0

    def observe(self, v):
        self.history.append(v)

    def feedback(self, v):
        pass


class MajorityPredictor:
    """Always predict the most frequently seen value so far."""

    def __init__(self):
        self.history: list = []
        self._counts: dict = defaultdict(int)

    def predict(self):
        if not self._counts:
            return None, 0.0
        best = max(self._counts, key=self._counts.get)
        total = sum(self._counts.values())
        return best, self._counts[best] / total

    def observe(self, v):
        self.history.append(v)

    def feedback(self, v):
        self._counts[v] += 1


class NgramPredictor:
    """
    Variable-order n-gram with Laplace smoothing and exact backoff from
    max_order down to order 1.  Falls back to unigram when no context matches.

    Protocol note: observe() appends v; feedback() indexes relative to
    the already-appended history, so the order-k context for successor v is
    history[-(k+1):-1].
    """

    def __init__(self, max_order: int = 5):
        self.max_order = max_order
        self.history: list = []
        # _counts[k][context_tuple][symbol] = count  (k = 1..max_order)
        self._counts: list = [defaultdict(lambda: defaultdict(int))
                               for _ in range(max_order + 1)]
        self._unigram: dict = defaultdict(int)
        self._vocab: set = set()

    def predict(self):
        if not self._vocab:
            return None, 0.0

        V = len(self._vocab)

        for k in range(min(self.max_order, len(self.history)), 0, -1):
            ctx = tuple(self.history[-k:])
            dist = self._counts[k][ctx]
            if not dist:
                continue
            total = sum(dist.values())
            best = max(self._vocab, key=lambda s, d=dist, t=total, v=V:
                       (d.get(s, 0) + 1) / (t + v))
            conf = (dist.get(best, 0) + 1) / (total + V)
            return best, conf

        # Unigram fallback
        total = sum(self._unigram.values())
        best = max(self._vocab, key=lambda s: (self._unigram.get(s, 0) + 1) / (total + V))
        conf = (self._unigram.get(best, 0) + 1) / (total + V)
        return best, conf

    def observe(self, v):
        self.history.append(v)

    def feedback(self, v):
        # history[-1] == v; order-k context = history[-(k+1):-1]
        self._vocab.add(v)
        self._unigram[v] += 1
        for k in range(1, min(self.max_order, len(self.history) - 1) + 1):
            ctx = tuple(self.history[-(k + 1):-1])
            self._counts[k][ctx][v] += 1


class PPMPredictor:
    """
    PPM-D (Prediction by Partial Matching, variant D) with Witten-Bell
    escape probabilities and the exclusion principle.

    At each order k (from max_order down to 1):
        N_k = total count at this context
        C_k = number of unique symbols seen after this context (Witten-Bell C)
        escape = C_k / (N_k + C_k)
        Each non-excluded symbol s gets: remaining_mass * count(s) / (N_k + C_k)
        Then excluded.add(all symbols at this context)
        remaining_mass *= escape

    Remaining mass after all orders is split uniformly over vocab − excluded.
    Final distribution is normalised for numerical safety.
    """

    def __init__(self, max_order: int = 5):
        self.max_order = max_order
        self.history: list = []
        # _counts[k][context_tuple][symbol] = count  (k = 1..max_order)
        self._counts: list = [defaultdict(lambda: defaultdict(int))
                               for _ in range(max_order + 1)]
        self._vocab: set = set()

    def _distribution(self) -> dict:
        excluded: set = set()
        probs: dict = defaultdict(float)
        remaining = 1.0

        for k in range(min(self.max_order, len(self.history)), 0, -1):
            ctx = tuple(self.history[-k:])
            ctx_dist = self._counts[k][ctx]
            if not ctx_dist:
                continue

            N = sum(ctx_dist.values())
            C = len(ctx_dist)            # Witten-Bell unique-symbol count

            for sym, cnt in ctx_dist.items():
                if sym not in excluded:
                    probs[sym] += remaining * cnt / (N + C)

            excluded.update(ctx_dist.keys())
            remaining *= C / (N + C)

            if remaining < 1e-12:
                break

        # Order-0 fallback: uniform over vocab − excluded
        if remaining > 1e-12 and self._vocab:
            leftover = self._vocab - excluded
            targets = leftover if leftover else self._vocab
            p_each = remaining / len(targets)
            for sym in targets:
                probs[sym] += p_each

        total = sum(probs.values())
        if total < 1e-12:
            return {}
        return {s: p / total for s, p in probs.items()}

    def predict(self):
        if not self._vocab:
            return None, 0.0
        dist = self._distribution()
        if not dist:
            return None, 0.0
        best = max(dist, key=dist.get)
        return best, dist[best]

    def observe(self, v):
        self.history.append(v)

    def feedback(self, v):
        # history[-1] == v; order-k context = history[-(k+1):-1]
        self._vocab.add(v)
        for k in range(1, min(self.max_order, len(self.history) - 1) + 1):
            ctx = tuple(self.history[-(k + 1):-1])
            self._counts[k][ctx][v] += 1
