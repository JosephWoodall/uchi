"""
OnlineTokenizer
===============
Solves Problem 1 — Hard context ceiling, and
       Problem 10 — No joint optimization of compression and prediction.

A streaming BPE-style tokenizer that merges frequent adjacent token pairs
into single tokens while the sequence runs.  Because each of the k context
slots now covers more of the original sequence, the effective context window
grows without increasing model order.

Merge decisions are *scored* by whether they improve or hurt prediction
accuracy, creating a feedback loop between compression and prediction.
Merges that consistently hurt accuracy are undone automatically.

API
---
    tok = OnlineTokenizer(max_merges=64, merge_threshold=10)
    merged = tok.tokenize(raw_tokens)
    original = tok.detokenize(merged)
    tok.update(raw_tokens, predictor_accuracy)
    tok.stats()
    tok.active_merges
"""

from collections import Counter
from typing import Any


class OnlineTokenizer:
    """
    Online BPE-style tokenizer that merges frequent token pairs during streaming.

    Extends the effective context window by combining frequent adjacent tokens.
    Each of the k context slots then covers more of the original sequence.
    Merge decisions are scored by prediction accuracy impact.

    Parameters
    ----------
    max_merges : int
        Maximum number of merge rules to learn (default 64).
    merge_threshold : int
        Minimum pair frequency before considering a merge (default 10).
    score_window : int
        Number of tokens after a merge to measure accuracy impact (default 20).
    undo_threshold : float
        If a merge's running accuracy score drops below this, undo it (default -0.05).
    """

    __slots__ = (
        'max_merges',
        'merge_threshold',
        'score_window',
        'undo_threshold',
        '_pair_counts',
        '_merges',
        '_merge_scores',
        '_merge_baselines',
        '_merge_ages',
        '_n_updates',
        '_baseline_accuracy',
        '_undone_pairs',
    )

    def __init__(
        self,
        max_merges: int = 64,
        merge_threshold: int = 10,
        score_window: int = 20,
        undo_threshold: float = -0.05,
    ):
        self.max_merges = max_merges
        self.merge_threshold = merge_threshold
        self.score_window = score_window
        self.undo_threshold = undo_threshold

        # pair (tok_a, tok_b) -> count
        self._pair_counts: Counter = Counter()
        # pair (tok_a, tok_b) -> merged token  ('__merged__', tok_a, tok_b)
        self._merges: dict[tuple, Any] = {}
        # pair -> running EMA accuracy delta score
        self._merge_scores: dict[tuple, float] = {}
        # pair -> baseline accuracy at time of merge creation
        self._merge_baselines: dict[tuple, float] = {}
        # pair -> update step when merge was created
        self._merge_ages: dict[tuple, int] = {}
        # total update calls received
        self._n_updates: int = 0
        # EMA of predictor accuracy over all updates (serves as baseline)
        self._baseline_accuracy: float = 0.0
        # set of pairs that were merged then undone (avoid re-merging)
        self._undone_pairs: set[tuple] = set()

    # ── public API ────────────────────────────────────────────────────────────

    def tokenize(self, raw_tokens: list) -> list:
        """
        Apply all active merge rules greedily left-to-right.

        For each position, check if ``(tokens[i], tokens[i+1])`` has a merge
        rule; if so, replace with the merged token and advance past both.

        Parameters
        ----------
        raw_tokens : list
            Sequence of hashable tokens.

        Returns
        -------
        list
            Token sequence with merge rules applied.
        """
        if not self._merges or not raw_tokens:
            return list(raw_tokens)

        result: list = []
        i = 0
        n = len(raw_tokens)
        while i < n:
            if i + 1 < n:
                pair = (raw_tokens[i], raw_tokens[i + 1])
                merged = self._merges.get(pair)
                if merged is not None:
                    result.append(merged)
                    i += 2
                    continue
            result.append(raw_tokens[i])
            i += 1
        return result

    def detokenize(self, merged_tokens: list) -> list:
        """
        Recursively expand merged tokens back to their original components.

        Parameters
        ----------
        merged_tokens : list
            Token sequence potentially containing merged tokens.

        Returns
        -------
        list
            Fully expanded token sequence with only original (non-merged) tokens.
        """
        result: list = []
        for tok in merged_tokens:
            expanded = self._expand(tok)
            result.extend(expanded)
        return result

    def update(self, raw_tokens: list, predictor_accuracy: float) -> None:
        """
        Called after each training step.  Updates pair frequencies, considers
        new merges, and scores existing merges.

        Parameters
        ----------
        raw_tokens : list
            The latest raw (un-merged) token window.
        predictor_accuracy : float
            Accuracy of the predictor on this step (0.0-1.0).
        """
        self._n_updates += 1

        # Update running baseline accuracy (EMA)
        if self._n_updates == 1:
            self._baseline_accuracy = predictor_accuracy
        else:
            self._baseline_accuracy = (
                0.95 * self._baseline_accuracy + 0.05 * predictor_accuracy
            )

        # Count adjacent pairs in the raw token window
        self._count_pairs(raw_tokens)

        # Score existing merges against accuracy
        self._score_merges(predictor_accuracy)

        # Prune merges that consistently hurt accuracy
        self._prune_merges()

        # Consider adding a new merge
        self._consider_merge()

    def stats(self) -> dict:
        """
        Return merge table info, scores, and frequencies.

        Returns
        -------
        dict
            Keys: ``n_merges``, ``n_updates``, ``baseline_accuracy``,
            ``n_undone``, ``merges`` (list of per-merge dicts).
        """
        merge_info = []
        for pair, merged in self._merges.items():
            merge_info.append({
                'pair': pair,
                'merged_token': merged,
                'score': round(self._merge_scores.get(pair, 0.0), 6),
                'frequency': self._pair_counts.get(pair, 0),
                'age': self._n_updates - self._merge_ages.get(pair, 0),
            })
        # Sort by score descending
        merge_info.sort(key=lambda m: m['score'], reverse=True)

        return {
            'active_merges': len(self._merges),
            'total_merges': len(self._merges) + len(self._undone_pairs),
            'n_updates': self._n_updates,
            'baseline_accuracy': round(self._baseline_accuracy, 6),
            'undone_merges': len(self._undone_pairs),
            'merges': merge_info,
        }

    @property
    def active_merges(self) -> list[tuple]:
        """
        List of ``(pair, merged_token, score, frequency)`` tuples sorted by
        score descending.

        Returns
        -------
        list[tuple]
            Each entry is ``((tok_a, tok_b), merged_token, score, frequency)``.
        """
        entries = []
        for pair, merged in self._merges.items():
            entries.append((
                pair,
                merged,
                self._merge_scores.get(pair, 0.0),
                self._pair_counts.get(pair, 0),
            ))
        entries.sort(key=lambda e: e[2], reverse=True)
        return entries

    # ── internal ──────────────────────────────────────────────────────────────

    def _count_pairs(self, tokens: list) -> None:
        """
        Update pair frequency counts from a token sequence.

        Parameters
        ----------
        tokens : list
            Raw token sequence.
        """
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            self._pair_counts[pair] += 1

    def _consider_merge(self) -> None:
        """
        If we haven't hit max_merges and the most frequent unmerged pair
        exceeds merge_threshold, create a new merge rule.  Record the
        baseline accuracy at the time of merge.
        """
        if len(self._merges) >= self.max_merges:
            return

        # Find the most frequent pair not already merged or previously undone
        best_pair = None
        best_count = 0
        for pair, count in self._pair_counts.items():
            if pair in self._merges:
                continue
            if pair in self._undone_pairs:
                continue
            if count > best_count:
                best_count = count
                best_pair = pair

        if best_pair is None or best_count < self.merge_threshold:
            return

        # Create the merge rule
        merged_token = ('__merged__', best_pair[0], best_pair[1])
        self._merges[best_pair] = merged_token
        self._merge_scores[best_pair] = 0.0
        self._merge_baselines[best_pair] = self._baseline_accuracy
        self._merge_ages[best_pair] = self._n_updates

    def _score_merges(self, predictor_accuracy: float) -> None:
        """
        Update running accuracy delta for every active merge.

        The delta measures how the predictor's accuracy compares to the
        baseline that existed when the merge was created.  Updated with
        an exponential moving average (alpha = 0.1).

        Parameters
        ----------
        predictor_accuracy : float
            Current predictor accuracy (0.0-1.0).
        """
        for pair in list(self._merge_scores):
            baseline = self._merge_baselines.get(pair, self._baseline_accuracy)
            delta = predictor_accuracy - baseline
            self._merge_scores[pair] = (
                0.9 * self._merge_scores[pair] + 0.1 * delta
            )

    def _prune_merges(self) -> None:
        """
        Remove merges whose running accuracy score has dropped below
        undo_threshold.  These pairs are added to the undone set so
        they won't be re-merged.
        """
        to_remove = [
            pair for pair, score in self._merge_scores.items()
            if score < self.undo_threshold
            # Don't prune too early — give the merge at least score_window
            # updates to prove itself.
            and (self._n_updates - self._merge_ages.get(pair, 0)) >= self.score_window
        ]
        for pair in to_remove:
            del self._merges[pair]
            del self._merge_scores[pair]
            del self._merge_baselines[pair]
            del self._merge_ages[pair]
            self._undone_pairs.add(pair)

    def _expand(self, token: Any) -> list:
        """
        Recursively expand a (possibly merged) token into original tokens.

        A merged token is a tuple ``('__merged__', a, b)``.  Each of ``a``
        and ``b`` may themselves be merged tokens, so expansion is recursive.

        Parameters
        ----------
        token : Any
            A single token, possibly a merged composite.

        Returns
        -------
        list
            Flat list of original (non-merged) tokens.
        """
        if (
            isinstance(token, tuple)
            and len(token) == 3
            and token[0] == '__merged__'
        ):
            left = self._expand(token[1])
            right = self._expand(token[2])
            return left + right
        return [token]
