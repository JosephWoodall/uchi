"""Goal state tracking for OmniRouter.

Maintains a persistent model of what the user is trying to accomplish across
turns. Distinguishes per-turn intent (what does this message ask?) from
cross-turn objective (what is the user ultimately building toward?).

The tracker is intentionally lightweight — it uses token overlap and a
sliding window of recent concepts, not a learned model. This keeps it fast,
interpretable, and free of the LLM dependency this project forbids.

The goal state is stored on OmniRouter and survives between calls. It is
serialized with the brain (as a plain dataclass, no special handling needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# A query is considered "related" to the current goal thread if it shares
# at least this fraction of content tokens with recent queries.
_CONTINUITY_THRESHOLD = 0.25
# Number of recent turns to consider for goal inference.
_WINDOW = 6
# Minimum turns before we commit to a non-trivial long-term objective.
_MIN_TURNS_FOR_LT = 3


@dataclass
class GoalState:
    """Persistent cross-turn goal model.

    Attributes:
        ultimate_objective: The inferred long-term objective as a string of
            key concept tokens. Empty until enough turns accumulate.
        objective_confidence: Proportion of recent turns that share the
            objective vocabulary (0.0 – 1.0).
        short_term_intent: The intent of the most recent single turn.
        long_term_intent: The dominant intent class across recent turns
            (e.g. "code", "math", "factual", "convo").
        turn_count: How many consecutive turns have been on the same topic.
        recent_concepts: Sliding window of concept token lists from recent turns.
    """

    ultimate_objective:    str         = ""
    objective_confidence:  float       = 0.0
    short_term_intent:     str         = ""
    long_term_intent:      str         = ""
    turn_count:            int         = 0
    recent_concepts:       List[List[str]] = field(default_factory=list)

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _content_tokens(tokens: List[str]) -> set:
        """Filter out special/stop tokens and very short tokens."""
        stop = {"<|user|>", "<|assistant|>", "<|end|>", "the", "a", "an",
                "is", "are", "was", "were", "be", "of", "in", "to", "and",
                "or", "for", "it", "this", "that"}
        return {t.lower() for t in tokens if len(t) > 2 and t not in stop}

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, concepts: List[str], intent_key: Optional[str] = None) -> None:
        """Update goal state with the new turn's concepts and intent.

        Args:
            concepts:   Content tokens from the current query.
            intent_key: Procedural intent class ("code", "math", etc.) or None.
        """
        current_set = self._content_tokens(concepts)

        # Compute similarity to recent turns.
        recent_sets = [self._content_tokens(c) for c in self.recent_concepts]
        if recent_sets:
            similarities = [self._jaccard(current_set, s) for s in recent_sets]
            avg_sim = sum(similarities) / len(similarities)
        else:
            avg_sim = 0.0

        # Decide whether this turn continues the current goal thread.
        if avg_sim >= _CONTINUITY_THRESHOLD:
            self.turn_count += 1
        else:
            # New goal thread — reset but keep one-turn history for comparison.
            self.turn_count = 1

        # Update sliding window.
        self.recent_concepts.append(concepts)
        if len(self.recent_concepts) > _WINDOW:
            self.recent_concepts.pop(0)

        # Short-term intent.
        self.short_term_intent = intent_key or ""

        # Long-term intent: majority vote over recent intent keys.
        # (We don't store history of intents here, so we approximate by
        # keeping the first non-None intent of the current thread if consistent.)
        if intent_key and self.long_term_intent != intent_key:
            if self.turn_count <= 1:
                self.long_term_intent = intent_key
            # else: keep existing LT intent unless we've seen a clear shift

        # Ultimate objective: union of most-common content tokens across window.
        if len(self.recent_concepts) >= _MIN_TURNS_FOR_LT:
            all_tokens: dict = {}
            for c in self.recent_concepts:
                for tok in self._content_tokens(c):
                    all_tokens[tok] = all_tokens.get(tok, 0) + 1
            # Keep tokens that appear in at least half the window turns.
            min_freq = max(1, len(self.recent_concepts) // 2)
            core = [tok for tok, cnt in all_tokens.items() if cnt >= min_freq]
            core.sort(key=lambda t: -all_tokens[t])  # most frequent first
            self.ultimate_objective = " ".join(core[:8])
            self.objective_confidence = (
                sum(1 for _, cnt in all_tokens.items() if cnt >= min_freq)
                / max(len(all_tokens), 1)
            )
        else:
            self.ultimate_objective   = " ".join(sorted(current_set)[:6])
            self.objective_confidence = 0.0

    def objective_tokens(self) -> List[str]:
        """Return the current objective as a token list for bias injection."""
        return self.ultimate_objective.split() if self.ultimate_objective else []

    def is_new_thread(self) -> bool:
        """True if the current turn starts a new goal thread."""
        return self.turn_count <= 1

    def summary(self) -> str:
        """Short human-readable summary of the current goal state."""
        if not self.ultimate_objective:
            return f"turn_count={self.turn_count} intent={self.short_term_intent or 'unknown'}"
        return (
            f"objective='{self.ultimate_objective}' "
            f"confidence={self.objective_confidence:.2f} "
            f"turns={self.turn_count} "
            f"lt_intent={self.long_term_intent or 'unknown'}"
        )
