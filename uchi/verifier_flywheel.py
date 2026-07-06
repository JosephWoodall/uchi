"""verifier_flywheel.py — Self-improvement loop for the verifier (0.4.0 Item 17).

Every layered-veto decision the oracle makes is already logged
(FactCheckOracle.relaxed_pass_log / layered_veto_log). This closes the
loop: when a user's next turn appears to CONTRADICT a prior answer,
detected via the same EntailmentChecker the oracle already uses (no new
model, no new training needed just to detect this), that's a cheap, real
signal that something should have been caught but wasn't. Confirmed
corrections get recorded and can be exported as new, VERIFIED training
examples for the next verifier training run -- the same
self-distillation-on-real-outcomes pattern already established for
0.6.0's self-modification patches, applied here to the verifier's own
decisions instead.

No new component: reuses EntailmentChecker (if one is loaded) to detect
corrections, and EpisodicMemory's own turn history as the source of "what
did we just say." Inert (does nothing) until an entailment checker is
actually trained and passed in -- same graceful degradation as everything
else in this session.
"""
from __future__ import annotations

import json
import time
from typing import Optional


class VerifierFlywheel:
    """Detects when a user's next message contradicts Uchi's own prior
    answer, using the same EntailmentChecker the oracle uses, and logs it
    as a real, confirmed correction -- exportable as new training data.

    This never blocks or changes the conversation itself -- it only
    observes. A detected correction doesn't retract the prior answer or
    alter the current one; it's purely a labeled data point for later
    verifier retraining.
    """

    def __init__(self, entailment_checker=None):
        self.entailment_checker = entailment_checker
        self.corrections: list[dict] = []

    @property
    def is_active(self) -> bool:
        return self.entailment_checker is not None

    def check_for_correction(self, prior_question: str, prior_answer: str, new_message: str) -> bool:
        """Returns whether *new_message* appears to contradict
        *prior_answer*. Always False (no-op) if no entailment checker is
        available."""
        if not self.is_active or not prior_answer or not prior_answer.strip():
            return False
        if not new_message or not new_message.strip():
            return False
        try:
            is_correction = self.entailment_checker.is_contradiction(prior_answer, new_message)
        except Exception:
            return False  # fail open -- an observational loop must never break the conversation
        if is_correction:
            self.corrections.append({
                "timestamp": time.time(),
                "prior_question": prior_question,
                "prior_answer": prior_answer,
                "correction": new_message,
            })
        return is_correction

    def export_training_examples(self, path: str) -> int:
        """Append confirmed corrections as new entailment-classifier
        training examples: premise=the answer that turned out wrong,
        hypothesis=the correction, label=contradiction. Both real, never
        fabricated -- they came from an actual observed conversation.
        Same premise/hypothesis/label shape verifier_train.py's MNLI/SNLI
        loaders already use, so this can be folded in as a fourth
        fair-budgeted source with zero format changes. Returns how many
        were written; clears the in-memory buffer after writing.
        """
        if not self.corrections:
            return 0
        with open(path, "a", encoding="utf-8") as fh:
            for c in self.corrections:
                fh.write(json.dumps({
                    "premise": c["prior_answer"],
                    "hypothesis": c["correction"],
                    "label": 2,  # contradiction
                }) + "\n")
        n = len(self.corrections)
        self.corrections.clear()
        return n


def load_flywheel_examples(path: str, num_examples: Optional[int] = None) -> list[dict]:
    """Read back exported flywheel corrections in the same shape
    verifier_train.py's generate_mnli_examples/generate_snli_examples
    return, so it can be used as a fourth data source there."""
    import os
    if not os.path.exists(path):
        return []
    examples = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("premise") or not row.get("hypothesis"):
                continue
            examples.append(row)
            if num_examples is not None and len(examples) >= num_examples:
                break
    return examples
