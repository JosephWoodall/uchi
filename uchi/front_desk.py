"""front_desk.py — Front-Desk friendly-tone translation pass (0.4.0 Item 16.6).

Bridges the gap between strict logic and warm conversation: feeds a
verified, dry grounded answer back through FLUX with a constrained
prompt ("deliver this fact in a friendly tone, do NOT add new facts"),
then checks the friendlied version with the *same* ``FactCheckOracle``
used everywhere else in the pipeline, using the original dry fact as the
sole evidence. If the rewrite isn't grounded in the original — i.e. FLUX
added something not actually there — it's rejected and the original dry
answer is returned unchanged. A tone pass doesn't get an exemption from
the North Star's grounding requirement just because it "only" changes
phrasing.
"""
from __future__ import annotations

from typing import Any, Optional


def friendly_tone_pass(answer: str, proposer: Any, oracle: Optional[Any] = None) -> str:
    """Rewrite *answer* in a warmer tone via *proposer*, verified against
    the original fact so no new claims can slip in. Returns *answer*
    unchanged if there's no proposer, the rewrite fails, or it isn't
    grounded in the original.
    """
    if proposer is None or not answer or not answer.strip():
        return answer

    prompt = (
        "Deliver the following verified fact in a warm, friendly, conversational "
        "tone. Do NOT add any new facts, numbers, or claims not already present. "
        "Only rephrase.\n\n"
        f"Verified fact: {answer}"
    )
    try:
        friendly = proposer.propose(prompt, evidence=[])
    except Exception:
        return answer
    if not friendly or not friendly.strip():
        return answer

    if oracle is not None:
        try:
            if not oracle.is_grounded(friendly, [answer]):
                return answer
        except Exception:
            return answer

    return friendly
