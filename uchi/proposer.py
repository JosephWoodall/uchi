"""
proposer.py — the pluggable GENERATOR half of Generate-and-Ground (the merge seam).

Uchi is the **verifier**. The Proposer is the fallible **generator** it gates. The
whole architecture is "a fallible proposer + a reality-anchored verifier" — so the
proposer is deliberately swappable:

    - `DecoderProposer` — Uchi's small from-scratch decoder (baseline; ships today).
    - `FluxProposer`    — the distilled FLUX SSM/ternary model (efficient_llm_training).
    - (future) `LLMProposer` — any local open LLM.

A better proposer raises the ceiling on **reasoning and OOD generalization**; Uchi's
oracle + answerability gate keep the output **honest regardless of how good or bad
the proposer is** — a weak proposal just becomes an abstention. That is the point:
you get to use a strong-but-fallible generator *safely*, because the verifier never
lets an ungrounded claim out.

Interface
---------
    propose(question, evidence) -> str          # a grounded candidate answer (RAG-conditioned)
    plan(question) -> list[str] | None           # OPTIONAL: decompose into sub-steps
                                                 #   (this is how a smart proposer solves the
                                                 #    ReasoningEngine's hard planner problem)
"""
from __future__ import annotations

import re
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class Proposer(Protocol):
    """Any generator Uchi can gate. `propose` is required; `plan` is optional."""

    def propose(self, question: str, evidence: list[str]) -> str:
        """Generate a candidate answer conditioned on the retrieved evidence."""
        ...

    def plan(self, question: str) -> Optional[list[str]]:
        """Decompose a complex question into verifiable sub-steps, or None."""
        ...


def _parse_numbered(text: str) -> list[str]:
    steps = re.split(r"(?:^|\n)\s*\d+[.)]\s+", text)
    return [s.strip() for s in steps if s.strip()][:8]


# ── adapter 1: the from-scratch decoder (baseline, ships today) ────────────────
class DecoderProposer:
    """Wraps `uchi.decoder.NeuralDecoder`. Grounds well, but is small and rough and
    cannot plan — the honest baseline proposer."""

    def __init__(self, decoder) -> None:
        self._d = decoder

    def propose(self, question: str, evidence: list[str]) -> str:
        return self._d.generate(question, evidence)

    def plan(self, question: str) -> Optional[list[str]]:
        return None                      # the small decoder can't decompose

    @classmethod
    def load(cls, path: str) -> "Optional[DecoderProposer]":
        from uchi.decoder import NeuralDecoder
        return cls(NeuralDecoder.load(path)) if NeuralDecoder.exists(path) else None


# ── adapter 2: FLUX — the distilled small SSM/ternary model ────────────────────
class FluxProposer:
    """Wraps a FLUX model (efficient_llm_training) as a Proposer.

    INTEGRATION SEAM — FLUX supplies one callable::

        generate_fn(prompt: str, max_tokens: int) -> str

    This adapter owns the *prompting*: it builds a retrieval-augmented prompt for
    `propose` (answer strictly from the evidence) and a decomposition prompt for
    `plan`. FLUX proposes and plans; Uchi's oracle/answerability verify. Because a
    distilled FLUX has real (if imperfect) reasoning, this is what closes the
    general-reasoning / OOD gap — safely, behind the verifier.

    Wire it by passing FLUX's inference function, e.g. from
    `efficient_llm_training/src/inference_engine.py`.
    """

    _ANSWER = ("Context:\n{ctx}\n\nUsing ONLY the context above, answer the question "
               "concisely. If the context does not answer it, say you don't know.\n"
               "Q: {q}\nA:")
    _PLAN = ("Break the question into a short numbered list of simple, checkable "
             "sub-steps (each a lookup or a calculation).\nQuestion: {q}\nSteps:\n1. ")

    def __init__(self, generate_fn, max_answer_tokens: int = 64, max_plan_tokens: int = 128) -> None:
        self._gen = generate_fn
        self.max_answer_tokens = max_answer_tokens
        self.max_plan_tokens = max_plan_tokens

    def propose(self, question: str, evidence: list[str]) -> str:
        ctx = "\n".join(evidence[:4]) if evidence else "(no context)"
        prompt = self._ANSWER.format(ctx=ctx, q=question)
        try:
            return (self._gen(prompt, self.max_answer_tokens) or "").strip()
        except Exception:
            return ""

    def plan(self, question: str) -> Optional[list[str]]:
        try:
            raw = "1. " + (self._gen(self._PLAN.format(q=question), self.max_plan_tokens) or "")
            steps = _parse_numbered(raw)
            return steps if len(steps) > 1 else None
        except Exception:
            return None

    @classmethod
    def from_inference_fn(cls, generate_fn, **kw) -> "FluxProposer":
        return cls(generate_fn, **kw)

    @classmethod
    def load(cls, checkpoint: Optional[str] = None):
        """Load FLUX from the vendored `uchi.flux` package (the model now lives in
        this repo). Returns None if the checkpoint/deps are missing, so the loader
        degrades gracefully to the decoder."""
        try:
            from uchi.flux import build_generate_fn
            return cls(build_generate_fn(checkpoint=checkpoint))
        except Exception:
            return None


# ── factory: pick the best available proposer ─────────────────────────────────
def load_proposer(prefer: str = "flux", decoder_path: Optional[str] = None):
    """Return the best available Proposer, or None (extractive fallback downstream).

    Order: FLUX (if preferred + available) → from-scratch decoder → None.
    """
    if prefer == "flux":
        flux = FluxProposer.load()
        if flux is not None:
            return flux
    if decoder_path:
        dec = DecoderProposer.load(decoder_path)
        if dec is not None:
            return dec
    return None
