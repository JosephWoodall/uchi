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

    def propose(self, question: str, evidence: list[str], think: bool = False) -> str:
        """Generate a candidate answer conditioned on the retrieved evidence.

        `think=True` primes the model to reason step-by-step before answering
        (the CoT-trained <|think|>...<|/think|> behaviour) instead of jumping
        straight to <|assistant|>. Proposers without that training ignore it.
        """
        ...

    def plan(self, question: str) -> Optional[str]:
        """Decompose a complex question into a structural DSL Grid, or None."""
        ...

    def propose_batch(self, question: str, evidence: list[str], n: int, think: bool = False) -> list[str]:
        """OPTIONAL: generate n candidate answers for self-consistency voting.

        A proposer without a genuinely batched path may just call `propose`
        n times sequentially -- this method exists so callers that WANT the
        vectorized speedup (0.5.0 Item 7's "dynamic-N doesn't cost N×
        latency") can ask for it without caring which proposer they're
        using; callers that don't care can keep looping `propose` directly.
        """
        ...


# ── adapter 1: the from-scratch decoder (baseline, ships today) ────────────────
class DecoderProposer:
    """Wraps `uchi.decoder.NeuralDecoder`. Grounds well, but is small and rough and
    cannot plan — the honest baseline proposer."""

    def __init__(self, decoder) -> None:
        self._d = decoder

    def propose(self, question: str, evidence: list[str], think: bool = False) -> str:
        return self._d.generate(question, evidence)   # from-scratch decoder has no think-trace format

    def propose_batch(self, question: str, evidence: list[str], n: int, think: bool = False) -> list[str]:
        return [self.propose(question, evidence, think=think) for _ in range(max(n, 1))]

    def plan(self, question: str) -> Optional[str]:
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
    _PLAN = ("Before answering the question, map out the variables, entities, and logical constraints "
             "into a structured DSL Grid (e.g., State(A)=1, Relation(A,B)=True). Use this grid as a scratchpad.\n\n"
             "Question: {q}\nDSL Grid:\n")

    def __init__(self, generate_fn, generate_batch_fn=None, max_answer_tokens: int = 64,
                 max_plan_tokens: int = 128, max_think_tokens: int = 160) -> None:
        self._gen = generate_fn
        # OPTIONAL (0.5.0 Item 7's last bullet) -- from build_batch_generate_fn,
        # a second loaded model instance, not the same one generate_fn uses (see
        # that factory's docstring on why they don't share weights). None by
        # default, same graceful-degradation shape as every other optional
        # signal in this codebase (proprioception, answerability, ...): a
        # caller that doesn't explicitly ask for it via load(with_batch=True)
        # pays no extra GPU memory for it.
        self._gen_batch = generate_batch_fn
        self.max_answer_tokens = max_answer_tokens
        self.max_plan_tokens = max_plan_tokens
        self.max_think_tokens = max_think_tokens   # reasoning + answer needs more room than answer alone

    def _build_prompt(self, question: str, evidence: list[str], think: bool) -> tuple[str, int]:
        """Shared prompt-building logic for `propose`/`propose_batch` -- kept
        in one place so the batched path can't silently drift from the
        exact prompt shape `propose` uses."""
        if think:
            # CoT trained on the RAW question with no wrapper text at all
            # (cot_distill.py: prompt_ids = [user_id] + encode_text(question)) —
            # no "Context:\nUsing ONLY..." instruction paragraph, no context
            # block. Wrapping it in the SFT-style _ANSWER template here is an
            # out-of-distribution prompt shape that produced degenerate/looping
            # output when tried; matching training's exact format is what makes
            # <|think|> prime real reasoning instead. Grounding against evidence
            # still happens downstream via the oracle regardless of prompt shape.
            prompt = question
        else:
            ctx = "\n".join(evidence[:4]) if evidence else "(no context)"
            prompt = self._ANSWER.format(ctx=ctx, q=question)
        max_tokens = self.max_think_tokens if think else self.max_answer_tokens
        return prompt, max_tokens

    def propose(self, question: str, evidence: list[str], think: bool = False) -> str:
        prompt, max_tokens = self._build_prompt(question, evidence, think)
        try:
            # `think=True` primes <|think|> instead of jumping straight to
            # <|assistant|>, eliciting CoT's trained reasoning-before-answering.
            # generate_fn degrades to plain answering if it doesn't support the
            # kwarg (e.g. an older cached generate_fn from before this feature).
            try:
                return (self._gen(prompt, max_tokens, think=think) or "").strip()
            except TypeError:
                return (self._gen(prompt, max_tokens) or "").strip()
        except Exception:
            return ""

    def propose_batch(self, question: str, evidence: list[str], n: int, think: bool = False) -> list[str]:
        """n candidates for the SAME question in one vectorized forward pass
        when `generate_batch_fn` is available (0.5.0 Item 7's last bullet);
        degrades to `n` sequential `propose()` calls otherwise -- so a
        caller can always use this method without checking first, same
        graceful-degradation shape as `plan()` returning None when the
        proposer can't decompose.
        """
        if n <= 1:
            return [self.propose(question, evidence, think=think)]
        if self._gen_batch is None:
            return [self.propose(question, evidence, think=think) for _ in range(n)]
        prompt, max_tokens = self._build_prompt(question, evidence, think)
        try:
            return [(c or "").strip() for c in self._gen_batch(prompt, n, max_tokens, think=think)]
        except Exception:
            return [self.propose(question, evidence, think=think) for _ in range(n)]

    def plan(self, question: str) -> Optional[str]:
        try:
            return (self._gen(self._PLAN.format(q=question), self.max_plan_tokens) or "").strip()
        except Exception:
            return None

    @classmethod
    def from_inference_fn(cls, generate_fn, **kw) -> "FluxProposer":
        return cls(generate_fn, **kw)

    @classmethod
    def load(cls, checkpoint: Optional[str] = None, pruned_vocab: Optional[str] = None,
             with_batch: bool = False):
        """Load FLUX from the vendored `uchi.flux` package (the model now lives in
        this repo). Returns None if the checkpoint/deps are missing, so the loader
        degrades gracefully to the decoder.

        *pruned_vocab* overrides the pruned-vocab tokenizer file
        ``build_generate_fn`` auto-detects when a checkpoint's vocab_size
        doesn't match the full tokenizer's (0.4.0 Item 0) -- pass it
        explicitly only if a checkpoint used a pruned vocab other than the
        one at the default path.

        *with_batch*: also load `build_batch_generate_fn`'s vectorized
        N-candidates path (0.5.0 Item 7's last bullet), enabling
        `propose_batch`'s real speedup instead of its sequential fallback.
        Off by default -- it loads a SECOND model instance (see that
        factory's docstring), so this is an explicit opt-in to the extra
        GPU memory, not a silent default change for existing callers.
        """
        try:
            from uchi.flux import build_generate_fn
            gen_fn = build_generate_fn(checkpoint=checkpoint, pruned_vocab=pruned_vocab)
        except Exception:
            return None

        gen_batch_fn = None
        if with_batch:
            try:
                from uchi.flux import build_batch_generate_fn
                gen_batch_fn = build_batch_generate_fn(checkpoint=checkpoint, pruned_vocab=pruned_vocab)
            except Exception:
                pass  # propose_batch degrades to sequential propose() calls

        return cls(gen_fn, generate_batch_fn=gen_batch_fn)


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
