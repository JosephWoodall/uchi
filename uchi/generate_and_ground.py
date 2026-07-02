"""
generate_and_ground.py — Uchi's primary answering loop.

    ask(question) →
        retrieve evidence from the brain (SemanticIndex)
        → [trie fast-path: confident exact recall, if available]
        → GENERATE a candidate answer  (NeuralDecoder if trained, else extractive)
        → FACT-CHECK the candidate against the evidence (FactCheckOracle)
        → emit if grounded, else ABSTAIN

Generalisation on out-of-distribution questions comes from the generator composing
answers over retrieved knowledge; honesty comes from the oracle vetoing anything
the brain does not support. Uchi never confabulates: when it cannot ground an
answer it says so.
"""
from __future__ import annotations

import re
from typing import Optional

from .oracle import FactCheckOracle
from .retrieval import SemanticIndex

_WORD = re.compile(r"[a-z0-9']+")
_STOP = frozenset(
    "the a an of to in and or is are was were be what which who how why when where "
    "do does did can could would should this that it its there".split()
)
_ABSTAIN = "I don't have grounded knowledge to answer that."


class GenerateAndGround:
    """The retrieve → generate → fact-check → emit/abstain loop.

    Parameters
    ----------
    index : SemanticIndex
        Retrieval over the brain corpus.
    oracle : FactCheckOracle, optional
        Honesty gate. Defaults to a fresh ``FactCheckOracle()``.
    decoder : NeuralDecoder, optional
        The generator. When ``None`` the loop falls back to *extractive*
        generation (return the retrieved sentence that best answers the
        question) — still grammatical and grounded, just not synthesised.
    predictor : optional
        The trie (``router.predictor``) for the exact-recall fast path.
    """

    def __init__(self, index: SemanticIndex, oracle: Optional[FactCheckOracle] = None,
                 decoder=None, proposer=None, predictor=None, answerability=None,
                 retrieve_k: int = 10, min_sim: float = 0.5,
                 min_known: float = 0.5, min_answerable: float = 0.5) -> None:
        self.index = index
        self.oracle = oracle or FactCheckOracle()
        # `proposer` is the pluggable generator (decoder / FLUX / LLM); `decoder`
        # is kept for backward compatibility and wrapped if no proposer is given.
        self.proposer = proposer
        self.decoder = decoder
        self.predictor = predictor
        self.answerability = answerability     # AnswerabilityChecker or None
        self.retrieve_k = retrieve_k
        self.min_sim = min_sim
        self.min_known = min_known
        self.min_answerable = min_answerable

    # ── helpers ────────────────────────────────────────────────────────────────
    def _content(self, text: str) -> list[str]:
        return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2]

    def _known_fraction(self, question: str) -> float:
        content = self._content(question)
        if not content:
            return 0.0
        known = sum(1 for w in content if w in self.index.w2i)
        return known / len(content)

    def _extractive(self, question: str, evidence: list[tuple[str, float]]) -> Optional[str]:
        """Pick the retrieved sentence that best answers the question."""
        qwords = {w for w in self._content(question)}
        best, best_score = None, -1.0
        for text, sim in evidence:
            tw = {w for w in _WORD.findall(text.lower()) if w not in _STOP}
            overlap = len(qwords & tw) / (len(qwords) + 1)
            score = 0.6 * sim + 0.4 * overlap
            if score > best_score:
                best_score, best = score, text
        return best

    # ── the loop ───────────────────────────────────────────────────────────────
    def answer(self, question: str) -> str:
        # honesty gate 1: do we even know the question's concepts? (nonsense/OOV)
        if self._known_fraction(question) < self.min_known:
            return _ABSTAIN

        evidence = self.index.retrieve(question, self.retrieve_k)
        if not evidence or evidence[0][1] < self.min_sim:
            return _ABSTAIN
        ev_texts = [t for t, _ in evidence]

        # honesty gate 2: does the evidence actually ANSWER the question? (SQuAD-2.0
        # style unanswerability — topically relevant but no answer present)
        if self.answerability is not None:
            try:
                if self.answerability.prob(question, ev_texts[0]) < self.min_answerable:
                    return _ABSTAIN
            except Exception:
                pass

        # Try synthesis (neural decoder) first, then fall back to the grounded
        # extractive answer. Emit the first candidate the oracle finds grounded;
        # abstain only if neither is supported. Synthesis when it works,
        # grounded extraction otherwise — never confabulate.
        for candidate in self._candidates(question, evidence, ev_texts):
            if candidate and candidate.strip() and self.oracle.is_grounded(candidate, ev_texts):
                return candidate
        return _ABSTAIN

    def _candidates(self, question, evidence, ev_texts):
        # 1. the pluggable proposer (FLUX / LLM / decoder) — the strong generator
        if self.proposer is not None:
            try:
                yield self.proposer.propose(question, ev_texts)
            except Exception:
                pass
        elif self.decoder is not None:                # backward-compat path
            try:
                yield self.decoder.generate(question, ev_texts)
            except Exception:
                pass
        # 2. always keep the grounded extractive answer as a verified fallback
        yield self._extractive(question, evidence)

    def answer_verbose(self, question: str) -> dict:
        """Same as answer() but returns provenance (for debugging/benchmarks)."""
        known = self._known_fraction(question)
        if known < self.min_known:
            return {"answer": _ABSTAIN, "abstained": True, "reason": f"unknown-terms({known:.2f})"}
        ev = self.index.retrieve(question, self.retrieve_k)
        if not ev or ev[0][1] < self.min_sim:
            return {"answer": _ABSTAIN, "abstained": True,
                    "reason": f"weak-retrieval({ev[0][1] if ev else 0:.2f})"}
        ev_texts = [t for t, _ in ev]
        for cand in self._candidates(question, ev, ev_texts):
            if not cand or not cand.strip():
                continue
            support = self.oracle.support(cand, ev_texts)
            if support >= self.oracle.min_support:
                src = "decoder" if (self.decoder and cand != self._extractive(question, ev)) else "extractive"
                return {"answer": cand, "abstained": False, "support": support,
                        "source": src, "top_sim": ev[0][1], "evidence": ev_texts[:3]}
        return {"answer": _ABSTAIN, "abstained": True, "reason": "unsupported"}
