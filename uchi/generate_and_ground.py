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
    def answer(self, question: str, callback=None) -> str:
        # honesty gate 1: do we even know the question's concepts? (nonsense/OOV)
        if callback: callback("thinking", "Checking semantic vocabulary...")
        if self._known_fraction(question) < self.min_known:
            return _ABSTAIN

        if callback: callback("thinking", f"Retrieving top {self.retrieve_k} memories...")
        evidence = self.index.retrieve(question, self.retrieve_k)
        if not evidence or evidence[0][1] < self.min_sim:
            return _ABSTAIN
        ev_texts = [t for t, _ in evidence]

        # honesty gate 2: does the evidence actually ANSWER the question? (SQuAD-2.0
        # style unanswerability — topically relevant but no answer present)
        if self.answerability is not None:
            if callback: callback("thinking", "Evaluating answerability of evidence...")
            try:
                if self.answerability.prob(question, ev_texts[0]) < self.min_answerable:
                    return _ABSTAIN
            except Exception:
                pass

        # Try synthesis (neural decoder) first, then fall back to the grounded
        # extractive answer. We evaluate ALL candidates to perform Plural Voting (Simulation Engine).
        valid_candidates = []
        for candidate in self._candidates(question, evidence, ev_texts, callback=callback):
            if not candidate or not candidate.strip():
                continue
            if callback: callback("thinking", f"Oracle verifying candidate: '{candidate[:40]}...'")
            if self.oracle.is_grounded(candidate, ev_texts):
                valid_candidates.append(candidate)
            else:
                if callback: callback("prune", "Ungrounded claim pruned by Oracle.")
                
        if valid_candidates:
            # Plural vote: pick the most frequent verified candidate (Self-Consistency)
            from collections import Counter
            counts = Counter(valid_candidates)
            best_candidate = counts.most_common(1)[0][0]
            if callback: callback("reinforce", f"Plural vote winner! (votes: {counts[best_candidate]}/{len(valid_candidates)}).")
            return best_candidate
            
        # 3. Empirical Synthesis Loop (Fallback when text grounding fails)
        if self.proposer is not None:
            if callback: callback("thinking", "Text grounding failed. Initiating Empirical Synthesis Loop...")
            empirical_prompt = (f"The answer to the following question cannot be found in the text. "
                                f"Write a complete, standalone Python function named 'run' that takes no arguments and calculates or discovers the answer. "
                                f"Question: {question}\n\nOnly output the Python code.")
                                
            from .procedural_memory import REPLOracle
            repl = REPLOracle()
            
            for attempt in range(2 + 1):
                try:
                    code_cand = self.proposer.propose(empirical_prompt, [])
                except Exception:
                    break
                    
                if not code_cand: break
                    
                # Extract code block
                code_clean = code_cand
                if "```python" in code_clean: code_clean = code_clean.split("```python")[1].split("```")[0].strip()
                elif "```" in code_clean: code_clean = code_clean.split("```")[1].strip()
                
                if callback: callback("thinking", f"Executing empirical hypothesis in REPL (Attempt {attempt+1})...")
                passed, output = repl.execute(code_clean)
                
                if passed and output and not output.startswith("Error:"):
                    if callback: callback("reinforce", "Empirical hypothesis succeeded! Translating to human-readable format...")
                    translation_prompt = (f"You empirically proved the answer to the user's question via a Python script. "
                                          f"The script successfully calculated the following raw data: {output}\n\n"
                                          f"Question: {question}\n"
                                          f"Write a warm, conversational, human-readable explanation of the answer based on the output. Do NOT show the code.")
                    try:
                        final_answer = self.proposer.propose(translation_prompt, [])
                        if final_answer: return final_answer
                    except Exception:
                        pass
                    break
                else:
                    err_msg = output if output else "No output returned."
                    if callback: callback("prune", f"Empirical hypothesis failed: {err_msg[:40]}")
                    if attempt < 2:
                        if callback: callback("thinking", "Feeding traceback to FLUX for reflection...")
                        empirical_prompt += f"\n\n[Feedback]: Your python code failed to execute properly. Error/Output: {err_msg}. Please fix the logic and try again. Output ONLY python code."
                
        return _ABSTAIN

    def _candidates(self, question, evidence, ev_texts, callback=None, n_votes=3, max_reflections=2):
        # 1. the pluggable proposer (FLUX / LLM / decoder) — the strong generator
        if self.proposer is not None:
            # DSL Grid Planning Phase
            if callback: callback("thinking", "FLUX Proposer building DSL Grid (Scratchpad)...")
            dsl_grid = self.proposer.plan(question)
            base_prompt = question
            if dsl_grid:
                if callback: callback("thinking", f"DSL Grid established:\n{dsl_grid[:60]}...")
                base_prompt = f"Problem context (DSL Grid):\n{dsl_grid}\n\nQuestion: {question}"

            for i in range(n_votes):
                if callback: callback("thinking", f"FLUX Proposer generating candidate {i+1}/{n_votes} using DSL Grid...")
                current_prompt = base_prompt
                
                for attempt in range(max_reflections + 1):
                    try:
                        candidate = self.proposer.propose(current_prompt, ev_texts)
                    except Exception:
                        candidate = ""
                        
                    if not candidate or not candidate.strip():
                        break
                        
                    # Actor-Critic Reflection: Verify immediately to provide feedback
                    if self.oracle.is_grounded(candidate, ev_texts):
                        yield candidate
                        break  # Passed! Move to next plural vote
                    else:
                        if callback: callback("prune", f"Candidate {i+1} ungrounded (Attempt {attempt+1}).")
                        if attempt < max_reflections:
                            if callback: callback("thinking", "Providing feedback to FLUX for self-reflection...")
                            critique = (f"\n\n[Feedback]: Your previous answer '{candidate}' was rejected by the FactCheckOracle "
                                        f"because it hallucinates information not present in the context. "
                                        f"Please reflect on your mistake and generate a new, concise answer using ONLY the provided context.")
                            current_prompt += critique
                        else:
                            # Out of retries, yield the failed candidate so it can be formally pruned by the outer loop
                            yield candidate

        elif self.decoder is not None:                # backward-compat path
            if callback: callback("thinking", "Decoder generating candidate...")
            try:
                yield self.decoder.generate(question, ev_texts)
            except Exception:
                pass
        # 2. always keep the grounded extractive answer as a verified fallback
        if callback: callback("thinking", "Extracting verified fallback...")
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
