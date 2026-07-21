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
                 min_known: float = 0.5, min_answerable: float = 0.5,
                 web_search_enabled: bool = False, task_config_cache=None,
                 proprioception=None, proprioception_model=None, proprioception_tokenizer=None) -> None:
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
        # 0.4.0 Item 16.2 ("Lazy" World Knowledge): when local evidence
        # retrieval comes up empty, silently fall back to a live web
        # search, learn what it finds, and retry retrieval once — before
        # abstaining, not instead of grounding.
        self.web_search_enabled = web_search_enabled
        # Dynamic-N self-consistency voting (follow-on to Item 17): recalls
        # a recommended vote count keyed by the question's structural shape
        # (TaskConfigCache, ODUSP-backed), falling back to a complexity-
        # score-derived baseline when unfitted or unconfident. None by
        # default -- n_votes stays a fixed 3 unless this is supplied,
        # zero behavior change for anyone not using it.
        self.task_config_cache = task_config_cache
        # Proprioception (experimental, additive-only): FLUX's own sense of
        # whether a question is familiar, checked BEFORE generation. Never
        # blocks or reduces anything -- an "unfamiliar" verdict can only
        # increase n_votes (spend more self-consistency compute), the same
        # direction as scaling up for a hard question, never a reason to
        # answer with LESS scrutiny. Requires all three of proprioception/
        # proprioception_model/proprioception_tokenizer; None by default,
        # zero behavior change for anyone not using it.
        self.proprioception = proprioception
        self.proprioception_model = proprioception_model
        self.proprioception_tokenizer = proprioception_tokenizer

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
        # A question that asserts nothing specific (no proper noun, no
        # number -- "Hello!", "Thanks!", "How are you?") isn't a factual
        # lookup in the first place. The two honesty gates below exist to
        # catch nonsense/OOV questions and weak-evidence factual questions
        # -- applied to a bare greeting, they were abstaining on it the
        # same way they'd abstain on a genuine unanswerable question,
        # because retrieval naturally can't find high-similarity matches
        # for something that isn't a factual query. Gated on specificity:
        # a real factual question with weak evidence still abstains
        # exactly as before; a generic conversational one proceeds to
        # candidate generation, where the oracle's own no-evidence
        # relaxation (oracle.py) makes the final call.
        question_is_specific = bool(self.oracle._specific_terms(question))

        # honesty gate 1: do we even know the question's concepts? (nonsense/OOV)
        if callback: callback("thinking", "Checking semantic vocabulary...")
        if question_is_specific and self._known_fraction(question) < self.min_known:
            return _ABSTAIN

        if callback: callback("thinking", f"Retrieving top {self.retrieve_k} memories...")
        evidence = self.index.retrieve(question, self.retrieve_k)
        if question_is_specific and (not evidence or evidence[0][1] < self.min_sim) and self.web_search_enabled:
            if callback: callback("thinking", "No local evidence — falling back to web search...")
            try:
                from .web_search import perform_web_search
                web_text = perform_web_search(question)
            except Exception:
                web_text = ""
            if web_text:
                self.index.build_from_corpus(web_text)
                evidence = self.index.retrieve(question, self.retrieve_k)
        if question_is_specific and (not evidence or evidence[0][1] < self.min_sim):
            return _ABSTAIN
        # Only carry forward evidence confident enough to actually mean
        # something -- a weak/irrelevant match (below min_sim) shouldn't
        # count as "evidence" for the oracle check either, or a generic
        # reply would get strictly checked against irrelevant retrieved
        # text instead of correctly hitting the oracle's no-evidence
        # relaxation (this is what was happening for "Hello!": retrieval
        # found real but irrelevant passages, so ev_texts was non-empty
        # and the strict check applied to a candidate that never asserted
        # anything those passages could support in the first place).
        ev_texts = [t for t, sim in evidence if sim >= self.min_sim]

        # honesty gate 2: does the evidence actually ANSWER the question? (SQuAD-2.0
        # style unanswerability — topically relevant but no answer present)
        if self.answerability is not None:
            if callback: callback("thinking", "Evaluating answerability of evidence...")
            try:
                if self.answerability.prob(question, ev_texts[0]) < self.min_answerable:
                    return _ABSTAIN
            except Exception:
                pass

        # Dynamic-N: how many self-consistency votes this question actually
        # gets, instead of a fixed 3. Baseline from the same complexity
        # heuristic already used to gate swarm decomposition; refined by
        # TaskConfigCache's recall if it has a confident one for this
        # question's structural shape. This only changes how many
        # candidates get generated -- every candidate still goes through
        # the full, unchanged oracle cascade below, so this is a compute
        # budget knob, not a correctness gate: it can move in either
        # direction safely.
        from .iq_router import estimate_complexity
        complexity = estimate_complexity(question)
        n_votes = 1 if complexity < 0.3 else (3 if complexity < 0.6 else 5)
        if self.task_config_cache is not None:
            recalled_n, conf = self.task_config_cache.recall_n(question)
            if recalled_n is not None and conf >= 0.5:
                n_votes = recalled_n

        # Proprioception: if FLUX itself flags this question as unfamiliar
        # (far from its own training distribution), that's a reason to
        # spend MORE self-consistency votes, never fewer -- additive-only,
        # same direction as every other signal that touches n_votes. This
        # can only raise n_votes above whatever complexity/TaskConfigCache
        # already decided, never lower it. Silently a no-op unless all
        # three components (proprioception, model, tokenizer) are present.
        if self.proprioception is not None and self.proprioception_model is not None:
            try:
                if self.proprioception.is_unfamiliar(
                    self.proprioception_model, self.proprioception_tokenizer, question
                ):
                    n_votes = max(n_votes, 8)
            except Exception:
                pass  # fail open -- additive signal, never the sole gate

        # Try synthesis (neural decoder) first, then fall back to the grounded
        # extractive answer. We evaluate ALL candidates to perform Plural Voting (Simulation Engine).
        #
        # Real bug found (0.5.0 Item 9 investigation): `_candidates()` always
        # yields `self._extractive(question, evidence)` as an unconditional
        # FINAL candidate (guaranteed by its own code structure -- the "2.
        # always keep..." step runs unconditionally after the proposer/
        # decoder branch, every call). Because it IS a retrieved passage, it
        # trivially passes `is_grounded()` (support against itself is 1.0)
        # regardless of whether that passage actually answers the question
        # -- e.g. a "New France" (the historical Quebec colony) passage
        # confidently "answering" a "capital of France" (the country)
        # question, because both share the literal word "france". When
        # every genuinely generated candidate fails grounding (the common
        # case against an undertrained proposer) and the extractive
        # fallback is the ONLY thing that passes, no real synthesis was
        # ever verified -- silently returning it anyway is exactly the
        # confabulation this system's own stated principle says to avoid.
        #
        # Tracked by POSITION, not string equality: a real proposer/decoder
        # candidate can legitimately be identical text to the extractive
        # passage (e.g. a simple factual lookup correctly quoting the one
        # piece of evidence verbatim -- a real prior test already covers
        # exactly this: a fake proposer configured to answer correctly).
        # `_candidates()`'s structure guarantees its LAST yield is always
        # the extractive fallback and nothing else can come after it, so
        # only that final slot is exempt from counting as "synthesized";
        # every earlier slot is a genuine proposer/decoder attempt. When no
        # proposer/decoder is configured at all (pure extractive mode, a
        # fully legitimate configuration per this class's own docstring),
        # exactly one candidate is ever yielded and this whole check is
        # skipped -- there was nothing to "synthesize" in the first place.
        raw_candidates = list(self._candidates(question, evidence, ev_texts, callback=callback, n_votes=n_votes))
        valid_candidates = []
        synthesized_passed = len(raw_candidates) <= 1
        for i, candidate in enumerate(raw_candidates):
            if not candidate or not candidate.strip():
                continue
            if callback: callback("thinking", f"Oracle verifying candidate: '{candidate[:40]}...'")
            if self.oracle.is_grounded(candidate, ev_texts):
                valid_candidates.append(candidate)
                if i < len(raw_candidates) - 1:
                    synthesized_passed = True
            else:
                if callback: callback("prune", "Ungrounded claim pruned by Oracle.")

        if valid_candidates and not synthesized_passed:
            if callback: callback("prune", "Only the raw extractive fallback passed grounding -- "
                                           "no verified synthesis, treating as unresolved rather than answering.")
            valid_candidates = []

        if valid_candidates:
            # Plural vote: pick the most frequent verified candidate (Self-Consistency)
            from collections import Counter
            counts = Counter(valid_candidates)
            best_candidate = counts.most_common(1)[0][0]
            if callback: callback("reinforce", f"Plural vote winner! (votes: {counts[best_candidate]}/{len(valid_candidates)}).")
            if self.task_config_cache is not None:
                # Retrospective, honest signal: unanimous agreement across
                # every valid candidate suggests this was easy enough that
                # fewer votes would likely have sufficed; any real
                # disagreement means the full budget was genuinely used.
                actual_n = 1 if len(set(valid_candidates)) == 1 else n_votes
                self.task_config_cache.record_outcome(question, actual_n)
            return best_candidate
            
        # 3. Empirical Synthesis Loop (Fallback when text grounding fails)
        if self.proposer is not None:
            if callback: callback("thinking", "Text grounding failed. Initiating Empirical Synthesis Loop...")
            empirical_prompt = (f"The answer to the following question cannot be found in the text. "
                                f"Write a complete, standalone Python function named 'run' that takes no arguments and calculates or discovers the answer. "
                                f"CRITICAL: You MUST include at least 3 `assert` statements at the bottom of your script to test "
                                f"the logic and constraints of the problem before returning the final result.\n"
                                f"Question: {question}\n\nOnly output the Python code.")
                                
            from .code_engine import REPLOracle
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

            # 0.5.0 Item 7's last bullet ("dynamic-N doesn't cost N× latency"):
            # every branch's FIRST attempt uses the identical base_prompt (only
            # a later Devil's Advocate/oracle-rejection reflection diverges
            # per-branch), so that round -- and only that round -- can be
            # generated for all n_votes branches in one vectorized call
            # instead of n_votes sequential propose() calls. Falls back to
            # None (the existing sequential path, unchanged) if the proposer
            # has no batch path or the batch call itself fails; the reflection
            # loop below is untouched either way.
            first_round = None
            if n_votes > 1:
                try:
                    first_round = self.proposer.propose_batch(base_prompt, ev_texts, n_votes, think=True)
                except Exception:
                    first_round = None

            for i in range(n_votes):
                if callback: callback("thinking", f"FLUX Proposer generating candidate {i+1}/{n_votes} using DSL Grid...")
                current_prompt = base_prompt

                for attempt in range(max_reflections + 1):
                    if attempt == 0 and first_round is not None:
                        candidate = first_round[i]
                    else:
                        try:
                            # think=True elicits CoT's trained reasoning-before-answering
                            # so production candidates surface the full trace, not just
                            # a bare answer. (Devil's Advocate/reflection prompts below
                            # stay plain — they ask for a short critique, not a proof.)
                            candidate = self.proposer.propose(current_prompt, ev_texts, think=True)
                        except Exception:
                            candidate = ""

                    if not candidate or not candidate.strip():
                        break

                    # Surface FLUX's full internal monologue (the actual <|think|>
                    # trace), distinct from the "thinking" status-update events —
                    # this is the literal reasoning content, not meta-commentary.
                    if callback: callback("reasoning", candidate)

                    # Actor-Critic Reflection: Verify immediately to provide feedback
                    if self.oracle.is_grounded(candidate, ev_texts):
                        # Cross-Examination Flywheel (Multi-Agent Debate)
                        if callback: callback("thinking", "Candidate passed Oracle. Initiating Cross-Examination (Devil's Advocate)...")
                        advocate_prompt = (
                            f"Review the following question and proposed answer. Act as a ruthless Devil's Advocate. "
                            f"Identify any logical flaws, assumptions, or gaps in reasoning. "
                            f"If the answer is perfectly logical and sound, output EXACTLY 'PASS'. "
                            f"Otherwise, output your critique.\n\n"
                            f"Question: {current_prompt}\nAnswer: {candidate}"
                        )
                        try:
                            critique = self.proposer.propose(advocate_prompt, [])
                        except Exception:
                            critique = "PASS"
                            
                        if "PASS" in critique or not critique.strip():
                            yield candidate
                            break  # Passed debate!
                        else:
                            if callback: callback("critique", critique)
                            if callback: callback("prune", "Devil's Advocate found a logical flaw. Forcing reflection...")
                            if attempt < max_reflections:
                                if callback: callback("thinking", "Feeding Devil's Advocate critique back to FLUX...")
                                current_prompt += f"\n\n[Devil's Advocate Critique]: {critique}\nReflect on this critique and generate a logically flawless answer."
                            else:
                                yield candidate
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
