import concurrent.futures
import json
import re

class SwarmSynthesizer:
    """
    Flywheel #4: The Swarm Synthesizer (Map-Reduce)
    Decomposes complex questions into independent concepts,
    spins up parallel FLUX flywheels to solve them, and aggregates
    the empirically verified solutions into a single master answer.
    """
    def __init__(self, qa_pipeline):
        self.qa = qa_pipeline
        self.proposer = qa_pipeline.proposer
        # 0.4.0 Item 6 (closing the gap noted in the deliverables doc):
        # self-healing for delegation, not just tool calls. There's no
        # in-call retry loop here to guard (decomposition + dispatch +
        # aggregation happens once per answer() call) -- the failure mode
        # this guards is the SAME question being asked again after swarm
        # delegation already failed for it, and blindly repeating the
        # identical decompose -> dispatch -> aggregate sequence that
        # already didn't work. Fingerprinted on the question text, same
        # mechanism as the tool-call loop guard.
        from .loop_guard import LoopGuard
        self.loop_guard = LoopGuard()

    def _decompose(self, question: str) -> list[str]:
        if not self.proposer: return [question]
        
        prompt = (
            "Break the following complex question into a JSON list of 1 to 3 independent sub-questions "
            "that must be solved to synthesize the final answer. Keep them simple.\n"
            f"Question: {question}\n\n"
            "Output ONLY a valid JSON array of strings, e.g. [\"sub q 1\", \"sub q 2\"]"
        )
        
        try:
            raw = self.proposer.propose(prompt, [])
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                sub_qs = json.loads(match.group(0))
                if isinstance(sub_qs, list) and len(sub_qs) > 0:
                    # Filter out the original question if it just echoed it
                    sub_qs = [q for q in sub_qs if len(q) > 5 and q != question]
                    if sub_qs:
                        return sub_qs[:3]
        except Exception:
            pass
            
        return [question]

    def answer(self, question: str, callback=None) -> str:
        if callback: callback("thinking", "Swarm Orchestrator analyzing problem complexity...")

        # 0.4.0 Item 9 (IQ Task Router): a cheap heuristic pre-check gates
        # the FLUX round-trip in _decompose() — most questions are atomic
        # lookups and shouldn't pay for a decomposition attempt at all.
        from .iq_router import should_decompose
        if not should_decompose(question):
            if callback: callback("thinking", "IQ router: low complexity, skipping decomposition.")
            return self.qa.answer(question, callback=callback)

        # Self-healing: this exact question already failed swarm delegation
        # before -- don't repeat the identical decompose/dispatch/aggregate
        # sequence (including the FLUX round-trip _decompose() itself would
        # spend), fall straight to the single pipeline instead. Checked
        # BEFORE _decompose() runs, not after, so a repeat doesn't even pay
        # for the decomposition attempt it already knows fails.
        if self.loop_guard.is_penalized(question):
            n = self.loop_guard.failure_count(question)
            if callback:
                callback(
                    "prune",
                    f"Swarm delegation for this question already failed {n} time(s) in a row — "
                    f"falling back to single pipeline instead of repeating it.",
                )
            return self.qa.answer(question, callback=callback)

        sub_questions = self._decompose(question)

        if len(sub_questions) <= 1:
            if callback: callback("thinking", "Problem is atomic. Running single pipeline...")
            return self.qa.answer(question, callback=callback)

        if callback:
            sub_list = "\n".join(f"  - {q}" for q in sub_questions)
            callback("thinking", f"Swarm launched for {len(sub_questions)} independent concepts:\n{sub_list}")
        
        sub_answers = []
        # Run sub-questions in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(sub_questions)) as executor:
            def solve_sub(sq):
                # Suppress deep callbacks to prevent interleaving TUI garbage
                return sq, self.qa.answer(sq, callback=None)
                
            futures = [executor.submit(solve_sub, sq) for sq in sub_questions]
            for future in concurrent.futures.as_completed(futures):
                try:
                    sq, ans = future.result()
                    if ans and "don't have grounded knowledge" not in ans:
                        sub_answers.append(f"Verified Sub-Answer ({sq}): {ans}")
                        if callback: callback("reinforce", f"Swarm Node solved concept: '{sq[:30]}...'")
                    else:
                        if callback: callback("prune", f"Swarm Node failed concept: '{sq[:30]}...'")
                except Exception:
                    pass
                    
        if not sub_answers:
            self.loop_guard.record_failure(question)
            if callback: callback("prune", "All swarm nodes failed. Falling back to single pipeline...")
            return self.qa.answer(question, callback=callback)
            
        if callback: callback("thinking", "Swarm complete. Aggregating sub-answers into final synthesis...")
        context_block = "\n".join(sub_answers)
        
        agg_prompt = (
            f"Here are the independently verified solutions to the sub-components of the user's question:\n\n"
            f"{context_block}\n\n"
            f"Original Question: {question}\n\n"
            f"Using ONLY the proven sub-answers above, synthesize a complete, human-readable final answer."
        )
        
        try:
            final_answer = self.proposer.propose(agg_prompt, [])
            if final_answer:
                self.loop_guard.record_success(question)
                return final_answer
        except Exception:
            pass

        self.loop_guard.record_failure(question)
        return "I am sorry, the swarm failed to aggregate a final answer."
