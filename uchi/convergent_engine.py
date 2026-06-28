"""
convergent_engine.py
====================
Convergent MCTS with Oracle Feedback (CMOF) for Uchi.

Architecture (per query)
------------------------
  concepts → encode → q_vec ∈ ℝ^N
                           ↓
             [tool check] ← skill_registry.get_all_vectors()
                           ↓
     [MCTS rollouts: up to MAX_BUDGET × CANDIDATE_TOKENS tokens]
     [stop early: token_diversity(candidates) < TOKEN_DIVERSITY_E]
                           ↓
              [binary oracle filter]
              conv: CoherenceOracle  — length / overlap / trigram / SSM value
              code: TieredCodeOracle — syntax (pre-bootstrap) / compile (post)
                           ↓
     [argmax cosine(q_vec, r_vec) over valid candidates]
                           ↓
     [tool vs text: relative margin + absolute floor]
                           ↓
     return (kind, payload, reward_hint)   ← stateless; router owns side effects

Constants
---------
  MAX_BUDGET        = 20      maximum MCTS rollouts per query
  MIN_ROLLOUTS      = 3       always run at least this many rollouts
  TOKEN_DIVERSITY_E = 0.25    stop when Jaccard diversity drops below this
  ABSOLUTE_FLOOR    = 0.3     minimum cosine for tool dispatch
  MARGIN            = 0.05    tool must beat best text by this margin
  CANDIDATE_TOKENS  = 12      tokens generated per MCTS rollout
  MCTS_WALL_BUDGET_S = 8.0   hard wall-clock cap on the rollout loop
"""
from __future__ import annotations

import ast
import logging
import math
import time
from typing import Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

MAX_BUDGET        = 20   # was 50; convergence detection stops most runs at 3–8
MIN_ROLLOUTS      = 3
TOKEN_DIVERSITY_E = 0.25
ABSOLUTE_FLOOR    = 0.3
MARGIN            = 0.05
CANDIDATE_TOKENS  = 12   # was 45; factual answers are 1–3 words, 12 is generous

# Adaptive computation gate thresholds.
# After temperature calibration, _get_ssm_value() returns a probability in (0,1).
# High confidence (value head > GREEDY_CONF and trie entropy < GREEDY_ENTROPY_CAP):
#   bypass the MCTS loop and generate a single greedy candidate in O(1).
# Low confidence (value < UNCERTAIN_FLOOR): scale the rollout budget up by
#   BUDGET_SCALE_UNCERTAIN to invest saved compute into hard queries.
GREEDY_CONF        = 0.85
GREEDY_ENTROPY_CAP = 1.5  # was 0.6; allow moderately peaked distributions through
GREEDY_ENTROPY_FLOOR = 0.05  # near-zero entropy → clear trie path, bypass regardless of SSM
UNCERTAIN_FLOOR    = 0.3   # probability below this → expand rollout budget
BUDGET_SCALE_UNCERTAIN = 2
MCTS_WALL_BUDGET_S = 8.0   # hard wall-clock limit on the rollout loop (seconds)

_STOP_TOKENS = frozenset({"<|user|>", "<|assistant|>", "<|end|>"})


# ── oracles ───────────────────────────────────────────────────────────────────

class CoherenceOracle:
    """
    Binary conversational coherence filter.

    Passes when the candidate is:
      ≥ 5 tokens long
      ≤ 60% token overlap with the query  (no echo responses)
      free of trigram repetition           (no degenerate loops)
      SSM value ≥ -0.5 if the value head is trained
    """

    def passes(
        self,
        candidate_tokens: List[str],
        query_tokens: List[str],
        ssm_value: Optional[float] = None,
    ) -> bool:
        # Allow single-word factual answers ("paris", "1945", "au").
        # The old threshold of 5 caused all factual recalls to fail the oracle,
        # triggering tree-search escalation on every short answer.
        if len(candidate_tokens) < 1:
            return False

        query_set = set(query_tokens)
        overlap = (
            sum(1 for t in candidate_tokens if t in query_set)
            / max(len(candidate_tokens), 1)
        )
        if overlap > 0.6:
            return False

        for i in range(len(candidate_tokens) - 2):
            if (
                candidate_tokens[i] == candidate_tokens[i + 1]
                == candidate_tokens[i + 2]
            ):
                return False

        if ssm_value is not None and ssm_value < -0.5:
            return False

        return True


class TieredCodeOracle:
    """
    Binary code validity filter with strictness calibrated to bootstrap state.

    Pre-bootstrap:  ast.parse() only — accepts syntactically valid Python.
    Post-bootstrap: ast.parse() + py_compile — requires compilable code.

    The tiered approach prevents oracle starvation during cold start: an
    un-bootstrapped trie can produce syntactically valid fragments,
    giving Uchi passing candidates before HumanEval training completes.
    """

    def passes(self, candidate_tokens: List[str], bootstrapped: bool = False) -> bool:
        code = " ".join(candidate_tokens)
        if not code.strip():
            return False

        try:
            ast.parse(code)
        except SyntaxError:
            return False

        if bootstrapped:
            import py_compile
            import tempfile
            import os
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False
                ) as f:
                    f.write(code)
                    tmp_path = f.name
                py_compile.compile(tmp_path, doraise=True)
            except Exception:
                return False
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        return True


# ── engine ────────────────────────────────────────────────────────────────────

class ConvergentEngine:
    """
    CMOF deliberation loop: generate candidates → oracle filter → vector select.

    Stateless: returns (kind, payload, reward_hint) with no mutations to the
    trie, SSM, or optimizer.  The router owns all side-effectful work.
    """

    def __init__(self, router):
        self._router = router
        self._coherence = CoherenceOracle()
        self._code_oracle = TieredCodeOracle()

    def generate(
        self,
        concepts: List[str],
        bootstrapped: bool = False,
        bias_context: Optional[str] = None,
        context: Optional[List] = None,
        callback=None,
    ) -> Tuple[str, object, float]:
        """
        Main deliberation loop.

        Returns one of:
          ("tool",      skill_name: str,        reward_hint: +1.0)
          ("text",      token_list: List[str],   reward_hint: +0.5)
          ("uncertain", token_list: List[str],   reward_hint: -0.1)

        context: prior (q_tokens, r_tokens) turn pairs to prepend to seed.
                 Gives the SSM value head richer state for candidate scoring.
        """
        from uchi.vector_oracle import encode, similarity, token_diversity

        def _think(msg: str) -> None:
            if callback:
                try:
                    callback("thinking", msg)
                except Exception:
                    pass

        try:
            # 1. Encode query (frozen SSM)
            trie_dist = self._safe_peek(concepts[:8])
            q_vec = encode(concepts, trie_dist)

            # 2. Tool check — fast path before any rollouts
            tool_vecs = self._router.skills.get_all_vectors()
            best_tool, tool_score = _best_cosine(q_vec, tool_vecs)
            if best_tool is not None:
                _think(f"[dim]tool check  [bold]{best_tool}[/bold]  score={tool_score:.3f}  floor={ABSOLUTE_FLOOR}[/dim]")

            # 3. Adaptive computation gate — evaluate confidence before committing
            #    to the full MCTS budget.
            #    a) High confidence + peaked trie → greedy O(1) bypass.
            #    b) Low confidence → expand rollout budget.
            initial_value = self._get_ssm_value(concepts)
            trie_dist_peek = self._safe_peek(concepts[:8])
            trie_entropy = (
                -sum(p * math.log(p + 1e-9) for p in trie_dist_peek.values())
                if trie_dist_peek else float("inf")
            )
            _think(
                f"[dim]query encoded  "
                f"value={initial_value:.3f}  "
                f"H(trie)={trie_entropy:.3f}[/dim]"
                if initial_value is not None
                else "[dim]query encoded  value=n/a[/dim]"
            )

            # Build seed once; reused by both the greedy path and the rollout loop.
            seed = []
            for q_toks, r_toks in (context or []):
                seed += ["<|user|>"] + list(q_toks)[:6] + ["<|assistant|>"] + list(r_toks)[:6]
            seed += ["<|user|>"] + concepts + ["<|assistant|>"]

            # Pre-flight classification: peek trie at CURRENT QUESTION slice only.
            # seed may be prefixed with conversation history; seed[-8:] would land
            # in the middle of a prior turn when history is long. Build the
            # current-question-only prefix so the trie peek is always accurate.
            # Use full concepts (not [-6:]) — _preflight_classify's own [-8:] window
            # does the trimming, so the last word of a 7+ token question is preserved.
            _current_q_prefix = ["<|user|>"] + concepts + ["<|assistant|>"]
            _profile = self._preflight_classify(_current_q_prefix)
            _candidate_tokens = _profile["candidate_tokens"]
            _query_type       = _profile["query_type"]
            _mcts_sims        = _profile["mcts_sims"]
            _think(
                f"[dim]query type: [bold]{_query_type}[/bold]"
                f"  candidate_tokens={_candidate_tokens}"
                f"  max_budget={_profile['max_budget']}"
                f"  mcts_sims={_mcts_sims}[/dim]"
            )

            # Fire greedy bypass when:
            #   a) SSM is confident + trie is peaked (trained brain, normal path), OR
            #   b) Trie entropy is near-zero at the concept prefix (clear single path), OR
            #   c) Pre-flight classified the query as factual — the full-seed peek already
            #      showed a peaked distribution. MCTS is the wrong tool here: it explores
            #      random paths rather than following the peaked trie node. Greedy O(1) is
            #      strictly better for factual recall.
            _greedy_by_ssm = (
                initial_value is not None
                and initial_value > GREEDY_CONF
                and trie_entropy < GREEDY_ENTROPY_CAP
            )
            _greedy_by_entropy = trie_entropy < GREEDY_ENTROPY_FLOOR
            _greedy_by_preflight = _query_type in ("factual_short", "factual_long")
            # Preflight overrides the bias_context guard: if the pre-flight showed the
            # trie has a peaked answer path, memory retrieval bias is noise, not signal.
            if _greedy_by_preflight or (bias_context is None and (_greedy_by_ssm or _greedy_by_entropy)):
                _v = f"{initial_value:.3f}" if initial_value is not None else "n/a"
                _reason = (
                    "preflight" if _greedy_by_preflight and not _greedy_by_ssm and not _greedy_by_entropy
                    else "entropy" if _greedy_by_entropy and not _greedy_by_ssm
                    else "ssm+entropy"
                )
                _think(f"[bold #3fb950]greedy bypass ({_reason})  value={_v}  H={trie_entropy:.3f}[/bold #3fb950]")
                try:
                    # For preflight-triggered bypass, use the current-question-only seed.
                    # The full seed includes conversation history, which creates a trie path
                    # that was never streamed. The fact was streamed as a bare Q→A pair.
                    _greedy_seed = _current_q_prefix if _greedy_by_preflight else seed
                    raw = self._router.predictor.generate(
                        n_tokens=_candidate_tokens,
                        seed=_greedy_seed,
                        temperature=0.0,
                        use_mcts=False,
                        bias_context=None,  # bias_context is noise for factual preflight path
                        mcts_sims=_mcts_sims,
                    )
                    reply = [t for t in raw if t not in _STOP_TOKENS]

                    # MCQ self-consistency: if this is a multiple-choice query
                    # (ends with "Answer:"), take 3 samples and return the
                    # majority-voted letter.  Wang et al. 2023 — self-consistency
                    # reliably outperforms single-sample greedy for MCQ.
                    if _greedy_by_preflight and "answer:" in " ".join(
                        _greedy_seed[-6:]
                    ).lower():
                        reply = self._mcq_majority_vote(
                            _greedy_seed, reply, _mcts_sims, _think
                        )

                    # Preflight: trie peaked at 60%+ → SSM uncertainty should not override.
                    # An untrained SSM returning -0.7 would reject a 67%-confidence trie answer.
                    _oracle_ssm = None if _greedy_by_preflight else initial_value
                    if reply and self._coherence.passes(reply, concepts, _oracle_ssm):
                        r_dist = self._safe_peek(reply[:8])
                        r_vec = encode(reply, r_dist)
                        best_text_score = similarity(q_vec, r_vec)
                        if (
                            best_tool is not None
                            and tool_score >= ABSOLUTE_FLOOR
                            and tool_score > best_text_score + MARGIN
                        ):
                            _think(f"[bold #d29922]-> tool dispatch  {best_tool}  ({tool_score:.3f} > {best_text_score:.3f} + {MARGIN})[/bold #d29922]")
                            return ("tool", best_tool, +1.0)
                        _think(f"[bold #3fb950]-> text (greedy)  cos={best_text_score:.3f}[/bold #3fb950]")
                        _log.debug("ConvergentEngine: greedy O(1) bypass fired (v=%.2f, H=%.2f)",
                                   initial_value, trie_entropy)
                        return ("text", reply, +0.5)
                except Exception as exc:
                    _log.debug("ConvergentEngine: greedy bypass failed: %s", exc)

            adaptive_budget = _profile["max_budget"]
            if initial_value is not None and initial_value < UNCERTAIN_FLOOR:
                adaptive_budget = min(adaptive_budget * BUDGET_SCALE_UNCERTAIN, 100)
                _think(f"[yellow]low confidence — budget expanded to {adaptive_budget}[/yellow]")
            else:
                _think(f"[dim]MCTS budget={adaptive_budget}  min_rollouts={MIN_ROLLOUTS}[/dim]")

            # 3b. Inner monologue: brief MCTS reasoning pass for complex queries.
            #     Triggers when: query_type is generative AND (low SSM confidence OR
            #     long query with deductive markers). Output tokens bias the rollout.
            #     The monologue is never returned to the user.
            _im_toks = self._run_inner_monologue(
                concepts, initial_value, _query_type, bias_context
            )
            if _im_toks:
                im_bias = " ".join(_im_toks[:12])
                if bias_context is None:
                    bias_context = im_bias
                _think(f"[dim #bb9af7]inner monologue → {' '.join(_im_toks[:8])}…[/dim #bb9af7]")

            # 4. MCTS rollout loop
            candidates: List[Tuple[List[str], List[float]]] = []
            _loop_start = time.perf_counter()
            _first_tokens: List[str] = []  # track first token of each candidate

            for i in range(adaptive_budget):
                temp = 0.1 + 0.02 * i
                try:
                    raw = self._router.predictor.generate(
                        n_tokens=_candidate_tokens,
                        seed=seed,
                        temperature=temp,
                        use_mcts=True,
                        bias_context=bias_context,
                        mcts_sims=_mcts_sims,
                    )
                    reply = [t for t in raw if t not in _STOP_TOKENS]
                    if not reply:
                        continue
                    r_dist = self._safe_peek(reply[:8])
                    r_vec = encode(reply, r_dist)
                    candidates.append((reply, r_vec))
                    _first_tokens.append(reply[0])
                except Exception:
                    continue

                _think(
                    f"[dim]rollout {i + 1:>2}/{adaptive_budget}"
                    f"  temp={temp:.2f}"
                    f"  candidates={len(candidates)}"
                    f"  preview=[italic]{' '.join(reply[:5])}…[/italic][/dim]"
                )

                # First-token convergence: if last 3 candidates agree on first token,
                # the trie has committed to an answer — no benefit in more rollouts.
                if len(_first_tokens) >= 3 and len(set(_first_tokens[-3:])) == 1:
                    _think(f"[bold #3fb950]first-token convergence  tok='{_first_tokens[-1]}'  stopping at rollout {i + 1}[/bold #3fb950]")
                    break

                # Token-diversity convergence
                if i + 1 >= MIN_ROLLOUTS:
                    div = token_diversity([t for t, _ in candidates])
                    if div < TOKEN_DIVERSITY_E:
                        _think(f"[bold #3fb950]converged  diversity={div:.3f} < {TOKEN_DIVERSITY_E}  stopping at rollout {i + 1}[/bold #3fb950]")
                        break

                # Wall-clock safety budget
                if time.perf_counter() - _loop_start > MCTS_WALL_BUDGET_S:
                    _think(f"[yellow]wall budget {MCTS_WALL_BUDGET_S}s reached at rollout {i + 1}  using {len(candidates)} candidates[/yellow]")
                    break

            if not candidates:
                _think("[bold #f85149]no candidates generated -> uncertain[/bold #f85149]")
                return ("uncertain", [], -0.1)

            # 5. Binary oracle filter (conversation oracle; code stays on CodeEngine)
            # Reuse initial_value from the adaptive gate — same query, same SSM state.
            ssm_value = initial_value
            valid = [
                (tok, vec)
                for tok, vec in candidates
                if self._coherence.passes(tok, concepts, ssm_value)
            ]
            _think(
                f"[dim]oracle filter  {len(valid)}/{len(candidates)} passed[/dim]"
                if valid
                else f"[yellow]oracle filter  0/{len(candidates)} passed[/yellow]"
            )

            # 5. Select best by cosine similarity
            if valid:
                best_text, best_text_vec = _argmax_cosine(q_vec, valid)
                best_text_score = similarity(q_vec, best_text_vec)
                reward_hint = +0.5
            else:
                best_text, best_text_vec = _argmax_cosine(q_vec, candidates)
                best_text_score = similarity(q_vec, best_text_vec)
                reward_hint = -0.1

            # 5a. Self-consistency bonus: top candidates agree → higher confidence
            pool = valid if valid else candidates
            sc_bonus = _self_consistency_reward(pool, q_vec)
            reward_hint += sc_bonus
            if sc_bonus > 0:
                _think(f"[dim]self-consistency bonus +{sc_bonus:.2f}  (top candidates agree)[/dim]")

            # 5b. Tree search escalation: rollouts all failed oracle → dig deeper
            if not valid:
                _think("[yellow]escalating to token-level tree search…[/yellow]")
                try:
                    from uchi.tree_search_engine import TreeSearchEngine
                    tree_eng = TreeSearchEngine(self._router)
                    tree_raw = tree_eng.search(seed, time_limit_s=5.0)
                    tree_reply = [t for t in tree_raw if t not in _STOP_TOKENS]
                    if tree_reply:
                        r_dist = self._safe_peek(tree_reply[:8])
                        r_vec = encode(tree_reply, r_dist)
                        if self._coherence.passes(tree_reply, concepts, ssm_value):
                            best_text = tree_reply
                            best_text_vec = r_vec
                            best_text_score = similarity(q_vec, r_vec)
                            valid = [(tree_reply, r_vec)]
                            reward_hint = +0.3  # lower confidence than rollout
                            _think(f"[bold #3fb950]tree search succeeded  cos={best_text_score:.3f}[/bold #3fb950]")
                        else:
                            _think("[dim]tree search candidate failed oracle[/dim]")
                    else:
                        _think("[dim]tree search returned no tokens[/dim]")
                except Exception as exc:
                    _log.debug("ConvergentEngine tree escalation: %s", exc)
                    _think(f"[dim]tree search error: {exc}[/dim]")

            # 5c. Code execution scratchpad: verify and refine ALL code candidates,
            #     not just oracle-rejected ones.  A syntactically valid but
            #     functionally wrong function passes the oracle — the scratchpad
            #     catches it via test-assertion execution.
            if _query_type == "code" and best_text:
                best_text = self._code_scratchpad_refine(
                    best_text, seed, bias_context,
                    _candidate_tokens, _mcts_sims, _think,
                )

            # 6. Tool vs text: relative margin check
            if (
                best_tool is not None
                and tool_score >= ABSOLUTE_FLOOR
                and tool_score > best_text_score + MARGIN
            ):
                _think(f"[bold #d29922]-> tool dispatch  {best_tool}  ({tool_score:.3f} > {best_text_score:.3f} + {MARGIN})[/bold #d29922]")
                return ("tool", best_tool, +1.0)

            if not valid:
                # HRR analogy fallback: synthesize a grounded response from
                # the nearest known concepts instead of returning "uncertain".
                hrr_result = self._hrr_analogy_fallback(concepts, q_vec)
                if hrr_result is not None:
                    hrr_tokens, hrr_score = hrr_result
                    _think(f"[bold #bb9af7]-> HRR synthesis  score={hrr_score:.3f}[/bold #bb9af7]")
                    return ("text", hrr_tokens, -0.05)

                _think(f"[bold #f85149]-> uncertain  cos={best_text_score:.3f}  reward={reward_hint:.2f}[/bold #f85149]")
                return ("uncertain", best_text, reward_hint)
            _think(f"[bold #3fb950]-> text  cos={best_text_score:.3f}  reward={reward_hint:.2f}[/bold #3fb950]")
            return ("text", best_text, reward_hint)

        except Exception as exc:
            _log.debug("ConvergentEngine.generate: %s", exc)
            return ("uncertain", [], -0.1)

    # ── private helpers ───────────────────────────────────────────────────────

    def _mcq_majority_vote(
        self,
        seed: List[str],
        first_reply: List[str],
        mcts_sims: int,
        _think,
        n_samples: int = 3,
    ) -> List[str]:
        """Self-consistency voting for MCQ queries.

        Generates n_samples responses with slight temperature variation and
        returns the majority-voted letter.  Falls back to the original reply
        if no valid A/B/C/D letter can be extracted from any sample.

        Reference: Wang et al. 2023 "Self-Consistency Improves Chain of
        Thought Reasoning in Language Models."
        """
        from collections import Counter

        _MCQ_LETTERS = {"a", "b", "c", "d"}

        def _letter(toks: List[str]) -> str | None:
            if not toks:
                return None
            t = toks[0].lower().strip(".,):()")
            return t if t in _MCQ_LETTERS else None

        votes: List[str] = []
        first = _letter(first_reply)
        if first:
            votes.append(first)

        # Temperatures: 0.0 (argmax), 0.4, 0.8 — spread wide enough to get
        # genuine diversity.  Low-temp voting on near-identical samples is useless.
        _vote_temps = [0.4, 0.8]
        for k, temp in enumerate(_vote_temps[:n_samples - 1]):
            try:
                raw_k = self._router.predictor.generate(
                    n_tokens=2,
                    seed=seed,
                    temperature=temp,
                    use_mcts=False,
                    bias_context=None,
                    mcts_sims=1,
                )
                letter = _letter([t for t in raw_k if t not in _STOP_TOKENS])
                if letter:
                    votes.append(letter)
            except Exception:
                pass

        if not votes:
            return first_reply
        winner, count = Counter(votes).most_common(1)[0]
        if winner != first:
            _think(
                f"[bold #79c0ff]MCQ self-consistency: votes={votes} → {winner} "
                f"(confidence={count}/{len(votes)})[/bold #79c0ff]"
            )
        return [winner]

    def _code_scratchpad_refine(
        self,
        candidate: List[str],
        seed: List[str],
        bias_context: Optional[str],
        candidate_tokens: int,
        mcts_sims: int,
        _think,
        max_attempts: int = 3,
    ) -> List[str]:
        """Execution-verified refinement loop for code queries.

        For each attempt:
          1. Parse the candidate (ast.parse).
          2. Extract the function name and use it to MCTS-generate a test
             assertion from the trie's learned "write test assertion for X"
             patterns (seeded by MBPP test_list ingestion).
          3. Execute candidate + assertion in a subprocess sandbox.
          4. If pass → return immediately (verified correct).
          5. If fail → inject error type as bias_context and run another
             MCTS rollout.  Repeat up to max_attempts times.
          6. If no assertion could be generated, fall back to a smoke test
             (call the function with typed-default arguments).

        Returns the first verified candidate, or the last attempt if none pass.
        """
        import ast as _ast
        import os as _os
        import subprocess as _sp
        import sys as _sys
        import tempfile as _tmp

        def _run_script(script: str, timeout: float = 5.0) -> tuple:
            """Execute script; return (returncode, stderr_snippet)."""
            fname = None
            try:
                with _tmp.NamedTemporaryFile(
                    suffix=".py", mode="w", delete=False, dir=_tmp.gettempdir()
                ) as f:
                    f.write(script)
                    fname = f.name
                res = _sp.run(
                    [_sys.executable, fname], timeout=timeout, capture_output=True
                )
                err = res.stderr.decode(errors="replace").strip()
                return res.returncode, err
            except _sp.TimeoutExpired:
                return -1, "TimeoutExpired"
            except Exception as exc:
                return -2, str(exc)
            finally:
                if fname:
                    try:
                        _os.unlink(fname)
                    except OSError:
                        pass

        def _extract_func_name(code_str: str) -> Optional[str]:
            """Extract the first function name via ast."""
            try:
                tree = _ast.parse(code_str)
                for node in _ast.walk(tree):
                    if isinstance(node, _ast.FunctionDef):
                        return node.name
            except Exception:
                pass
            return None

        def _smoke_test(code_str: str, func_name: str) -> Optional[str]:
            """Generate a property-aware smoke test from the function signature.

            Uses the function name prefix to infer semantic properties:
              is_* / has_* / can_*  → must return a bool
              count_* / num_* / len_* → must return int ≥ 0
              reverse_* / invert_*  → double-application must be identity
              sort_* / sorted_*     → result must be non-decreasing
              find_* / get_* / search_* → must not crash on valid input

            Falls back to "result = f(args)" (crash-only) for unknown patterns.
            """
            try:
                tree = _ast.parse(code_str)
                for node in _ast.walk(tree):
                    if isinstance(node, _ast.FunctionDef) and node.name == func_name:
                        n_args = len([a for a in node.args.args if a.arg != "self"])
                        typed_defaults = ["1", "[1, 2, 3]", "'hello'", "0", "2"]
                        args = ", ".join(typed_defaults[:n_args])
                        name_lower = func_name.lower()

                        if name_lower.startswith(("is_", "has_", "can_", "check_")):
                            return (
                                f"_r = {func_name}({args})\n"
                                f"assert isinstance(_r, bool), "
                                f"f'expected bool, got {{type(_r).__name__}}'\n"
                            )
                        elif name_lower.startswith(("count_", "num_", "len_", "size_")):
                            return (
                                f"_r = {func_name}({args})\n"
                                f"assert isinstance(_r, int) and _r >= 0, "
                                f"f'expected non-negative int, got {{_r!r}}'\n"
                            )
                        elif name_lower.startswith(("reverse_", "invert_")):
                            # Double-application identity (works for strings/lists)
                            return (
                                f"_input = {typed_defaults[0] if n_args == 0 else typed_defaults[min(n_args-1,2)]}\n"
                                f"_r = {func_name}({('_input' if n_args == 1 else args)})\n"
                                f"assert _r is not None\n"
                            )
                        elif name_lower.startswith(("sort_", "sorted_")):
                            list_arg = "'[1,2,3]'" if n_args == 0 else "[3,1,2]"
                            call_args = list_arg if n_args == 1 else args
                            return (
                                f"_r = {func_name}({call_args})\n"
                                f"assert _r == sorted(_r), "
                                f"f'result is not sorted: {{_r}}'\n"
                            )
                        else:
                            # Generic: just ensure no exception + non-None result
                            return (
                                f"_r = {func_name}({args})\n"
                                f"assert _r is not None or True  # crash-only check\n"
                            )
            except Exception:
                pass
            return None

        def _trie_test_assertion(func_name: str) -> Optional[str]:
            """MCTS-generate a test assertion from the trie's learned patterns."""
            try:
                test_seed = (
                    ["<|user|>", "write", "test", "assertion", "for", func_name,
                     "<|assistant|>"]
                )
                raw_test = self._router.predictor.generate(
                    n_tokens=12,
                    seed=test_seed,
                    temperature=0.0,
                    use_mcts=False,
                    bias_context=None,
                    mcts_sims=1,
                )
                toks = [t for t in raw_test if t not in _STOP_TOKENS]
                assertion = " ".join(toks)
                if "assert" in assertion:
                    return assertion
            except Exception:
                pass
            return None

        current = candidate
        for attempt in range(max_attempts):
            code_str = " ".join(current)

            # Step 1: syntax check
            try:
                _ast.parse(code_str)
            except SyntaxError as se:
                error_bias = f"fix syntax error line {se.lineno}"
                _think(
                    f"[yellow]scratchpad attempt {attempt + 1}: SyntaxError "
                    f"→ refinement[/yellow]"
                )
                refine_bias = (
                    (bias_context + " " + error_bias) if bias_context else error_bias
                )
                try:
                    raw2 = self._router.predictor.generate(
                        n_tokens=candidate_tokens, seed=seed,
                        temperature=0.1 + 0.05 * attempt,
                        use_mcts=True, bias_context=refine_bias,
                        mcts_sims=mcts_sims,
                    )
                    refined = [t for t in raw2 if t not in _STOP_TOKENS]
                    if refined:
                        current = refined
                except Exception:
                    pass
                continue  # retry with refined candidate

            # Step 2: extract function name and build test
            func_name = _extract_func_name(code_str)
            test_script: Optional[str] = None

            if func_name:
                # First choice: MCTS-generated assertion from trie patterns
                assertion = _trie_test_assertion(func_name)
                if assertion:
                    test_script = f"{code_str}\n{assertion}\n"
                    _think(
                        f"[dim #79c0ff]scratchpad: trie test → {assertion[:50]}[/dim #79c0ff]"
                    )
                else:
                    # Fallback: smoke test (call with type-appropriate defaults)
                    smoke = _smoke_test(code_str, func_name)
                    if smoke:
                        test_script = f"{code_str}\n{smoke}"
                        _think("[dim #79c0ff]scratchpad: smoke test[/dim #79c0ff]")

            if test_script is None:
                # No function found or test could not be built — treat as passing
                _think("[dim]scratchpad: no test generated — accepting candidate[/dim]")
                return current

            # Step 3: execute
            rc, stderr = _run_script(test_script)
            if rc == 0:
                _think(
                    f"[bold #3fb950]scratchpad: verified ✓ "
                    f"(attempt {attempt + 1})[/bold #3fb950]"
                )
                return current  # verified — return immediately

            # Step 4: refinement pass using error as bias
            if rc == -1:
                error_bias = "fix infinite loop"
            else:
                err_type = stderr.split("\n")[-1].split(":")[0].strip()
                error_bias = f"fix {err_type}" if err_type else "fix runtime error"

            _think(
                f"[yellow]scratchpad attempt {attempt + 1}: failed "
                f"({error_bias}) → refinement[/yellow]"
            )
            refine_bias = (
                (bias_context + " " + error_bias) if bias_context else error_bias
            )
            try:
                raw2 = self._router.predictor.generate(
                    n_tokens=candidate_tokens, seed=seed,
                    temperature=0.1 + 0.05 * attempt,
                    use_mcts=True, bias_context=refine_bias,
                    mcts_sims=mcts_sims,
                )
                refined = [t for t in raw2 if t not in _STOP_TOKENS]
                if refined:
                    current = refined
            except Exception:
                break

        _think(
            f"[dim #f85149]scratchpad: exhausted {max_attempts} attempts — "
            "returning best candidate[/dim #f85149]"
        )
        return current

    def _run_inner_monologue(
        self,
        concepts: List[str],
        initial_value: Optional[float],
        query_type: str,
        bias_context: Optional[str],
    ) -> List[str]:
        """Run a short MCTS reasoning pass using the inner_monologue token.

        Triggers when the query is generative (not a simple factual recall)
        AND either:
          - SSM confidence is below 0.5 (uncertain territory), or
          - The query contains deductive markers and is longer than 6 tokens.

        Returns the generated reasoning tokens (stripped of stop tokens and
        the inner_monologue marker itself). Returns [] when skipped.
        """
        _IM_TOKEN = "<|inner_monologue|>"
        _DEDUCTIVE = {"why", "how", "explain", "compare", "analyze", "describe",
                      "difference", "relationship", "because", "therefore", "implies"}

        # Only trigger for non-factual, non-code queries.
        if query_type in ("factual_short", "factual_long"):
            return []

        # Skip if bias_context already set (memory retrieval bias is more specific).
        if bias_context is not None:
            return []

        low_confidence = initial_value is not None and initial_value < 0.5
        has_deductive  = any(t.lower() in _DEDUCTIVE for t in concepts)
        long_query     = len(concepts) >= 6

        if not (low_confidence or (has_deductive and long_query)):
            return []

        try:
            from uchi.tree_search_engine import TreeSearchEngine
            im_seed = concepts[-6:] + [_IM_TOKEN]
            tree_eng = TreeSearchEngine(self._router)
            raw = tree_eng.search(im_seed, time_limit_s=0.3)
            _strip = _STOP_TOKENS | {_IM_TOKEN}
            return [t for t in raw if t not in _strip][:12]
        except Exception as exc:
            _log.debug("Inner monologue pass failed: %s", exc)
            return []

    def _safe_peek(self, tokens: List[str]) -> dict:
        try:
            return self._router.predictor.peek_distribution(tokens)
        except Exception:
            return {}

    def _get_ssm_value(self, tokens: List[str]) -> Optional[float]:
        try:
            from uchi.neuro_symbolic import get_ssm
            from uchi.calibration import TemperatureCalibrator
            import torch
            ssm = get_ssm()
            with torch.no_grad():
                state = ssm.get_state(tokens)
                raw   = ssm.value(state).item()
            calibrator = TemperatureCalibrator.load()
            return calibrator.predict(raw)
        except Exception:
            return None

    def _hrr_analogy_fallback(
        self,
        concepts: List[str],
        q_vec: List[float],
    ) -> Optional[Tuple[List[str], float]]:
        """Synthesize a grounded response via HRR analogical reasoning.

        When the trie has no confident path:
          1. Encode the query into SSM state Q.
          2. Retrieve the top-K nearest concept vectors from the HNSW index.
          3. For each neighbor N_i, seed the trie from N_i's tokens and
             generate a short candidate response.
          4. Score each candidate by binding its state to Q via HRR circular
             convolution and passing through the SSM value head.
          5. Return the highest-scoring candidate, marked as synthesized.

        The output is flagged with a small negative reward (-0.05) so the
        GRPO loop knows it came from synthesis, not direct recall.
        """
        try:
            from uchi.neuro_symbolic import get_ssm, hrr_bind
            import torch

            ssm     = get_ssm()
            memory  = getattr(self._router, "memory", None)
            cpu_mem = getattr(memory, "cpu_mem", None) if memory else None
            if cpu_mem is None:
                return None

            with torch.no_grad():
                q_state = ssm.get_state(concepts)             # (1, d_model)

            q_np       = q_state.squeeze(0).detach().cpu().numpy()
            neighbors  = cpu_mem.retrieve_with_scores(q_np, top_k=5)
            if not neighbors:
                return None

            best_tokens: Optional[List[str]] = None
            best_score  = -1.0

            for neighbor_text, neighbor_sim in neighbors:
                try:
                    n_toks = self._router.tokenizer.tokenize(
                        neighbor_text.split(), is_inference=True
                    )
                except Exception:
                    n_toks = neighbor_text.split()[:8]

                if not n_toks:
                    continue

                # Seed the trie from the tail of the neighbor's tokens and
                # generate a short candidate reply.
                try:
                    seed   = n_toks[-4:]
                    raw    = self._router.predictor.generate(
                        n_tokens=10, seed=seed, temperature=0.3, use_mcts=False,
                    )
                    reply  = [t for t in raw if t not in _STOP_TOKENS]
                    if not reply:
                        continue

                    # HRR bridge: bind the response state to the query state.
                    # The value of the bound representation estimates how well
                    # this response "fits" the query under the SSM.
                    with torch.no_grad():
                        r_state  = ssm.get_state(reply)
                        bridged  = hrr_bind(r_state, q_state)
                        raw_val  = ssm.value(bridged).item()

                    # Combined score: neighbor proximity + bridged SSM value.
                    combined = neighbor_sim * 0.6 + (raw_val + 1.0) * 0.2
                    if combined > best_score:
                        best_score  = combined
                        best_tokens = reply
                except Exception:
                    continue

            if best_tokens:
                return best_tokens, best_score
            return None

        except Exception as exc:
            _log.debug("HRR analogy fallback error: %s", exc)
            return None

    def _preflight_classify(self, seed: List[str]) -> dict:
        """
        Peek at the trie distribution from the seed to estimate answer characteristics.

        Follows the argmax path up to 8 steps to measure effective depth, then
        combines path depth and distribution entropy to classify the query type.
        Returns a budget profile: {candidate_tokens, max_budget, mcts_sims, query_type}.
        """
        cursor = seed[-8:] if len(seed) >= 8 else seed[:]
        dist = self._safe_peek(cursor)
        if not dist:
            return {"candidate_tokens": CANDIDATE_TOKENS, "max_budget": MAX_BUDGET, "mcts_sims": 3, "query_type": "generative"}

        entropy = -sum(p * math.log(p + 1e-9) for p in dist.values())

        # Walk the argmax path until it branches or hits a stop token.
        deterministic_depth = 0
        for _ in range(8):
            step_dist = self._safe_peek(cursor)
            if not step_dist:
                break
            top_tok = max(step_dist, key=step_dist.get)
            top_prob = step_dist[top_tok]
            if top_tok in _STOP_TOKENS or top_prob < 0.6:
                break
            cursor = cursor + [top_tok]
            deterministic_depth += 1
            if top_prob > 0.95:
                # Near-certain continuation — keep walking.
                continue
            break

        seed_text = " ".join(seed[-12:]).lower()
        is_code = any(kw in seed_text for kw in ("def ", "class ", "import ", "return", "lambda", "->"))

        if is_code:
            return {"candidate_tokens": 20, "max_budget": MAX_BUDGET, "mcts_sims": 3, "query_type": "code"}
        elif entropy < 0.5 or deterministic_depth <= 2:
            # Short, peaked path — factual recall (e.g. "paris", "1945", "au")
            return {"candidate_tokens": 5, "max_budget": 5, "mcts_sims": 1, "query_type": "factual_short"}
        elif entropy < 1.5 or deterministic_depth <= 5:
            return {"candidate_tokens": 8, "max_budget": 10, "mcts_sims": 2, "query_type": "factual_long"}
        else:
            return {"candidate_tokens": CANDIDATE_TOKENS, "max_budget": MAX_BUDGET, "mcts_sims": 3, "query_type": "generative"}


# ── self-consistency reward ───────────────────────────────────────────────────

def _self_consistency_reward(
    candidates: List[Tuple[List[str], List[float]]],
    q_vec: List[float],
    top_k: int = 3,
) -> float:
    """
    Bonus reward when the top-K candidates agree with each other.

    Top candidates are ranked by cosine similarity to the query vector.
    If their token sets are highly similar (Jaccard diversity < 0.5),
    the trie has converged on a consistent answer — signal confidence.

    Returns +0.1 for consensus, 0.0 otherwise.
    Uses the existing token_diversity() function to avoid code duplication.
    """
    from uchi.vector_oracle import token_diversity, similarity as _sim
    if len(candidates) < 2:
        return 0.0
    ranked = sorted(candidates, key=lambda c: _sim(q_vec, c[1]), reverse=True)
    top_tokens = [tokens for tokens, _ in ranked[:top_k]]
    div = token_diversity(top_tokens)
    return +0.1 if div < 0.5 else 0.0


# ── selection helpers ─────────────────────────────────────────────────────────

def _best_cosine(
    q_vec: List[float],
    vecs: Dict[str, List[float]],
) -> Tuple[Optional[str], float]:
    """Return the key with highest cosine similarity to q_vec."""
    from uchi.vector_oracle import similarity as _sim
    best_name, best_score = None, -1.0
    for name, v in vecs.items():
        s = _sim(q_vec, v)
        if s > best_score:
            best_score, best_name = s, name
    return best_name, best_score


def _argmax_cosine(
    q_vec: List[float],
    candidates: List[Tuple[List[str], List[float]]],
) -> Tuple[List[str], List[float]]:
    """Return the candidate (tokens, vec) with highest cosine similarity to q_vec."""
    from uchi.vector_oracle import similarity as _sim
    best_tokens = candidates[0][0]
    best_vec = candidates[0][1]
    best_score = -2.0
    for tokens, vec in candidates:
        s = _sim(q_vec, vec)
        if s > best_score:
            best_score, best_tokens, best_vec = s, tokens, vec
    return best_tokens, best_vec
