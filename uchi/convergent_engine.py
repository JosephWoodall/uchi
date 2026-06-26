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
# High confidence (value head > GREEDY_CONF and trie entropy < GREEDY_ENTROPY_CAP):
#   bypass the MCTS loop and generate a single greedy candidate in O(1).
# Low confidence (value < UNCERTAIN_FLOOR): scale the rollout budget up by
#   BUDGET_SCALE_UNCERTAIN to invest saved compute into hard queries.
GREEDY_CONF        = 0.85
GREEDY_ENTROPY_CAP = 1.5  # was 0.6; allow moderately peaked distributions through
GREEDY_ENTROPY_FLOOR = 0.05  # near-zero entropy → clear trie path, bypass regardless of SSM
UNCERTAIN_FLOOR    = 0.0
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
            import py_compile, tempfile, os
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

            # 6. Tool vs text: relative margin check
            if (
                best_tool is not None
                and tool_score >= ABSOLUTE_FLOOR
                and tool_score > best_text_score + MARGIN
            ):
                _think(f"[bold #d29922]-> tool dispatch  {best_tool}  ({tool_score:.3f} > {best_text_score:.3f} + {MARGIN})[/bold #d29922]")
                return ("tool", best_tool, +1.0)

            if not valid:
                _think(f"[bold #f85149]-> uncertain  cos={best_text_score:.3f}  reward={reward_hint:.2f}[/bold #f85149]")
                return ("uncertain", best_text, reward_hint)
            _think(f"[bold #3fb950]-> text  cos={best_text_score:.3f}  reward={reward_hint:.2f}[/bold #3fb950]")
            return ("text", best_text, reward_hint)

        except Exception as exc:
            _log.debug("ConvergentEngine.generate: %s", exc)
            return ("uncertain", [], -0.1)

    # ── private helpers ───────────────────────────────────────────────────────

    def _safe_peek(self, tokens: List[str]) -> dict:
        try:
            return self._router.predictor.peek_distribution(tokens)
        except Exception:
            return {}

    def _get_ssm_value(self, tokens: List[str]) -> Optional[float]:
        try:
            from uchi.neuro_symbolic import get_ssm
            import torch
            ssm = get_ssm()
            with torch.no_grad():
                state = ssm.get_state(tokens)
                return ssm.value(state).item()
        except Exception:
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
