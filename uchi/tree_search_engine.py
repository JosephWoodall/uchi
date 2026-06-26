"""
tree_search_engine.py
=====================
Token-level Monte Carlo Tree Search using the SSM value head as a per-step
evaluator (AlphaGo Zero PUCT variant).

Unlike ConvergentEngine's rollout search (generate 45 tokens → grade whole
sequence), the tree search evaluates each token individually:

  - Bad branches are pruned at token 3 instead of wasting 42 more tokens.
  - Promising branches get deeper exploration via UCB1 priority.
  - The trie's P(token|context) serves as the UCB1 prior (policy network).
  - The SSM value head serves as V(state) (value network).

PUCT selection formula (per node):
  score(child) = V(child) + C * P(child) * sqrt(N_parent) / (1 + N_child)

  V = SSM value head estimate for the token sequence at this node
  P = trie P(this_token | parent_tokens) — pattern-matching prior
  N = visit count
  C = 1.4 (exploration constant, √2 ≈ standard for games)

Quality ceiling: the tree search is only as good as the SSM value head.
At cold start the value head outputs ~0 for all inputs, so UCB1 degrades to
pure prior-guided search (trie probabilities steer the tree). As contrastive
training progresses and RL refines the value head, the tree search quality
rises with it.

Used as a fallback escalation inside ConvergentEngine: if the rollout phase
produces zero oracle-valid candidates, the tree search attempts a deeper,
more targeted exploration of the same knowledge space.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

UCB_C           = 1.4   # exploration constant
PRUNE_THRESHOLD = -0.5  # prune nodes whose SSM value falls below this
EXPAND_K        = 5     # top-K trie tokens expanded at each leaf
DEFAULT_MAX_NODES = 200  # total node expansions per search call
DEFAULT_MAX_DEPTH = 40   # maximum token depth from root

# Weight of the neural policy prior vs. the trie statistical prior.
# hybrid_prior = (1 - POLICY_ALPHA) * trie_prob + POLICY_ALPHA * neural_prob
# At α=0 the tree is pure trie-guided; at α=1 it is pure actor-guided.
POLICY_ALPHA    = 0.4

# Batched leaf evaluation with virtual loss (AlphaZero-style).
# Each outer MCTS step collects BATCH_SIZE diverse leaves simultaneously,
# then evaluates all candidate tokens in a single GRU+value forward pass.
BATCH_SIZE      = 8   # leaves gathered per batch step
VIRTUAL_LOSS    = 3   # visit-count penalty applied during parallel selection

# Repetition penalty: multiplicative decay on hybrid_prior for tokens already
# in the current path.  First repeat → 0.15×, second → 0.02×, effectively zero.
REP_DECAY       = 0.15

_INNER_MONOLOGUE = "<|inner_monologue|>"

_STOP_TOKENS = frozenset({"<|user|>", "<|assistant|>", "<|end|>"})


# ── tree node ─────────────────────────────────────────────────────────────────

class _Node:
    """One node in the MCTS search tree."""
    __slots__ = (
        "tokens", "value", "prior", "visits", "children",
        "pruned", "hidden", "kv_cache", "virtual_loss",
    )

    def __init__(
        self,
        tokens: List[str],
        value: float = 0.0,
        prior: float = 1.0,
        hidden=None,
        kv_cache=None,
    ):
        self.tokens: List[str] = tokens
        self.value: float = value
        self.prior: float = prior
        self.visits: int = 0
        self.children: Dict[str, _Node] = {}
        self.pruned: bool = False
        self.hidden = hidden
        self.kv_cache = kv_cache
        # Temporary virtual-loss counter: incremented during parallel selection,
        # decremented before true backprop so concurrent traversals explore
        # distinct branches without needing separate threads.
        self.virtual_loss: int = 0


# ── engine ────────────────────────────────────────────────────────────────────

class TreeSearchEngine:
    """
    Token-level PUCT tree search for Uchi.

    Builds a search tree token-by-token from a seed sequence.
    Returns the token extension that has the highest cumulative
    SSM value through the best leaf found within the node budget.
    """

    def __init__(self, router):
        self._router = router

    # ── public API ────────────────────────────────────────────────────────────

    def search(
        self,
        seed: List[str],
        max_nodes: int = DEFAULT_MAX_NODES,
        max_depth: int = DEFAULT_MAX_DEPTH,
        time_limit_s: float = 5.0,
    ) -> List[str]:
        """
        Run PUCT tree search from seed.

        Returns the generated tokens (seed not included).
        Returns [] if no promising path is found.
        """
        if not seed:
            return []

        import torch
        import torch.nn.functional as _F

        try:
            import uchi.telemetry as _tel
            _tel.reset()
            _tel.record("mcts", "seed_len", len(seed))
        except Exception:
            _tel = None

        _nodes_explored    = 0
        _batch_sizes_used: list = []
        _rep_penalties_applied = 0
        _oracle_interventions  = 0

        # Compute root hidden state once; all descendants reuse it incrementally.
        root_val, root_hidden = self._eval_with_hidden(seed)
        root_kv = self._build_seed_cache(seed)
        root = _Node(tokens=list(seed), value=root_val, prior=1.0, hidden=root_hidden, kv_cache=root_kv)
        root.visits = 1

        _search_start = time.perf_counter()

        # Outer loop: iterate in batches of BATCH_SIZE for parallelised leaf eval.
        for _batch_start in range(0, max_nodes, BATCH_SIZE):
            if time.perf_counter() - _search_start > time_limit_s:
                logging.debug("TreeSearchEngine: time_limit_s=%.1f reached at batch %d", time_limit_s, _batch_start)
                break
            _batch_n = min(BATCH_SIZE, max_nodes - _batch_start)

            # ── Phase 1: select BATCH_SIZE diverse leaves via virtual loss ────────
            # Each traversal applies a virtual-loss penalty to visited nodes so the
            # next traversal is steered toward a different branch, giving us
            # BATCH_SIZE distinct leaves without needing parallel threads.
            _batch_leaves: List[Tuple[_Node, int, List[_Node]]] = []
            for _ in range(_batch_n):
                node, depth = root, 0
                _vl_path: List[_Node] = []
                while node.children and depth < max_depth:
                    selected = self._ucb1_select(node)
                    if selected is None or selected.pruned:
                        break
                    selected.virtual_loss += VIRTUAL_LOSS
                    _vl_path.append(selected)
                    node = selected
                    depth = len(node.tokens) - len(seed)
                _batch_leaves.append((node, depth, _vl_path))

            # ── Phase 2: expand each leaf, batch-evaluating all candidates ────────
            for (node, depth, _vl_path) in _batch_leaves:
                # Always remove virtual loss before any continue so UCB1 stays consistent
                for _vl_node in _vl_path:
                    _vl_node.virtual_loss -= VIRTUAL_LOSS

                if node.pruned or depth >= max_depth:
                    continue

                dist = self._safe_peek(node.tokens)
                if not dist:
                    continue

                _pre_mask_count = len([p for p in dist.values() if p > 0])
                try:
                    from uchi.grammar_mask import apply as _grammar_apply
                    dist = _grammar_apply(node.tokens, dist)
                except Exception:
                    pass
                _post_mask_count = len([p for p in dist.values() if p > 0])
                if _post_mask_count < _pre_mask_count:
                    _oracle_interventions += _pre_mask_count - _post_mask_count

                sorted_toks = sorted(dist.items(), key=lambda x: x[1], reverse=True)
                candidates_to_expand = [
                    (str(tok_str), float(prob))
                    for tok_str, prob in sorted_toks[:EXPAND_K]
                    if str(tok_str) not in node.children and str(tok_str) not in _STOP_TOKENS
                ]
                if not candidates_to_expand:
                    continue

                tok_strs = [tok for tok, _ in candidates_to_expand]

                # Temperature annealing
                T = max(1.0, 5.0 * (1.0 - (depth / max(max_depth, 1))))
                raw_trie = [p / T for _, p in candidates_to_expand]
                trie_probs_annealed = _F.softmax(
                    torch.tensor(raw_trie, dtype=torch.float32), dim=0
                ).tolist()

                _batch_sizes_used.append(len(tok_strs))

                if node.hidden is not None:
                    # Batch policy scores: one PolicyHead forward for all EXPAND_K
                    policy_logits = self._batch_policy_scores(node.hidden, tok_strs)
                    neural_priors = _F.softmax(
                        torch.tensor(policy_logits, dtype=torch.float32), dim=0
                    ).tolist()
                    # Batch GRU+value: one GRU pass for all EXPAND_K children
                    child_evals = self._eval_incremental_batch(
                        node.hidden, tok_strs, node.kv_cache
                    )
                else:
                    n_cands = len(candidates_to_expand)
                    neural_priors = [1.0 / max(n_cands, 1)] * n_cands
                    child_evals = []
                    for tok in tok_strs:
                        cv, ch = self._eval_with_hidden(node.tokens + [tok])
                        child_evals.append((cv, ch, None))

                for (tok, _), neural_prob, trie_prob, (child_val, child_hidden, child_kv) in zip(
                    candidates_to_expand, neural_priors, trie_probs_annealed, child_evals
                ):
                    hybrid_prior = (1.0 - POLICY_ALPHA) * trie_prob + POLICY_ALPHA * neural_prob

                    # Repetition penalty: exponential decay for tokens already in path.
                    # Breaks absorbing loops (e.g. "see you later see you later...").
                    tok_count = node.tokens.count(tok)
                    if tok_count > 0:
                        hybrid_prior *= REP_DECAY ** tok_count
                        _rep_penalties_applied += 1

                    _nodes_explored += 1
                    child = _Node(
                        tokens=node.tokens + [tok],
                        value=child_val,
                        prior=hybrid_prior,
                        hidden=child_hidden,
                        kv_cache=child_kv,
                    )
                    if child_val < PRUNE_THRESHOLD:
                        child.pruned = True
                    node.children[tok] = child

                # Inner-monologue injection when all children pruned
                active_after = [c for c in node.children.values() if not c.pruned]
                if not active_after and node.children and _INNER_MONOLOGUE not in node.children:
                    if node.hidden is not None:
                        im_val, im_hidden, im_kv = self._eval_incremental(
                            node.hidden, _INNER_MONOLOGUE, node.kv_cache
                        )
                    else:
                        im_val, im_hidden = self._eval_with_hidden(node.tokens + [_INNER_MONOLOGUE])
                        im_kv = None
                    im_child = _Node(
                        tokens=node.tokens + [_INNER_MONOLOGUE],
                        value=im_val,
                        prior=0.5,
                        hidden=im_hidden,
                        kv_cache=im_kv,
                    )
                    node.children[_INNER_MONOLOGUE] = im_child
                    _log.debug(
                        "TreeSearchEngine: <|inner_monologue|> injected at depth %d",
                        len(node.tokens),
                    )

                active = [c for c in node.children.values() if not c.pruned]
                backup = max((c.value for c in active), default=node.value)
                self._backprop(root, node, backup)

        # ── Flush MCTS telemetry ──────────────────────────────────────────────
        try:
            if _tel is not None:
                avg_batch = (sum(_batch_sizes_used) / len(_batch_sizes_used)
                             if _batch_sizes_used else 0.0)
                _tel.record("mcts", "total_nodes_explored",    _nodes_explored)
                _tel.record("mcts", "gpu_batch_utilization",   round(avg_batch / max(EXPAND_K, 1), 3))
                _tel.record("mcts", "repetition_penalty_applied", _rep_penalties_applied)
                _tel.record("mcts", "oracle_interventions",    _oracle_interventions)

                # Top branches: walk tree to collect visible leaves
                _branches = []
                _q: list = [(root, 0)]
                while _q:
                    _n, _d = _q.pop()
                    if not _n.children and _d > 0:
                        _gen = _n.tokens[len(seed):]
                        _strip = _STOP_TOKENS | {_INNER_MONOLOGUE}
                        _seq = [t for t in _gen if t not in _strip]
                        if _seq:
                            _branches.append({
                                "sequence":        " ".join(_seq[:8]),
                                "neural_value":    round(_n.value, 4),
                                "visits":          _n.visits,
                            })
                    for _c in _n.children.values():
                        if not _c.pruned:
                            _q.append((_c, _d + 1))

                _branches.sort(key=lambda b: b["visits"] * max(b["neural_value"], 0.0), reverse=True)
                _tel.record("mcts", "top_branches", _branches[:10])
                _tel.flush()
        except Exception:
            pass

        return self._best_path(root, len(seed))

    # ── private helpers ───────────────────────────────────────────────────────

    def _batch_policy_scores(self, parent_hidden, tokens: List[str]) -> List[float]:
        """
        Batch PolicyHead forward for K candidate tokens from one shared parent state.
        Returns K raw logits (pre-softmax).  Falls back to zeros on any error.
        """
        try:
            from uchi.neuro_symbolic import get_ssm
            import torch
            ssm = get_ssm()
            K = len(tokens)
            with torch.no_grad():
                tok_embeds = ssm.embedder(tokens)                      # (K, d_model)
                state_exp  = parent_hidden.expand(K, -1)               # (K, d_model)
                logits     = ssm.policy_head(state_exp, tok_embeds)    # (K, 1)
            return logits.squeeze(-1).tolist()
        except Exception as exc:
            _log.debug("TreeSearchEngine._batch_policy_scores: %s", exc)
            return [0.0] * len(tokens)

    def _eval_incremental_batch(
        self, parent_hidden, tokens: List[str], parent_kv=None
    ) -> List[Tuple[float, object, object]]:
        """
        Single GRU + value forward pass for K candidate tokens from one parent.

        Replaces K sequential _eval_incremental calls with one batched call:
          - Embeds all K tokens: (K, d_model)
          - Repeats parent hidden K times: (1, K, d_model)
          - One GRU step yields K new hidden states simultaneously
          - One value head pass scores all K states

        KV-cache is inherited from parent unchanged (cross-attention enrichment
        is skipped in the batch path to avoid padding different-length caches;
        the sequential path used when these children are later expanded restores
        full cross-attention via _eval_incremental).

        Returns list of (value, child_hidden, parent_kv) of length K.
        Falls back to sequential _eval_incremental on any error.
        """
        try:
            from uchi.neuro_symbolic import get_ssm
            import torch
            ssm = get_ssm()
            K = len(tokens)
            with torch.no_grad():
                tok_embeds  = ssm.embedder(tokens)                                     # (K, d_model)
                projected   = ssm.encoder.input_proj(tok_embeds)                       # (K, d_model)
                gru_in      = projected.unsqueeze(1)                                   # (K, 1, d_model)
                h0          = parent_hidden.unsqueeze(0).expand(1, K, -1).contiguous() # (1, K, d_model)
                ssm.encoder.gru.flatten_parameters()
                _, h_n      = ssm.encoder.gru(gru_in, h0)                             # (1, K, d_model)
                import torch.nn.functional as _F2
                new_hiddens = _F2.normalize(h_n.squeeze(0), p=2, dim=-1)              # (K, d_model)
                vals        = ssm.value(new_hiddens).squeeze(-1)                       # (K,)
            return [
                (vals[i].item(), new_hiddens[i].unsqueeze(0), parent_kv)
                for i in range(K)
            ]
        except Exception as exc:
            _log.debug("TreeSearchEngine._eval_incremental_batch: %s", exc)
            return [self._eval_incremental(parent_hidden, tok, parent_kv) for tok in tokens]

    def _get_policy_score(self, parent_hidden, token: str) -> float:
        """
        Actor score for expanding *token* from a cached parent hidden state.
        Returns a raw logit — callers must softmax over the full candidate set.
        Returns 0.0 on any error so the softmax degrades to uniform.
        """
        try:
            from uchi.neuro_symbolic import get_ssm
            ssm = get_ssm()
            with __import__("torch").no_grad():
                return ssm.get_policy_score(parent_hidden, token)
        except Exception as exc:
            _log.debug("TreeSearchEngine._get_policy_score: %s", exc)
            return 0.0

    def _eval(self, tokens: List[str]) -> float:
        """Evaluate a token sequence using the SSM value head (full re-encode)."""
        val, _ = self._eval_with_hidden(tokens)
        return val

    def _eval_with_hidden(self, tokens: List[str]) -> Tuple[float, Optional[object]]:
        """Full-sequence eval returning (value, hidden_state) for caching."""
        try:
            from uchi.neuro_symbolic import get_ssm
            import torch
            ssm = get_ssm()
            with torch.no_grad():
                hidden = ssm.get_state(tokens)        # (1, d_model)
                val = ssm.value(hidden).item()
            return val, hidden
        except Exception as exc:
            _log.debug("TreeSearchEngine._eval_with_hidden: %s", exc)
            return 0.0, None

    def _eval_incremental(
        self, parent_hidden, token: str, parent_kv=None
    ) -> Tuple[float, object, object]:
        """
        O(1) incremental eval: one GRU step from a cached parent hidden state.
        4–16x faster than _eval_with_hidden for deep MCTS nodes.

        If *parent_kv* is provided, the new state is cross-attention enriched
        over the episodic history before value scoring.

        Returns (value, new_hidden, new_kv_cache).
        Falls back to (0.0, parent_hidden, parent_kv) on any error.
        """
        try:
            from uchi.neuro_symbolic import get_ssm
            import torch
            ssm = get_ssm()
            with torch.no_grad():
                new_hidden, new_kv = ssm.get_state_incremental(parent_hidden, token, parent_kv)
                val = ssm.value(new_hidden).item()
            return val, new_hidden, new_kv
        except Exception as exc:
            _log.debug("TreeSearchEngine._eval_incremental: %s", exc)
            return 0.0, parent_hidden, parent_kv

    def _build_seed_cache(self, seed: List[str]):
        """Build initial episodic KV cache from the search seed sequence."""
        try:
            from uchi.neuro_symbolic import get_ssm
            import torch
            ssm = get_ssm()
            with torch.no_grad():
                return ssm.get_kv_cache(seed)
        except Exception as exc:
            _log.debug("TreeSearchEngine._build_seed_cache: %s", exc)
            return None

    def _safe_peek(self, tokens: List[str]) -> dict:
        """
        Return trie P(next | tokens) without modifying predictor state.
        Falls back to the semantic k-NN index when the trie has no children
        for this context (cold-start / OOV path).
        """
        try:
            # Iterative N-Gram backoff: N=8 down to N=2
            for n in range(8, 1, -1):
                dist = self._router.predictor.peek_distribution(tokens[-n:])
                if dist:
                    return dist
        except Exception:
            pass

        # ── Semantic k-NN fallback: synthesise distribution from the nearest
        #    historical SSM states stored in the dreaming index.
        try:
            from uchi.semantic_index import get_semantic_index
            from uchi.neuro_symbolic import get_ssm
            import torch
            ssm = get_ssm()
            with torch.no_grad():
                state = ssm.get_state(tokens[-8:])
            return get_semantic_index().query(state)
        except Exception:
            return {}

    def _ucb1_select(self, node: _Node) -> Optional[_Node]:
        """Return the child with the highest PUCT score, respecting virtual loss."""
        best_score = -math.inf
        best_child: Optional[_Node] = None
        sqrt_n = math.sqrt(max(node.visits, 1))
        
        # Dynamic UCB: scale exploration inversely with node value magnitude
        dynamic_ucb = UCB_C / (0.5 + abs(node.value))
        
        for child in node.children.values():
            if child.pruned:
                continue
            exploit = child.value
            # Virtual loss increases effective visit count so parallel traversals
            # are steered toward different branches.
            effective_visits = child.visits + child.virtual_loss
            explore = dynamic_ucb * child.prior * sqrt_n / (1 + effective_visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _backprop(
        self, root: _Node, leaf: _Node, backup: float
    ) -> None:
        """Walk from leaf toward root, incrementing visits and averaging value."""
        node = leaf
        while node is not root:
            node.visits += 1
            node.value += (backup - node.value) / node.visits
            # Walk up: find parent by matching token prefix
            parent = self._find_parent(root, node)
            if parent is None:
                break
            node = parent
        root.visits += 1

    def _find_parent(self, root: _Node, target: _Node) -> Optional[_Node]:
        """BFS from root to find the parent of target."""
        if not target.tokens or len(target.tokens) <= len(root.tokens):
            return None
        parent_tokens = target.tokens[:-1]
        queue = [root]
        while queue:
            current = queue.pop()
            if current.tokens == parent_tokens:
                return current
            queue.extend(current.children.values())
        return None

    def _best_path(self, root: _Node, seed_len: int) -> List[str]:
        """
        Follow highest-visit children from root to extract the best sequence.
        Returns only the generated tokens (strips the seed prefix).
        <|inner_monologue|> is an internal search control token and is always
        stripped from the final output even if it appears mid-path.
        """
        node = root
        while node.children:
            active = {k: v for k, v in node.children.items() if not v.pruned}
            if not active:
                break
            best_tok = max(active.items(), key=lambda x: x[1].visits)[0]
            node = active[best_tok]
        generated = node.tokens[seed_len:]
        _strip = _STOP_TOKENS | {_INNER_MONOLOGUE}
        return [t for t in generated if t not in _strip]
