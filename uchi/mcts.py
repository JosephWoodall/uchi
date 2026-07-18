"""mcts.py — 0.5.0 Item 8: value-guided search over hypothesis-level
actions, using `world_model.WorldModel` as the simulator.

Ports `efficient_llm_training/src/mcts.py`'s `MCTSNode`/`MCTS` PUCT
mechanics (`ucb`, `_select`, `_backprop`) as-is — the search algorithm
itself doesn't care what an "action" is, only that it has a state, a
prior, and a value. The one real redefinition, per `tasks/todo.md`'s Item
8 section ("Redefine 'action' as hypothesis/reasoning-step level, using
Item 6's Thought-tagged trajectory segments as the discrete units
(mean-pooled embeddings, same approach as `proprioception.py`"):

  - The source picks among top-K vocab tokens at each step, using the LM's
    own next-token logits as priors. Here, one action is a whole sampled
    Thought/Action continuation (`generate_fn` output), and its embedding
    is the same mean-pooled hidden state `proprioception.py`'s
    `pooled_hidden_for_question` already computes for a question — reused
    directly, not reimplemented, since pooling a free-form text into one
    `(d_model,)` vector is exactly the same operation either way.
  - Priors, without per-token LM logits to read off, come from
    `grpo.sequence_log_prob` (0.5.0 Item 6) instead: each candidate's own
    log-prob under the policy, softmax-normalized across the top-K
    candidates in the group. This is the natural analogue of the source's
    top-k-logits prior at the granularity Item 8 actually operates at.

Depth > 1 rollout (imagining further ahead without generating real text)
uses the same dynamics-chain approximation as the source: a neutral/zero
action probe, since there is no free-form candidate to embed at a purely
imagined node.

Real search quality depends on a `WorldModel.value` that actually
correlates with real pass/fail (0.5.0 Item 5's reward) — untrained here,
same gate as `grpo.py` and `world_model.py`. This module is buildable and
testable now against a stub dynamics/value function (see
`tests/test_mcts.py`); real search against a trained value head waits for
Phase 1->4 + Item 6 to produce one.
"""
from __future__ import annotations

import math
from typing import Callable

import torch

from .flux.verifier_model import OODDetector
from .grpo import TrajectoryStep, sequence_log_prob
from .proprioception import pooled_hidden_for_question


class MCTSNode:
    __slots__ = ("state_vec", "prior", "parent", "action",
                 "children", "visit_count", "value_sum")

    def __init__(self, state_vec, prior: float = 1.0, parent=None, action: TrajectoryStep | None = None):
        self.state_vec = state_vec   # (1, d_model) — compressed world state
        self.prior = prior           # P(action | parent), from sequence_log_prob softmax
        self.parent = parent
        self.action = action         # TrajectoryStep that led to this node
        self.children: dict = {}     # candidate index → MCTSNode
        self.visit_count: int = 0
        self.value_sum: float = 0.0

    @property
    def value(self) -> float:
        return self.value_sum / max(self.visit_count, 1)

    def ucb(self, c_puct: float = 1.5) -> float:
        if self.parent is None:
            return 0.0
        exploration = (c_puct * self.prior *
                       math.sqrt(self.parent.visit_count) / (1 + self.visit_count))
        return self.value + exploration

    def is_leaf(self) -> bool:
        return len(self.children) == 0


def _candidate_priors(log_probs: list[float]) -> list[float]:
    """Softmax-normalize a group of candidates' sequence log-probs into
    priors -- the hypothesis-level analogue of the source's top-k LM-logit
    softmax, using `grpo.sequence_log_prob` instead of per-token logits."""
    t = torch.tensor(log_probs, dtype=torch.float32)
    return torch.softmax(t, dim=0).tolist()


class MCTS:
    """
    Value-guided MCTS over hypothesis-level (Thought/Action) actions.

    Parameters
    ----------
    world_model : world_model.WorldModel
        Provides dynamics (next-state prediction) and value estimation.
    n_simulations : int
        Number of MCTS simulations per decision (default 8).
    top_k : int
        Branching factor — candidate hypotheses sampled from `generate_fn`
        per decision (default 4).
    depth : int
        Rollout depth. 1 = one-step lookahead; >1 uses dynamics chain
        (default 1).
    alpha : float
        Mix weight: score = alpha*value + (1-alpha)*log_prob (default 0.4).
    c_puct : float
        UCB exploration constant (default 1.5).
    """

    def __init__(self, world_model,
                 n_simulations: int = 8, top_k: int = 4,
                 depth: int = 1, alpha: float = 0.4, c_puct: float = 1.5):
        self.wm = world_model
        self.n_sims = n_simulations
        self.top_k = top_k
        self.depth = depth
        self.alpha = alpha
        self.c_puct = c_puct

    @torch.no_grad()
    def _leaf_value(self, state_vec: torch.Tensor) -> float:
        return self.wm.value(state_vec).item()

    @torch.no_grad()
    def _simulate(self, node: MCTSNode) -> float:
        """Roll out `depth` steps from a leaf node and return value estimate."""
        if self.depth <= 1:
            return self._leaf_value(node.state_vec)
        # Depth > 1: unroll dynamics with a neutral action probe -- there
        # is no free-form candidate to sample/embed at a purely imagined
        # node, same approximation the source used for token actions.
        state = node.state_vec
        for _ in range(self.depth - 1):
            dummy_action = torch.zeros_like(state)
            state = self.wm.dynamics(state, dummy_action)
        return self._leaf_value(state)

    @torch.no_grad()
    def _expand(self, node: MCTSNode, candidates: list[TrajectoryStep],
                priors: list[float], embeddings: list[torch.Tensor]):
        """Expand leaf using sampled hypothesis candidates and their
        policy-derived priors."""
        for i, (cand, prior, emb) in enumerate(zip(candidates, priors, embeddings)):
            if i in node.children:
                continue
            next_state = self.wm.dynamics(node.state_vec, emb)
            node.children[i] = MCTSNode(
                state_vec=next_state, prior=prior, parent=node, action=cand,
            )

    def _select(self, root: MCTSNode) -> MCTSNode:
        """UCB tree walk to a leaf."""
        node = root
        while not node.is_leaf():
            node = max(node.children.values(), key=lambda n: n.ucb(self.c_puct))
        return node

    def _backprop(self, node: MCTSNode, value: float):
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    def sample_candidates(
        self, model, tokenizer, generate_fn: Callable[..., str], prompt: str,
        max_tokens: int = 300, device: str = "cpu",
    ) -> tuple[list[TrajectoryStep], list[float], list[torch.Tensor]]:
        """Sample `top_k` candidate Thought/Action continuations from
        *generate_fn*, and compute each one's pooled embedding
        (`proprioception.pooled_hidden_for_question`, reused as-is) and
        policy log-prob (`grpo.sequence_log_prob`, reused as-is).

        Each raw response is treated as one hypothesis-level action --
        finer-grained per-Thought/per-Action splitting within a single
        response is left to a caller that wants it (`grpo.parse_transcript`
        already does that split for training trajectories); this method's
        job is just producing the candidate set MCTS chooses among.
        """
        candidates: list[TrajectoryStep] = []
        log_probs: list[float] = []
        embeddings: list[torch.Tensor] = []
        for _ in range(self.top_k):
            response = generate_fn(prompt, max_tokens=max_tokens, think=True)
            candidates.append(TrajectoryStep(kind="thought", text=response))
            log_probs.append(
                sequence_log_prob(model, tokenizer, prompt, response, device=device).item()
            )
            embeddings.append(pooled_hidden_for_question(model, tokenizer, response).unsqueeze(0))

        priors = _candidate_priors(log_probs)
        return candidates, priors, embeddings

    def select_action(
        self, state_vec: torch.Tensor, model, tokenizer,
        generate_fn: Callable[..., str], prompt: str,
        max_tokens: int = 300, device: str = "cpu",
    ) -> TrajectoryStep:
        """
        Run MCTS over sampled hypothesis candidates and return the best one.

        Parameters
        ----------
        state_vec : (1, d_model) — current world state (context so far)
        model, tokenizer : the policy candidates are sampled from and
            scored against
        generate_fn : callable(prompt, max_tokens, think) -> str
        prompt : str — the context to condition candidate generation on
        """
        candidates, priors, embeddings = self.sample_candidates(
            model, tokenizer, generate_fn, prompt, max_tokens=max_tokens, device=device,
        )

        root = MCTSNode(state_vec=state_vec)
        self._expand(root, candidates, priors, embeddings)

        for _ in range(self.n_sims):
            leaf = self._select(root)
            if leaf.visit_count > 0 and leaf.is_leaf():
                # Re-expand with the same candidate set (approximation at
                # depth > 1, same as the source).
                self._expand(leaf, candidates, priors, embeddings)
                leaf = self._select(root)
            value = self._simulate(leaf)
            self._backprop(leaf, value)

        best_idx, best_score = 0, float("-inf")
        for i, prior in enumerate(priors):
            child = root.children.get(i)
            if child is None:
                continue
            log_prob = math.log(max(prior, 1e-12))
            score = (1 - self.alpha) * log_prob + self.alpha * child.value
            if score > best_score:
                best_score, best_idx = score, i

        return candidates[best_idx]


class NoPrecedentCheck:
    """Operational "no precedent" gate (todo.md, Item 8): on a failed MCTS
    search, measure distance from the query to a reference set of known
    hypothesis embeddings. Close = tune rollout budget. Far = widen Item
    1's corpus / Item 6's curriculum -- a diagnostic surfaced to a human,
    not an automated branch (todo.md is explicit that this shouldn't just
    throw more search compute at a coverage gap).

    Reuses `OODDetector`'s Mahalanobis-distance mechanics
    (`uchi/flux/verifier_model.py`) exactly as `proprioception.py` already
    does for question familiarity -- not a new distance metric, the same
    "is this near anything we've actually seen" question at Item 8's
    hypothesis-embedding granularity instead of proprioception's
    whole-question granularity.
    """

    def __init__(self, threshold: float = 3.0):
        # Unlike proprioception.py's OOD gate (which deliberately starts
        # inert at threshold=inf until calibrate_threshold() sets a real
        # value from percentile data -- a past reused-default bug there),
        # this check has no such history and OODDetector's own 3.0 default
        # (Mahalanobis distance in std-dev units, Lee et al.) is a
        # reasonable out-of-the-box operating point for a diagnostic that
        # is supposed to actually flag something by default.
        self.detector = OODDetector(threshold=threshold)

    def fit(self, reference_embeddings: torch.Tensor) -> None:
        """reference_embeddings: (N, d_model) pooled embeddings of known-
        good hypothesis segments (e.g. from Item 6's PASS-outcome branches)."""
        self.detector.fit(reference_embeddings)

    def check(self, query_embedding: torch.Tensor) -> dict:
        """Returns {"distance": float, "close": bool} -- diagnostic only,
        no action taken automatically."""
        distance = self.detector.distance(query_embedding)
        return {"distance": distance, "close": not self.detector.is_ood(query_embedding)}
