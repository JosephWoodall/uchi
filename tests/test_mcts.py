"""Tests for uchi/mcts.py (0.5.0 Item 8, part 2).

Stub dynamics/value function with a deterministic, hand-checkable reward
landscape (same spirit as testing react_agent.py's parser against known
inputs rather than a live model) verifying `MCTS.select_action` actually
prefers the higher-value branch, plus tests for the PUCT node mechanics
and the "no precedent" diagnostic in isolation.
"""
import torch

from uchi.grpo import TrajectoryStep
from uchi.mcts import MCTS, MCTSNode, NoPrecedentCheck, _candidate_priors


class _StubWorldModel:
    """Deterministic stand-in for WorldModel: `dynamics` is the identity
    on the action embedding (next state = the candidate's own embedding),
    and `value` is that state's coordinate sum -- so whichever candidate
    embedding has the largest sum is, by construction, the correct answer
    the search should converge on."""

    def dynamics(self, state_vec: torch.Tensor, action_embed: torch.Tensor) -> torch.Tensor:
        return action_embed

    def value(self, state_vec: torch.Tensor) -> torch.Tensor:
        return state_vec.sum(dim=-1, keepdim=True)


def test_mcts_node_ucb_root_is_zero():
    root = MCTSNode(state_vec=torch.zeros(1, 4))
    assert root.ucb() == 0.0


def test_mcts_node_is_leaf_until_expanded():
    root = MCTSNode(state_vec=torch.zeros(1, 4))
    assert root.is_leaf()
    root.children[0] = MCTSNode(state_vec=torch.zeros(1, 4), parent=root)
    assert not root.is_leaf()


def test_candidate_priors_sum_to_one():
    priors = _candidate_priors([1.0, 2.0, 0.5])
    assert abs(sum(priors) - 1.0) < 1e-6
    assert all(p > 0 for p in priors)


def test_select_action_prefers_higher_value_branch(monkeypatch):
    wm = _StubWorldModel()
    # Priors are perfectly tied (1/3 each) so this isolates value's effect
    # on selection -- but PUCT's exploration term only grows as sqrt(n),
    # so overcoming the arbitrary index-0 tie-break on the very first,
    # completely uninformed decision needs enough simulations for
    # sqrt(root.visit_count) to outweigh the early leader's head start
    # (empirically ~65+ here; 100 leaves margin). This is a real property
    # of PUCT with symmetric priors, not a port bug -- real LM-derived
    # priors are essentially never exactly tied, which is why the source's
    # own default of 8 simulations was never an issue there.
    mcts = MCTS(world_model=wm, n_simulations=100, top_k=3, depth=1, alpha=0.4, c_puct=1.5)

    candidates = [TrajectoryStep(kind="thought", text=f"cand{i}") for i in range(3)]
    priors = [1 / 3, 1 / 3, 1 / 3]
    embeddings = [
        torch.full((1, 4), 1.0),   # value = 4
        torch.full((1, 4), 5.0),   # value = 20 -- the winner
        torch.full((1, 4), -2.0),  # value = -8
    ]

    monkeypatch.setattr(mcts, "sample_candidates", lambda *a, **kw: (candidates, priors, embeddings))

    state_vec = torch.zeros(1, 4)
    best = mcts.select_action(
        state_vec, model=None, tokenizer=None,
        generate_fn=lambda *a, **kw: "", prompt="irrelevant, sample_candidates is stubbed",
    )
    assert best.text == "cand1"


def test_select_action_respects_depth_gt_1_rollout(monkeypatch):
    # depth=2 exercises the dynamics-chain branch in _simulate rather than
    # a bare leaf value -- with the identity dynamics stub, an extra
    # zero-action dynamics step maps every candidate's state toward the
    # zero vector, collapsing values together. Just confirm it runs and
    # returns one of the real candidates, not that ranking is preserved
    # (the module docstring is explicit this is an approximation).
    wm = _StubWorldModel()
    mcts = MCTS(world_model=wm, n_simulations=4, top_k=2, depth=2, alpha=0.4, c_puct=1.5)

    candidates = [TrajectoryStep(kind="thought", text="a"), TrajectoryStep(kind="thought", text="b")]
    priors = [0.5, 0.5]
    embeddings = [torch.full((1, 4), 1.0), torch.full((1, 4), 2.0)]
    monkeypatch.setattr(mcts, "sample_candidates", lambda *a, **kw: (candidates, priors, embeddings))

    best = mcts.select_action(
        torch.zeros(1, 4), model=None, tokenizer=None,
        generate_fn=lambda *a, **kw: "", prompt="p",
    )
    assert best.text in ("a", "b")


def test_no_precedent_check_flags_far_query():
    check = NoPrecedentCheck()
    reference = torch.randn(50, 8) * 0.1 + torch.ones(8)  # tight cluster around ones
    check.fit(reference)

    close_query = torch.ones(8) + 0.05
    far_query = torch.ones(8) * 50.0

    close_result = check.check(close_query)
    far_result = check.check(far_query)

    assert close_result["close"] is True
    assert far_result["close"] is False
    assert far_result["distance"] > close_result["distance"]


def test_no_precedent_check_unfitted_is_never_far():
    check = NoPrecedentCheck()
    result = check.check(torch.randn(8))
    assert result["close"] is True
