"""
Shared pytest fixtures and patches.

Patches:
- OmniRouter._bootstrap_knowledge  — skips HuggingFace downloads
- OmniRouter._bootstrap_persona    — skips AssociativeMemory flooding (~14s loop)
- SSM update_value / train_dynamics — kept for backward compat with any callers
"""
import pytest
import torch
from unittest.mock import patch, MagicMock


def _zero_loss(*args, **kwargs):
    """Stub returning a differentiable zero so (v_loss + d_loss).backward() is a no-op."""
    return torch.zeros(1, requires_grad=True)


@pytest.fixture(autouse=True)
def patch_omni_bootstraps():
    """Speed up OmniRouter creation: skip bootstraps and legacy SSM update hooks."""
    with patch("uchi.omni_router.OmniRouter._bootstrap_knowledge"), \
         patch("uchi.omni_router.OmniRouter._bootstrap_persona"), \
         patch("uchi.neuro_symbolic.StateSpaceModel.update_value", _zero_loss), \
         patch("uchi.neuro_symbolic.StateSpaceModel.train_dynamics", _zero_loss):
        yield


@pytest.fixture
def fast_convergent():
    """
    Mock ConvergentEngine.generate to return an instant stub result.

    Apply to any test that exercises OmniRouter.chat() but does not need to
    test the MCTS loop itself — prevents the full rollout budget from running
    and keeps test suites under ~2 minutes.

    Usage:
        def test_something(fast_convergent):
            router = OmniRouter(use_bpe=False)
            reply = router.chat("hello")
            ...
    """
    with patch(
        "uchi.convergent_engine.ConvergentEngine.generate",
        return_value=("text", ["mock", "response"], 0.5),
    ):
        yield
