"""
Shared pytest fixtures and patches.

Patches:
- OmniRouter._bootstrap_knowledge  — skips HuggingFace downloads
- SSM update_value / train_dynamics — skips slow backward passes while still
  allowing _bootstrap_persona to populate the trie with conversation turns
"""
import pytest
import torch
from unittest.mock import patch, MagicMock


def _zero_loss(*args, **kwargs):
    """Stub returning a differentiable zero so (v_loss + d_loss).backward() is a no-op."""
    return torch.zeros(1, requires_grad=True)


@pytest.fixture(autouse=True)
def patch_omni_bootstraps():
    """Speed up OmniRouter creation: skip knowledge downloads and SSM training."""
    with patch("uchi.omni_router.OmniRouter._bootstrap_knowledge"), \
         patch("uchi.neuro_symbolic.StateSpaceModel.update_value", _zero_loss), \
         patch("uchi.neuro_symbolic.StateSpaceModel.train_dynamics", _zero_loss):
        yield
