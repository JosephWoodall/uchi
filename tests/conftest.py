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
         patch("uchi.omni_router.OmniRouter._bootstrap_persona"):
        yield
