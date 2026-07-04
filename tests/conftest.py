"""
Shared pytest fixtures.

Architecture note: the legacy OmniRouter (and its _bootstrap_knowledge /
_bootstrap_persona hooks) was removed. `Uchi` now boots a FLUX Proposer behind
the Generate-and-Ground verifier. For tests we force the proposer to degrade to
None so construction is fast, deterministic, and does not load a checkpoint onto
the GPU (which would contend with any live training run). This exercises the
verifier + extractive/abstention path independently of FLUX's trained weights.
"""
import pytest


@pytest.fixture(autouse=True)
def no_flux_checkpoint():
    """Force FluxProposer.load() -> None so Uchi() builds without a GPU checkpoint."""
    try:
        from unittest.mock import patch
        with patch("uchi.proposer.FluxProposer.load", return_value=None):
            yield
    except Exception:
        # If the proposer module is unavailable for some reason, don't block tests.
        yield
