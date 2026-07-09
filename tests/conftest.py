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


def pytest_collection_modifyitems(config, items):
    """`eval` is declared in pyproject.toml as "excluded from normal CI", but
    a marker declaration alone doesn't skip anything — pytest still collects
    and runs marked tests unless something acts on the marker. This is that
    something: skip `eval`-marked tests unless the run explicitly asked for
    them via `-m eval` (or any `-m` expression mentioning `eval`).
    """
    markexpr = config.getoption("-m", default="")
    if "eval" in markexpr:
        return
    skip_eval = pytest.mark.skip(reason="eval: needs live network — run explicitly with `pytest -m eval`")
    for item in items:
        if "eval" in item.keywords:
            item.add_marker(skip_eval)
