"""
FLUX + Uchi architecture tests.

Covers the current architecture: the `Uchi` public API (learn/ask compounding,
honest abstention), the pluggable Proposer seam, Generate-and-Ground, the skill
registry (analytical + overview + teach), the CLI helpers, and the REST API.

The autouse `no_flux_checkpoint` fixture (conftest) forces the FLUX proposer to
degrade to None, so these run fast on CPU and exercise the verifier / extractive /
abstention path independently of FLUX's trained weights.
"""
import numpy as np
import pytest


# ── Public API: compounding + human-readable I/O ──────────────────────────────
def test_import_and_construct():
    from uchi import Uchi
    u = Uchi()
    assert hasattr(u, "learn") and hasattr(u, "ask") and hasattr(u, "ingest")


def test_learn_and_ask_are_strings():
    """ask() always returns a str; learn() accepts a str (the compounding contract)."""
    from uchi import Uchi
    u = Uchi()
    u.learn("The Eiffel Tower is a wrought-iron lattice tower in Paris, France.")
    ans = u.ask("What is the Eiffel Tower?")
    assert isinstance(ans, str) and ans


def test_honest_abstention_on_nonsense():
    """With no grounding, Uchi abstains instead of confabulating."""
    from uchi import Uchi
    u = Uchi()
    ans = u.ask("What is the flprofnak of a quzzle on planet Xryttle?")
    assert isinstance(ans, str)
    assert "grounded" in ans.lower() or "don't" in ans.lower() or "sorry" in ans.lower()


def test_ask_output_feeds_learn():
    """The output of one instance's ask() is a valid learn() input for another."""
    from uchi import Uchi
    u1 = Uchi()
    report = u1.ask("/classify", X=np.random.default_rng(0).normal(size=(40, 3)),
                    y=(np.arange(40) % 2))
    u2 = Uchi()
    u2.learn(report)          # must not raise — strings compound across instances
    assert isinstance(report, str) and "Classification" in report


# ── Pluggable Proposer seam ───────────────────────────────────────────────────
def test_proposer_protocol_and_graceful_degradation():
    from uchi.proposer import FluxProposer, DecoderProposer, load_proposer
    # No checkpoint / no decoder path → degrades to None (extractive fallback).
    assert load_proposer(prefer="flux", decoder_path=None) is None
    assert hasattr(FluxProposer, "propose") and hasattr(DecoderProposer, "propose")


# ── Generate-and-Ground verifier ──────────────────────────────────────────────
def test_generate_and_ground_abstains_without_evidence():
    import numpy as np
    from uchi.retrieval import SemanticIndex
    from uchi.oracle import FactCheckOracle
    from uchi.generate_and_ground import GenerateAndGround
    idx = SemanticIndex({}, np.zeros((1, 1), dtype=np.float32))
    gg = GenerateAndGround(index=idx, oracle=FactCheckOracle(), proposer=None)
    out = gg.answer("anything at all")
    assert isinstance(out, str) and out


# ── Skill registry ────────────────────────────────────────────────────────────
def test_overview_skill():
    from uchi import Uchi
    out = Uchi().ask("/overview")
    assert "Knowledge Overview" in out and "FLUX proposer" in out


@pytest.mark.parametrize("mode", ["classify", "regress", "anomaly"])
def test_tabular_analytical_skills(mode):
    from uchi import Uchi
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 4)); y = (X[:, 0] > 0).astype(int)
    out = Uchi().ask(f"/{mode}", X=X, y=y)
    assert isinstance(out, str) and "complete" in out.lower()


def test_forecast_skill():
    from uchi import Uchi
    ts = np.cumsum(np.random.default_rng(2).normal(size=(50, 2)), axis=0)
    out = Uchi().ask("/forecast", X=ts, steps=4)
    assert "Forecast complete" in out


def test_teach_skill():
    from uchi import Uchi
    out = Uchi().ask("/teach what is gravity | the attraction between masses")
    assert "Learned" in out


# ── Advanced SDK predictor ────────────────────────────────────────────────────
def test_predictor_sequence_api():
    from uchi import Uchi
    u = Uchi()
    for _ in range(3):
        u.predictor.train(["a", "b", "c", "d"])
    assert u.predictor.predict_next(["b", "c"]) == "d"


# ── CLI helpers work against a Uchi instance ──────────────────────────────────
def test_cli_helpers(tmp_path):
    from uchi import Uchi
    from uchi.cli import ingest_file, save_brain
    u = Uchi()
    f = tmp_path / "note.txt"
    f.write_text("Photosynthesis converts sunlight into chemical energy.")
    ingest_file(u, str(f), quiet=True)          # must not raise
    brain = tmp_path / "brain.uchi"
    save_brain(u, str(brain))
    assert brain.exists() and brain.stat().st_size > 0


# ── REST API ──────────────────────────────────────────────────────────────────
def test_rest_api_endpoints():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from uchi.api_server import app
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        skills = client.get("/skills").json()["skills"]
        assert any(s["name"] == "classify" for s in skills)
        r = client.post("/ask", json={"query": "What is the capital of France?"})
        assert r.status_code == 200 and "answer" in r.json()
