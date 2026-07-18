"""Tests for uchi/world_model.py (0.5.0 Item 8, part 1).

Pure nn.Module shape/gradient tests against synthetic tensors, plus a
tiny stub "model" (embedding + norm_f, no real SSM layers -- `train_dynamics`
et al only ever touch `.embedding`/`.layers`/`.norm_f` by name, same
attributes HybridTSSM exposes) so DynamicsHead/ValueHead training can be
exercised end-to-end without a trained FLUX checkpoint.
"""
import torch

from uchi.world_model import DynamicsHead, ValueHead, WorldModel, extract_state


class _StubModel(torch.nn.Module):
    """Minimal stand-in for HybridTSSM: real embedding + norm_f, no SSM
    layers (empty list -- the training loops only ever do
    `for layer in model.layers: hidden, _ = layer(hidden)`, which is a
    no-op on an empty list, so this is a legitimate stub of the same
    attribute shape, not a shortcut around it)."""

    def __init__(self, vocab_size: int = 32, d_model: int = 8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.layers: list = []
        self.norm_f = torch.nn.LayerNorm(d_model)


def test_dynamics_head_shape():
    head = DynamicsHead(d_model=16)
    state = torch.randn(3, 16)
    action = torch.randn(3, 16)
    out = head(state, action)
    assert out.shape == (3, 16)


def test_value_head_shape():
    head = ValueHead(d_model=16)
    state = torch.randn(3, 16)
    out = head(state)
    assert out.shape == (3, 1)


def test_world_model_predict_next_and_rollout_shapes():
    wm = WorldModel(d_model=16)
    state = torch.randn(1, 16)
    action = torch.randn(1, 16)
    next_state, value = wm.predict_next(state, action)
    assert next_state.shape == (1, 16)
    assert value.shape == (1,)

    trajectory = wm.rollout(state, [action, action, action])
    assert len(trajectory) == 3
    for s, v in trajectory:
        assert s.shape == (1, 16)


def test_extract_state_shape():
    model = _StubModel(d_model=8)
    ids = torch.tensor([[1, 2, 3, 4]])
    state = extract_state(model, ids)
    assert state.shape == (1, 8)


def test_update_value_from_state_moves_prediction_toward_reward():
    wm = WorldModel(d_model=8)
    state = torch.randn(1, 8)

    with torch.no_grad():
        before = wm.value(state).item()
    for _ in range(200):
        wm.update_value_from_state(state, reward=1.0, device="cpu", lr=0.05)
    with torch.no_grad():
        after = wm.value(state).item()

    assert abs(after - 1.0) < abs(before - 1.0), "repeated updates should move V(state) toward the target reward"


def test_train_dynamics_reduces_loss_over_a_real_sequence():
    torch.manual_seed(0)
    model = _StubModel(vocab_size=32, d_model=8)
    model.eval()
    wm = WorldModel(d_model=8)

    token_sequences = [[i % 32 for i in range(j, j + 40)] for j in range(5)]

    # One step's loss before any training, vs after -- same underlying
    # per-chunk MSE computed inside train_dynamics, just sampled here to
    # confirm the loop actually reduces it rather than trusting a printed log.
    def _one_chunk_loss():
        ids = torch.tensor([token_sequences[0][:32]], dtype=torch.long)
        with torch.no_grad():
            hidden = model.embedding(ids)
            for layer in model.layers:
                hidden, _ = layer(hidden)
            states = model.norm_f(hidden)
            tok_embeds = model.embedding(ids)
        n = states.size(1) - 1
        state_t, action_t, target = states[0, :n], tok_embeds[0, 1:n + 1], states[0, 1:n + 1]
        with torch.no_grad():
            pred = wm.dynamics(state_t, action_t)
            return torch.nn.functional.mse_loss(pred, target).item()

    loss_before = _one_chunk_loss()
    wm.train_dynamics(model, token_sequences, device="cpu", steps=150, lr=1e-3, seq_len=32)
    loss_after = _one_chunk_loss()

    assert loss_after < loss_before


def test_world_model_save_load_round_trip(tmp_path):
    wm = WorldModel(d_model=8)
    path = str(tmp_path / "world_model.pt")
    wm.save(path)

    wm2 = WorldModel(d_model=8)
    loaded = wm2.load("cpu", path)
    assert loaded is True

    state = torch.randn(1, 8)
    action = torch.randn(1, 8)
    with torch.no_grad():
        out1, v1 = wm.predict_next(state, action)
        out2, v2 = wm2.predict_next(state, action)
    assert torch.allclose(out1, out2)
    assert torch.allclose(v1, v2)


def test_world_model_load_missing_file_returns_false(tmp_path):
    wm = WorldModel(d_model=8)
    assert wm.load("cpu", str(tmp_path / "nonexistent.pt")) is False
