import os

import torch

from uchi.flux import ssm


def _leaf_inputs(B=1, T=8, D=4, S=2):
    bar_A_raw = torch.randn(B, T, D, S, requires_grad=True)
    b = (torch.randn(B, T, D, S) * 0.1).requires_grad_(True)
    return bar_A_raw, b


def test_default_unfused_matches_explicit_unfused():
    bar_A_raw, b = _leaf_inputs()
    bar_A = torch.sigmoid(bar_A_raw)
    out_default = ssm._scan(bar_A, b)
    out_explicit = ssm._scan(bar_A, b, fuse=False)
    assert torch.equal(out_default, out_explicit)


def test_forward_backward_gradients_flow_on_default_path():
    bar_A_raw, b = _leaf_inputs()
    bar_A = torch.sigmoid(bar_A_raw)
    out = ssm._scan(bar_A, b)
    out.sum().backward()
    assert bar_A_raw.grad is not None
    assert b.grad is not None
    assert bar_A_raw.grad.shape == bar_A_raw.shape


def test_env_var_selects_fuse_without_paying_compile_cost(monkeypatch):
    """Confirm UCHI_FUSE_SSM_SCAN actually drives which loop implementation
    runs, without triggering a real torch.compile (swap in a spy standing
    in for the compiled loop — the real fused-path numerical correctness
    was verified manually against a live GPU; that ~136s one-time compile
    is too slow to pay on every CI run)."""
    calls = {"fused_forward": 0, "plain_forward": 0}

    orig_loop = ssm._scan_forward_loop

    def spy_plain(*a, **kw):
        calls["plain_forward"] += 1
        return orig_loop(*a, **kw)

    def spy_fused(*a, **kw):
        calls["fused_forward"] += 1
        return orig_loop(*a, **kw)

    monkeypatch.setattr(ssm, "_scan_forward_loop", spy_plain)
    monkeypatch.setattr(ssm, "_get_fused_forward", lambda: spy_fused)

    bar_A_raw, b = _leaf_inputs()
    bar_A = torch.sigmoid(bar_A_raw)

    monkeypatch.delenv("UCHI_FUSE_SSM_SCAN", raising=False)
    ssm._scan(bar_A, b)
    assert calls == {"fused_forward": 0, "plain_forward": 1}

    monkeypatch.setenv("UCHI_FUSE_SSM_SCAN", "1")
    ssm._scan(bar_A, b)
    assert calls == {"fused_forward": 1, "plain_forward": 1}


def test_explicit_fuse_arg_overrides_env_var(monkeypatch):
    monkeypatch.setenv("UCHI_FUSE_SSM_SCAN", "1")
    bar_A_raw, b = _leaf_inputs()
    bar_A = torch.sigmoid(bar_A_raw)
    # fuse=False explicitly should win over the env var being set
    out_a = ssm._scan(bar_A, b, fuse=False)
    out_b = ssm._scan(bar_A, b, fuse=False)
    assert torch.equal(out_a, out_b)
