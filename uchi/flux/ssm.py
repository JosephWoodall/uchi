import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Sequential scan with custom backward ──────────────────────────────────────

def _scan_forward_loop(bar_A, b):
    B, T, D, S = b.shape
    h = b.new_zeros(B, D, S)
    hs = []
    for t in range(T):
        h = bar_A[:, t] * h + b[:, t]
        hs.append(h)
    return torch.stack(hs, dim=1)


def _scan_backward_loop(bar_A, h_all, grad_h):
    B, T, D, S = bar_A.shape
    grad_b     = torch.empty_like(h_all)
    grad_bar_A = torch.empty_like(bar_A)
    h_minus1   = h_all.new_zeros(B, D, S)   # h[-1] = 0 by SSM convention
    G          = h_all.new_zeros(B, D, S)   # G[T] = 0 (no future)

    for t in range(T - 1, -1, -1):
        # G[t] = grad_h[t] + bar_A[t+1] * G[t+1]
        # At t=T-1, G[T]=0 so the second term vanishes (initial G is zeros).
        G = grad_h[:, t] + (bar_A[:, t + 1] * G if t < T - 1 else G)
        grad_b[:, t]     = G
        h_prev           = h_all[:, t - 1] if t > 0 else h_minus1
        grad_bar_A[:, t] = G * h_prev
    return grad_bar_A, grad_b


# 0.4.0 Item 0: kernel-fused variants of the same two loops above, via
# torch.compile over the *whole* loop (not per-step — per-step compilation
# was measured slower, see the module docstring below). Lazily compiled on
# first use so importing this module never pays the ~2min compile cost;
# only opting into fuse=True during training does.
_fused_forward_loop = None
_fused_backward_loop = None


def _get_fused_forward():
    global _fused_forward_loop
    if _fused_forward_loop is None:
        _fused_forward_loop = torch.compile(_scan_forward_loop, dynamic=False)
    return _fused_forward_loop


def _get_fused_backward():
    global _fused_backward_loop
    if _fused_backward_loop is None:
        _fused_backward_loop = torch.compile(_scan_backward_loop, dynamic=False)
    return _fused_backward_loop


class _SeqScanFunction(torch.autograd.Function):
    """
    Sequential SSM scan: h[t] = bar_A[t] * h[t-1] + b[t]

    WHY sequential instead of parallel:
    For our shapes (T=1024, D=1024, S=32, B=2, bf16), the parallel prefix scan
    runs 10 strides × 6 large ops × 134MB tensors = 8GB memory traffic per call,
    and the parallel adjoint backward spawns 6 extra 134MB allocations before
    running another full scan. Sequential replaces all of that with a single
    sequential pass over ~3 × 134MB — 1.65× faster (profiled: 28.7ms vs 47.5ms).

    Backward uses the sequential adjoint (reverse pass, no extra scan):
        G[t] = grad_h[t] + bar_A[t+1] * G[t+1]
        grad_b[t]     = G[t]
        grad_bar_A[t] = G[t] * h[t-1]

    Saves (bar_A, h_all): 2×134MB per call (bf16).
    Peak backward memory: ~536MB per call vs ~942MB for the parallel adjoint.

    0.4.0 Item 0 — fuse=True kernel-fuses this same sequential algorithm
    via torch.compile over the *entire* loop (not the algorithm choice
    above, which stays sequential — measured against a parallel scan
    already, see above). Per-step compilation of the inner `h = A*h+b`
    update was measured *slower* (94.8ms vs 33.8ms eager) — 1024 separate
    compiled-function calls pay dispatch/guard overhead T times over.
    Compiling the whole T=1024-step loop as one graph instead gives a
    genuine 3.08× forward speedup (28.9ms → 9.4ms, RTX 5070, bf16,
    B=2/T=1024/D=768/S=32), at a one-time ~136s compile cost per input
    shape — worth it once amortized over a multi-hour training run's
    thousands of steps, not for interactive single-token decode with
    varying shapes (which would thrash recompilation instead). Verified
    numerically consistent with the eager loop to within bf16 tolerance
    (max abs diff 0.0059 on the reference shape above). Opt-in via
    fuse=True (defaults off — this doesn't change any existing behavior
    or trained-checkpoint compatibility unless explicitly enabled), or
    globally via the ``UCHI_FUSE_SSM_SCAN=1`` environment variable for
    the training entrypoints in ``train_v2.py`` / ``qat_train.py`` /
    ``sft_train.py``.
    """

    @staticmethod
    def forward(ctx, bar_A, b, fuse=False):
        # bar_A: (B, T, D, S)  b: (B, T, D, S)
        with torch.no_grad():
            loop = _get_fused_forward() if fuse else _scan_forward_loop
            h_all = loop(bar_A, b)
        ctx.save_for_backward(bar_A, h_all)
        ctx.fuse = fuse
        return h_all

    @staticmethod
    def backward(ctx, grad_h):
        bar_A, h_all = ctx.saved_tensors
        grad_h = grad_h.to(h_all.dtype)

        with torch.no_grad():
            loop = _get_fused_backward() if ctx.fuse else _scan_backward_loop
            grad_bar_A, grad_b = loop(bar_A, h_all, grad_h)

        return grad_bar_A.to(grad_h.dtype), grad_b.to(grad_h.dtype), None


def _scan(bar_A, b, fuse=None):
    """Entry point for the sequential SSM scan.

    *fuse* controls the Item 0 kernel-fused (torch.compile) path — see
    ``_SeqScanFunction`` above. Defaults to the ``UCHI_FUSE_SSM_SCAN``
    environment variable (off unless set), so existing call sites and
    trained checkpoints are unaffected unless a training entrypoint opts
    in explicitly.
    """
    if fuse is None:
        fuse = os.environ.get("UCHI_FUSE_SSM_SCAN", "") == "1"
    return _SeqScanFunction.apply(bar_A, b, fuse)


# ── SSM module ─────────────────────────────────────────────────────────────────

class SSMCell(nn.Module):
    """
    Selective Dual-Resolution State SSM (S6-DRS-SSM).

    Four architectural fixes over the original DRS-SSM:

    Gap 1 — Selectivity: dt = softplus(dt_proj(x_t)) is now per-token, making
    bar_A[t] = exp(-dt[t] * A) content-dependent. Each token controls its own
    decay — the core S6/Mamba mechanism for selective state updates.

    Gap 4 — Timescale specialization: B_fast and B_slow are separate parameters.
    Fast and slow channels learn different "what to encode" strategies rather
    than sharing a single B projection that must serve both timescales.

    Retained: dual-scan A_fast/A_slow with HiPPO-spaced eigenvalues, bilinear
    input gate W, input-dependent blend gate, C/D output projection, O(1) decode.

    d_state reduced 64 → 32: dual-scan gives 2×32 = 64 effective state slots,
    with better coverage (specialized fast + slow) vs 64 shared slots.

    Speed note: all A/B/W/C/D parameters are cast to dt's dtype (bf16 under
    autocast) so scan tensors are 134MB instead of 268MB — halves memory
    bandwidth pressure on the sequential scan.
    """

    def __init__(self, d_model, d_state=32, a_fast_range=(0.0, 2.0), a_slow_range=(2.0, 3.5)):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # A initialization in a learnable regime.
        # Prior bug: linspace(3,8) gave bar_A < 0.1 at mean dt — state reset every step.
        # Model drove A to -5 (softplus saturation, sigmoid_grad=0.007) within 100 steps
        # to get any memory, destroying all timescale structure and freezing parameters.
        #
        # Default ranges (a_fast_range=(0,2.0), a_slow_range=(2.0,3.5)) keep
        # sigmoid_grad >= 0.05 (learnable) and bar_A in useful range, TUNED FOR
        # max_seq_len=256:
        #   fast: A in [0, -2.0] -> bar_A in [0.57, 0.90], horizon 2-10 steps
        #   slow: A in [-2.0, -3.5] -> bar_A in [0.90, 0.97], horizon 10-34 steps
        # Long-range context (beyond the slow horizon) is handled by the
        # attention layers. 0.5.0 Item 2: a checkpoint trained at a
        # different max_seq_len should pass ranges scaled proportionally
        # (e.g. train_v2.py computes fast/slow ranges from --seq-len) --
        # reusing the 256-token ranges at 1024 tokens would leave the SSM
        # state horizon covering under 3% of the context instead of the
        # ~13% it was tuned for, pushing far more long-range burden onto
        # the attention layers than they were sized for.
        fast_init = -torch.linspace(a_fast_range[0], a_fast_range[1], d_state).unsqueeze(0).expand(d_model, -1)
        slow_init = -torch.linspace(a_slow_range[0], a_slow_range[1], d_state).unsqueeze(0).expand(d_model, -1)
        self.A_fast = nn.Parameter(fast_init.clone())
        self.A_slow = nn.Parameter(slow_init.clone())

        # Separate input projections per timescale (Gap 4 fix)
        # Init at 0.1 (was 0.01): prevents the B-explosion trajectory where tiny B
        # forces the model to collapse A → -5 just to accumulate any useful state.
        self.B_fast = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.B_slow = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.W      = nn.Parameter(torch.randn(d_model, d_state) * 0.1)

        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))

        self.blend_proj = nn.Linear(d_model, d_state, bias=False)
        nn.init.zeros_(self.blend_proj.weight)

        # Per-token dt: rank-factored projection d_model → dt_rank → d_model.
        # Rank-64 uses 131K params/layer, 8× fewer FLOPS than full d_model×d_model.
        dt_rank = max(d_model // 16, 16)
        self.dt_proj_in  = nn.Linear(d_model, dt_rank, bias=False)
        self.dt_proj_out = nn.Linear(dt_rank, d_model, bias=True)
        nn.init.uniform_(self.dt_proj_out.bias, -4.0, -1.0)
        nn.init.normal_(self.dt_proj_in.weight,  std=0.01)
        nn.init.normal_(self.dt_proj_out.weight, std=0.01)

    def forward(self, x):
        """
        x : (batch, T, d_model)
        Returns: output (batch, T, d_model), final_h (batch, 2, d_model, d_state)
        """
        # Per-token selectivity: dt is content-dependent (S6 mechanism).
        # x arrives as bf16 from in_proj (BitLinear under autocast).
        # F.softplus runs in float32 even under autocast (not in the whitelist),
        # so we explicitly cast dt back to x.dtype to keep scan tensors at 134MB (bf16)
        # instead of 268MB (float32), halving memory bandwidth on the sequential scan.
        # x is bf16 when coming from in_proj under torch.autocast.
        # F.softplus and torch.exp are promoted to float32 by autocast for stability,
        # so we cast back to x.dtype after each to keep scan tensors at 134MB (bf16)
        # rather than 268MB (float32), halving memory bandwidth on the scan loops.
        scan_dtype = x.dtype   # bf16 under autocast, float32 otherwise
        dt = F.softplus(self.dt_proj_out(self.dt_proj_in(x))).to(scan_dtype).unsqueeze(-1)  # (B, T, D, 1)

        A_fast_sp = F.softplus(self.A_fast).to(scan_dtype)   # (D, S)
        A_slow_sp = F.softplus(self.A_slow).to(scan_dtype)
        # exp promotes to fp32 under autocast; cast back so bar_A stays in scan_dtype
        bar_A_fast = torch.exp(-dt * A_fast_sp).to(scan_dtype)   # (B, T, D, S)
        bar_A_slow = torch.exp(-dt * A_slow_sp).to(scan_dtype)

        x_exp    = x.unsqueeze(-1)                              # (B, T, D, 1)
        bar_W    = dt * self.W.to(scan_dtype)                  # (B, T, D, S)
        bilinear = bar_W * torch.tanh(x_exp)                   # tanh stays bf16
        b_fast   = (dt * self.B_fast.to(scan_dtype) + bilinear) * x_exp
        b_slow   = (dt * self.B_slow.to(scan_dtype) + bilinear) * x_exp

        h_fast = _scan(bar_A_fast, b_fast)
        h_slow = _scan(bar_A_slow, b_slow)

        # Input-dependent blend: when to trust fast vs slow memory
        blend = torch.sigmoid(self.blend_proj(x)).unsqueeze(-2)   # (B, T, 1, S)
        h     = blend * h_fast + (1.0 - blend) * h_slow           # (B, T, D, S)

        output  = (self.C.to(scan_dtype) * h).sum(dim=-1) + self.D.to(scan_dtype) * x
        final_h = torch.stack([h_fast[:, -1], h_slow[:, -1]], dim=1)
        return output, final_h

    def step(self, x_t, h_prev):
        """
        O(1) autoregressive decode step.

        x_t    : (batch, d_model)
        h_prev : (batch, 2, d_model, d_state)  — [fast, slow]
        Returns: y_t (batch, d_model), new_h (batch, 2, d_model, d_state)
        """
        scan_dtype = x_t.dtype
        dt = F.softplus(self.dt_proj_out(self.dt_proj_in(x_t))).to(scan_dtype).unsqueeze(-1)
        bar_A_fast = torch.exp(-dt * F.softplus(self.A_fast).to(scan_dtype)).to(scan_dtype)
        bar_A_slow = torch.exp(-dt * F.softplus(self.A_slow).to(scan_dtype)).to(scan_dtype)

        x_exp    = x_t.unsqueeze(-1)
        bar_W    = dt * self.W.to(scan_dtype)
        bilinear = bar_W * torch.tanh(x_exp)
        b_fast   = (dt * self.B_fast.to(scan_dtype) + bilinear) * x_exp
        b_slow   = (dt * self.B_slow.to(scan_dtype) + bilinear) * x_exp

        h_fast_t = bar_A_fast * h_prev[:, 0].to(scan_dtype) + b_fast
        h_slow_t = bar_A_slow * h_prev[:, 1].to(scan_dtype) + b_slow

        blend = torch.sigmoid(self.blend_proj(x_t)).unsqueeze(-2)
        h_t   = blend * h_fast_t + (1.0 - blend) * h_slow_t

        y_t   = (self.C.to(scan_dtype) * h_t).sum(dim=-1) + self.D.to(scan_dtype) * x_t
        new_h = torch.stack([h_fast_t, h_slow_t], dim=1)
        return y_t, new_h

    def prefill(self, x):
        return self.forward(x)

    def decode_step(self, x_t, cached_h):
        return self.step(x_t, cached_h)


# ── Cache ──────────────────────────────────────────────────────────────────────

class SSMCache:
    """Per-layer hidden state cache for stateful generation."""

    def __init__(self):
        self.layer_states = {}

    def set(self, layer_idx: int, state: torch.Tensor):
        self.layer_states[layer_idx] = state.clone()

    def get(self, layer_idx: int):
        return self.layer_states.get(layer_idx, None)

    def clear(self):
        self.layer_states.clear()

    @property
    def num_layers(self):
        return len(self.layer_states)
