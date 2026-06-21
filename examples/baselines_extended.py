"""
baselines_extended.py
=====================
Additional online-prediction baselines for the extended IEEE benchmark.

Classes
-------
KNPredictor          — Interpolated Kneser-Ney N-gram (Chen & Goodman 1998)
PPMStarPredictor     — PPM* with high max_order (= unbounded-order PPM)
OnlineLSTMPredictor  — Single-layer LSTM with BPTT-1 (numpy; no PyTorch)

All share the same interface as baselines.py:
    predict()   -> (value, confidence)   called BEFORE observe()
    observe(v)  -> None                  appends v to history
    feedback(v) -> None                  updates model
"""

from collections import defaultdict
from typing import Any

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# INTERPOLATED KNESER-NEY N-GRAM
# ══════════════════════════════════════════════════════════════════════════════

class KNPredictor:
    """
    Interpolated Kneser-Ney N-gram  (Chen & Goodman, 1998).

    At the highest order (max_order), raw n-gram counts are used with
    absolute discount D.  At all lower orders, continuation counts are used:

        KN_cnt(ctx, w)  = |{u : raw(u·ctx, w) > 0}|

    The unigram (order 0) is a pure continuation probability:

        P_cont(w)  =  cont_cnt(w)  /  Σ_{w'} cont_cnt(w')

    where cont_cnt(w) = number of distinct 1-gram left-contexts u such that
    bigram (u, w) has been seen at least once.

    Online update rule:  when the (k+1)-gram (ctx, w) is first observed
    (raw count transitions 0 → 1), the continuation count at depth k-1 for
    the suffix context ctx[1:] incremented by 1.  This propagates up to the
    unigram level, giving exact on-the-fly continuation-count maintenance.

    discount : fixed D ≈ 0.75 (Ney's formula requires batch count statistics
               unavailable online; 0.75 is a standard online approximation).
    """

    def __init__(self, max_order: int = 5, discount: float = 0.75):
        self.max_order = max_order
        self.discount  = discount
        self.history:  list = []
        self._vocab:   set  = set()

        # _raw[k][ctx_tuple][w]  = raw count  (k = 1..max_order)
        self._raw:  list = [defaultdict(lambda: defaultdict(int))
                            for _ in range(max_order + 1)]
        # _cont[k][ctx_tuple][w] = continuation count
        #   k = 0..max_order-1;  ctx has length k (k=0 ↔ unigram, ctx = ())
        self._cont: list = [defaultdict(lambda: defaultdict(int))
                            for _ in range(max_order)]

    # ── internal distribution ─────────────────────────────────────────────────

    def _kn_dist(self, depth: int, ctx: tuple) -> dict:
        """Recursive Kneser-Ney distribution at (depth, context)."""
        V = len(self._vocab)
        if V == 0:
            return {}

        if depth == 0:
            # Pure continuation probability at unigram level
            cont0 = self._cont[0][()]
            total = sum(cont0.values())
            if total == 0:
                return {w: 1.0 / V for w in self._vocab}
            # Laplace floor for symbols not yet seen as successors
            floor     = 0.5
            total_adj = total + floor * V
            return {w: (cont0.get(w, 0) + floor) / total_adj
                    for w in self._vocab}

        # depth ≥ 1 — choose count table
        if depth == self.max_order:
            counts = self._raw[depth][ctx]
        else:
            counts = self._cont[depth][ctx]

        n_ctx   = sum(counts.values())
        lower   = self._kn_dist(depth - 1, ctx[1:])

        if n_ctx == 0:
            return lower                # no data at this depth; fall back

        D        = self.discount
        n_unique = len(counts)          # N+(ctx): number of distinct successors
        gamma    = D * n_unique / n_ctx # back-off mass (keeps distribution normalised)

        dist  = {}
        inv_V = 1.0 / V
        for w in self._vocab:
            cnt     = counts.get(w, 0)
            dist[w] = max(cnt - D, 0.0) / n_ctx + gamma * lower.get(w, inv_V)

        total = sum(dist.values())
        if total > 1e-15:
            return {w: dist[w] / total for w in dist}
        return lower

    # ── public interface ──────────────────────────────────────────────────────

    def _distribution(self) -> dict:
        depth = min(self.max_order, len(self.history))
        ctx   = tuple(self.history[-depth:]) if depth > 0 else ()
        return self._kn_dist(depth, ctx)

    def predict(self):
        if not self._vocab:
            return None, 0.0
        dist = self._distribution()
        if not dist:
            return None, 0.0
        best = max(dist, key=dist.get)
        return best, float(dist[best])

    def observe(self, v):
        self.history.append(v)

    def feedback(self, v):
        self._vocab.add(v)
        h = self.history          # already includes v (observe was called first)
        n = len(h)

        for k in range(1, min(self.max_order, n - 1) + 1):
            ctx    = tuple(h[-(k + 1):-1])   # length-k context
            is_new = (self._raw[k][ctx].get(v, 0) == 0)
            self._raw[k][ctx][v] += 1

            # First time this (k+1)-gram is seen → propagate to continuation counts
            if is_new:
                # shorter_ctx has length k-1; for k=1 this is () (unigram level)
                shorter_ctx = ctx[1:]
                self._cont[k - 1][shorter_ctx][v] += 1


# ══════════════════════════════════════════════════════════════════════════════
# PPM*  (PPM with high max_order)
# ══════════════════════════════════════════════════════════════════════════════

class PPMStarPredictor:
    """
    PPM*:  PPM with unbounded / high max order.

    Identical to PPMPredictor from baselines.py but with max_order=20,
    approximating the theoretical PPM* (which uses the longest matching
    context present in the history).  Memory grows as O(n·max_order) in the
    worst case, but for typical sequences the trie remains sparse.
    """

    def __init__(self, max_order: int = 20):
        from baselines import PPMPredictor
        self._impl = PPMPredictor(max_order=max_order)

    def predict(self):
        return self._impl.predict()

    def observe(self, v):
        self._impl.observe(v)

    def feedback(self, v):
        self._impl.feedback(v)

    def _distribution(self) -> dict:
        return self._impl._distribution()


# ══════════════════════════════════════════════════════════════════════════════
# ONLINE LSTM  (numpy, BPTT-1, Adam)
# ══════════════════════════════════════════════════════════════════════════════

class OnlineLSTMPredictor:
    """
    Single-layer LSTM updated one gradient step per observation (BPTT-1).

    Architecture
    ~~~~~~~~~~~~
    Input  : one-hot over seen vocabulary (pre-allocated for max_vocab symbols)
    LSTM   : standard gated cell, hidden_size = H
    Output : linear projection → softmax restricted to seen vocab symbols

    Training
    ~~~~~~~~
    Optimiser  : Adam (beta1=0.9, beta2=0.999, eps=1e-8)
    lr         : 0.003  (small for online stability)
    BPTT depth : 1  (one LSTM step back)
    Gradient   : norm-clipped to 5.0 before LSTM weight update

    Vocabulary
    ~~~~~~~~~~
    Dynamic: symbols are assigned integer indices on first encounter.
    Predictions are restricted to *seen* symbols; feedback is skipped for
    steps where the true symbol was unknown at prediction time (first
    occurrence of each symbol).

    Weight initialisation
    ~~~~~~~~~~~~~~~~~~~~~
    LSTM W : Glorot uniform
    LSTM b : zeros
    Output Wy : zeros  → initial predictions are uniform over seen vocab
    Output by : zeros
    """

    def __init__(self, hidden_size: int = 64, max_vocab: int = 128,
                 lr: float = 0.003):
        self.H, self.V, self.lr = hidden_size, max_vocab, lr
        H, V = hidden_size, max_vocab

        rng = np.random.default_rng(42)
        # Glorot uniform init for LSTM weights  (fan_in = H+V, fan_out = H)
        lim     = np.sqrt(6.0 / (H + V + H))
        self.W  = rng.uniform(-lim, lim, (4 * H, H + V)).astype(np.float64)
        self.b  = np.zeros(4 * H, dtype=np.float64)
        # Zero-init output layer: new symbols begin with uniform logit contribution
        self.Wy = np.zeros((V, H),  dtype=np.float64)
        self.by = np.zeros(V,       dtype=np.float64)

        # Adam first/second moment accumulators (full shape, lazy init)
        self._adam_t: int        = 0
        self._adam_m: list | None = None   # [mW, mb, mWy, mby]
        self._adam_v: list | None = None   # [vW, vb, vWy, vby]

        # LSTM state
        self.h = np.zeros(H, dtype=np.float64)
        self.c = np.zeros(H, dtype=np.float64)

        # Cache from the LSTM step that produced the *current* h
        # (used for BPTT-1 in feedback)
        self._prev_cache: tuple | None = None   # (combined, f, i, g, o, c_new)
        self._prev_c_in:  np.ndarray | None = None  # cell state before that step

        # Prediction context (saved in predict(), consumed in feedback())
        self._pred_h:     np.ndarray | None = None
        self._pred_probs: np.ndarray | None = None
        self._pred_n:     int               = 0
        self._bptt_cache: tuple | None      = None  # cache for step producing _pred_h
        self._bptt_c_in:  np.ndarray | None = None

        # Vocabulary bookkeeping
        self._sym_to_idx: dict = {}
        self._idx_to_sym: dict = {}
        self._n_vocab:    int  = 0

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _add_symbol(self, s) -> int | None:
        if s not in self._sym_to_idx:
            if self._n_vocab >= self.V:
                return None          # vocab overflow; ignore new symbol
            idx = self._n_vocab
            self._sym_to_idx[s] = idx
            self._idx_to_sym[idx] = s
            self._n_vocab += 1
        return self._sym_to_idx[s]

    def _lstm_step(self, x_idx: int) -> tuple:
        H, V = self.H, self.V
        x_one_hot          = np.zeros(V, dtype=np.float64)
        x_one_hot[x_idx]   = 1.0

        combined = np.concatenate([self.h, x_one_hot])  # (H+V,)
        pre      = self.W @ combined + self.b            # (4H,)

        f      = self._sigmoid(pre[:H])
        i_gate = self._sigmoid(pre[H:2 * H])
        g      = np.tanh(pre[2 * H:3 * H])
        o      = self._sigmoid(pre[3 * H:])

        c_new = f * self.c + i_gate * g
        h_new = o * np.tanh(c_new)

        return h_new, c_new, (combined, f, i_gate, g, o, c_new)

    def _init_adam(self):
        self._adam_m = [np.zeros_like(self.W),  np.zeros_like(self.b),
                        np.zeros_like(self.Wy), np.zeros_like(self.by)]
        self._adam_v = [np.zeros_like(self.W),  np.zeros_like(self.b),
                        np.zeros_like(self.Wy), np.zeros_like(self.by)]

    def _adam_step(self, dW, db, dWy_n: np.ndarray, dby_n: np.ndarray, n: int):
        """Apply Adam update.  dWy_n / dby_n cover only the first n vocab rows."""
        if self._adam_m is None:
            self._init_adam()

        b1, b2, eps = 0.9, 0.999, 1e-8
        self._adam_t += 1
        t    = self._adam_t
        bc1  = b1 ** t
        bc2  = b2 ** t

        def _step(m, v, param, grad):
            np.multiply(b1, m, out=m);       m += (1.0 - b1) * grad
            np.multiply(b2, v, out=v);       v += (1.0 - b2) * (grad * grad)
            m_hat = m / (1.0 - bc1)
            v_hat = v / (1.0 - bc2)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

        # Output-layer updates (only first n rows/elements have non-zero gradient)
        dWy_full        = np.zeros_like(self.Wy)
        dWy_full[:n]    = dWy_n
        dby_full        = np.zeros_like(self.by)
        dby_full[:n]    = dby_n

        _step(self._adam_m[2], self._adam_v[2], self.Wy, dWy_full)
        _step(self._adam_m[3], self._adam_v[3], self.by, dby_full)

        # LSTM weight updates (may be None if BPTT cache was unavailable)
        if dW is not None:
            _step(self._adam_m[0], self._adam_v[0], self.W,  dW)
            _step(self._adam_m[1], self._adam_v[1], self.b,  db)

    # ── public interface ──────────────────────────────────────────────────────

    def predict(self):
        n = self._n_vocab
        if n == 0:
            return None, 0.0

        logits = self.Wy[:n] @ self.h + self.by[:n]  # (n,)
        logits = logits - logits.max()                 # numerical stability
        exp_l  = np.exp(logits)
        probs  = exp_l / exp_l.sum()                   # (n,)

        # Snapshot current state for BPTT in upcoming feedback()
        self._pred_h     = self.h.copy()
        self._pred_probs = probs
        self._pred_n     = n
        self._bptt_cache = self._prev_cache
        self._bptt_c_in  = self._prev_c_in

        best_idx = int(np.argmax(probs))
        return self._idx_to_sym[best_idx], float(probs[best_idx])

    def observe(self, v):
        idx = self._add_symbol(v)
        if idx is None:
            return                   # vocab overflow; skip

        h_new, c_new, cache = self._lstm_step(idx)

        # Rotate BPTT cache: the step producing h_new becomes "previous" for
        # the NEXT call to predict() → feedback()
        self._prev_c_in  = self.c.copy()
        self._prev_cache = cache

        self.h = h_new
        self.c = c_new

    def feedback(self, v):
        if self._pred_h is None:
            return
        n     = self._pred_n
        v_idx = self._sym_to_idx.get(v)
        # Skip if v was unknown at prediction time (cannot assign loss to it)
        if v_idx is None or v_idx >= n:
            return

        # ── output-layer gradient ─────────────────────────────────────────────
        d_logits         = self._pred_probs.copy()     # (n,)
        d_logits[v_idx] -= 1.0                         # cross-entropy derivative

        dWy_n = np.outer(d_logits, self._pred_h)       # (n, H)
        dby_n = d_logits                               # (n,)
        dh    = self.Wy[:n].T @ d_logits               # (H,)  ← into h_{t-1}

        # ── BPTT-1 through the LSTM step that produced _pred_h ───────────────
        dW = db = None
        if self._bptt_cache is not None:
            combined, f, i_gate, g, o, c_new_bptt = self._bptt_cache
            c_in  = self._bptt_c_in
            H     = self.H

            # Gradient clipping (prevents gate saturation blow-up)
            gnorm = float(np.linalg.norm(dh))
            if gnorm > 5.0:
                dh = dh * (5.0 / gnorm)

            tanh_c  = np.tanh(c_new_bptt)
            do      = dh * tanh_c                            # dL/do (gate output)
            dc      = dh * o * (1.0 - tanh_c ** 2)          # dL/dc_new

            # c_new = f*c_in + i_gate*g
            df_gate = dc * c_in
            di_gate = dc * g
            dg_gate = dc * i_gate

            do_pre = do     * o      * (1.0 - o)             # sigmoid derivative
            df_pre = df_gate * f      * (1.0 - f)
            di_pre = di_gate * i_gate * (1.0 - i_gate)
            dg_pre = dg_gate * (1.0 - g ** 2)               # tanh derivative

            dgates = np.concatenate([df_pre, di_pre, dg_pre, do_pre])  # (4H,)

            # Clip gate gradients too (safeguard for long sequences)
            gnorm2 = float(np.linalg.norm(dgates))
            if gnorm2 > 5.0:
                dgates = dgates * (5.0 / gnorm2)

            dW = np.outer(dgates, combined)   # (4H, H+V)
            db = dgates                       # (4H,)

        # ── Adam update ───────────────────────────────────────────────────────
        self._adam_step(dW, db, dWy_n, dby_n, n)

    def _distribution(self) -> dict:
        n = self._n_vocab
        if n == 0:
            return {}
        logits = self.Wy[:n] @ self.h + self.by[:n]
        logits = logits - logits.max()
        exp_l  = np.exp(logits)
        probs  = exp_l / exp_l.sum()
        return {self._idx_to_sym[i]: float(probs[i]) for i in range(n)}
