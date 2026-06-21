"""
ieee_benchmark.py
=================
Extended evaluation suite for IEEE journal submission.

Run:   python ieee_benchmark.py
Out:   ieee_tables/   — one .tex file per table

New baseline:
  CTW(5)   — Context Tree Weighting (Willems, Shtarkov & Tjalkens, 1995).
              The provably optimal variable-order predictor under the
              normalised maximum likelihood criterion; the theoretical ceiling.

New concept-drift streams:
  Gradual    — reversal probability ramps 0→1 over 200 steps (smooth transition)
  Recurring  — A→B→A→B, 300-step cycles × 4  (tests forgetting / reacquisition)
  Fast       — reversal every 150 steps (stress-tests adaptation speed)

New metrics:
  Log-loss   — bits/symbol  = −log₂ P(true next symbol);  lower is better.
  Wilcoxon p — signed-rank test, Predictor/Forest vs PPM-D across 5 seeds.

Ablation variants (UniversalPredictor, 6 conditions):
  full         baseline (all components active)
  −correction  no correction nodes
  −exploration no exploration nodes
  −credibility all node weights fixed at 1.0 (flat voting)
  −coupling    inter-node coupling disabled (λ = 0)
  surface only distributional similarity disabled; pure surface sim
"""

import math
import os
import random
import time
from collections import defaultdict
from typing import Any, Sequence

from predictor import _CRED_MIN, _CRED_MAX, _TrieNode

# ── optional imports ───────────────────────────────────────────────────────────
try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False

# ── project imports ────────────────────────────────────────────────────────────
from predictor import UniversalPredictor
from forest import PredictorForest
from baselines import (PersistencePredictor, MajorityPredictor,
                       NgramPredictor, PPMPredictor)
try:
    from baselines_extended import KNPredictor, PPMStarPredictor, OnlineLSTMPredictor
    _EXT = True
except ImportError:
    _EXT = False
from run_experiments import discretize, normalize
from similarity import gaussian, hamming
from datasets import (load_airline_passengers, load_gutenberg_text,
                      load_dna_sequence, load_weather_events, random_integers,
                      load_electricity, load_moby_dick)


# ══════════════════════════════════════════════════════════════════════════════
# CTW BASELINE
# ══════════════════════════════════════════════════════════════════════════════

class CTWPredictor:
    """
    Context Tree Weighting  (Willems, Shtarkov & Tjalkens, 1995).

    Uses the Krichevsky-Trofimov (KT) estimator at every context-tree node.
    Blends recursively:  P_CTW(depth d) = 0.5·P_KT(d) + 0.5·P_CTW(d−1)
    The resulting predictor is asymptotically optimal in log-loss.

    KT prior per symbol: add 0.5/|V| pseudo-count (Dirichlet-symmetric).
    """

    def __init__(self, max_order: int = 5):
        self.max_order = max_order
        self.history: list = []
        # _counts[context_tuple][symbol] = integer count
        self._counts: dict = defaultdict(lambda: defaultdict(int))
        self._vocab: set = set()

    # ── public interface ──────────────────────────────────────────────────────

    def observe(self, v) -> None:
        self.history.append(v)

    def feedback(self, v) -> None:
        self._vocab.add(v)
        # Update count at every suffix depth (0 = unigram, 1 = unigram, …)
        for d in range(0, min(self.max_order, len(self.history) - 1) + 1):
            ctx = tuple(self.history[-(d + 1):-1]) if d > 0 else ()
            self._counts[ctx][v] += 1

    def predict(self):
        if not self._vocab:
            return None, 0.0
        dist = self._ctw_dist()
        if not dist:
            return None, 0.0
        best = max(dist, key=dist.get)
        return best, dist[best]

    def _distribution(self) -> dict:
        return self._ctw_dist()

    # ── internal ──────────────────────────────────────────────────────────────

    def _kt_dist(self, ctx: tuple) -> dict:
        """Krichevsky-Trofimov distribution at one context node."""
        node = self._counts[ctx]
        n_total = sum(node.values())
        V = max(len(self._vocab), 2)
        raw = {s: (node.get(s, 0) + 0.5 / V) for s in self._vocab}
        total = sum(raw.values())
        return {s: p / total for s, p in raw.items()}

    def _ctw_dist(self) -> dict:
        """Full CTW distribution for the current history."""
        depth = min(self.max_order, len(self.history))
        # Start from depth-0 (unigram KT)
        d_ctw = self._kt_dist(())
        for d in range(1, depth + 1):
            ctx = tuple(self.history[-d:])
            d_kt = self._kt_dist(ctx)
            d_ctw = {s: 0.5 * d_kt.get(s, 0.0) + 0.5 * d_ctw.get(s, 0.0)
                     for s in self._vocab}
        return d_ctw


# ══════════════════════════════════════════════════════════════════════════════
# N-GRAM WITH DISTRIBUTION OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

class NgramPredictorDist(NgramPredictor):
    """N-gram extended with _distribution() for log-loss computation."""

    def _distribution(self) -> dict:
        if not self._vocab:
            return {}
        V = len(self._vocab)
        for k in range(min(self.max_order, len(self.history)), 0, -1):
            ctx = tuple(self.history[-k:])
            d = self._counts[k][ctx]
            if not d:
                continue
            total = sum(d.values())
            return {s: (d.get(s, 0) + 1) / (total + V) for s in self._vocab}
        total = sum(self._unigram.values())
        return {s: (self._unigram.get(s, 0) + 1) / (total + V)
                for s in self._vocab}


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

class AblatedPredictor(UniversalPredictor):
    """
    UniversalPredictor with individual components disabled for ablation study.
    Uses the trie-architecture hook points (_update_node_correct,
    _update_node_wrong, _blend_lambda, _feedback_get_node).

    variant options:
        'full'           — baseline; all components active
        'no_correction'  — wrong predictions degrade wrong successor but do NOT
                           boost correct successor (correction mechanism removed)
        'no_exploration' — feedback never creates new trie nodes; only existing
                           contexts are updated (exploration mechanism removed)
        'flat_cred'      — all node_cred values reset to 1.0 after every step
                           (credibility cannot differentiate reliable contexts)
        'no_coupling'    — no-op in trie version; coupling was already removed
        'surface_only'   — blend uses flat λ=0.5 regardless of node credibility
                           (depth selection ignores track record)
    """

    VARIANTS = ('full', 'no_correction', 'no_exploration',
                'flat_cred', 'no_coupling', 'surface_only')

    def __init__(self, *args, variant: str = 'full', **kwargs):
        super().__init__(*args, **kwargs)
        self._variant = variant

    # ── credibility update hooks ──────────────────────────────────────────────

    def _update_node_correct(self, node: _TrieNode, actual) -> None:
        if self._variant == 'flat_cred':
            node.succ_cred[actual] = node.succ_cred.get(actual, 1.0) + 1.0
        else:
            super()._update_node_correct(node, actual)

    def _update_node_wrong(self, node: _TrieNode, predicted, actual) -> None:
        if self._variant == 'no_correction':
            # Degrade wrong; do NOT boost correct (removes the correction effect)
            if predicted is not None and predicted in node.succ_cred:
                node.succ_cred[predicted] = max(_CRED_MIN,
                    node.succ_cred[predicted] * (1 - self.lr))
            node.node_cred = max(_CRED_MIN, node.node_cred * (1 - self.lr))
        elif self._variant == 'flat_cred':
            if predicted is not None and predicted in node.succ_cred:
                node.succ_cred[predicted] = max(0.01, node.succ_cred[predicted] - 1.0)
            node.succ_cred[actual] = node.succ_cred.get(actual, 1.0) + 1.0
        else:
            super()._update_node_wrong(node, predicted, actual)

    # ── blend hook ────────────────────────────────────────────────────────────

    def _blend_lambda(self, node_cred: float) -> float:
        if self._variant == 'surface_only':
            return 0.5   # flat blend; credibility plays no role in depth weighting
        return super()._blend_lambda(node_cred)

    # ── node creation hook ────────────────────────────────────────────────────

    def _feedback_get_node(self, ctx: tuple):
        if self._variant == 'no_exploration':
            return self._walk(ctx)   # only update existing nodes, never create
        return super()._feedback_get_node(ctx)

    # ── post-feedback reset ───────────────────────────────────────────────────

    def feedback(self, actual) -> None:
        super().feedback(actual)
        if self._variant == 'flat_cred':
            stack = [self._root]
            while stack:
                n = stack.pop()
                n.node_cred = 1.0
                stack.extend(n.children.values())


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC CONCEPT-DRIFT DATASETS
# ══════════════════════════════════════════════════════════════════════════════

def _cycle_step(v, n_symbols, forward):
    return (v + (1 if forward else -1)) % n_symbols


def gradual_drift_seq(n=1200, drift_center=600, drift_width=200,
                      n_symbols=3, noise=0.05, seed=42):
    """
    Forward cycle for the first ~drift_center steps, then backward cycle.
    The transition is smooth: P(new rule) ramps from 0→1 over drift_width steps.
    """
    rng = random.Random(seed)
    vocab = list(range(n_symbols))
    seq = [rng.choice(vocab)]
    for i in range(1, n):
        v = seq[-1]
        lo = drift_center - drift_width // 2
        hi = drift_center + drift_width // 2
        p_new = max(0.0, min(1.0, (i - lo) / max(drift_width, 1)))
        forward = rng.random() > p_new
        intended = _cycle_step(v, n_symbols, forward)
        seq.append(rng.choice(vocab) if rng.random() < noise else intended)
    return seq


def recurring_concepts_seq(n=1200, cycle_length=300,
                            n_symbols=3, noise=0.05, seed=42):
    """
    Alternates between forward and backward cycle every cycle_length steps.
    Tests whether the system can reacquire a forgotten concept.
    """
    rng = random.Random(seed)
    vocab = list(range(n_symbols))
    seq = [rng.choice(vocab)]
    for i in range(1, n):
        v = seq[-1]
        forward = (i // cycle_length) % 2 == 0
        intended = _cycle_step(v, n_symbols, forward)
        seq.append(rng.choice(vocab) if rng.random() < noise else intended)
    return seq


def fast_drift_seq(n=1200, drift_interval=150,
                   n_symbols=3, noise=0.05, seed=42):
    """
    Rule reverses every drift_interval steps (many rapid concept changes).
    """
    rng = random.Random(seed)
    vocab = list(range(n_symbols))
    seq = [rng.choice(vocab)]
    for i in range(1, n):
        v = seq[-1]
        forward = (i // drift_interval) % 2 == 0
        intended = _cycle_step(v, n_symbols, forward)
        seq.append(rng.choice(vocab) if rng.random() < noise else intended)
    return seq


# ══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTION EXTRACTION + LOG-LOSS
# ══════════════════════════════════════════════════════════════════════════════

def _get_dist(predictor, vocab: set) -> dict:
    """
    Return the full predictive distribution after a predict() call.

    All predictors now expose _distribution().  For CTW, N-gram, and PPM-D
    this returns the model's native distribution.  For UniversalPredictor /
    AblatedPredictor the trie _blend() populates _last_distribution which is
    returned by _distribution() — already a proper probability distribution
    over the full vocabulary with KT smoothing, so no extra Laplace needed.
    """
    if hasattr(predictor, '_distribution'):
        d = predictor._distribution()
        if d:
            # Ensure every vocab symbol is covered (handles CTW/N-gram/PPM
            # which may not have seen all symbols yet)
            missing = vocab - set(d)
            if missing:
                floor = 1e-6
                total = sum(d.values()) + floor * len(missing)
                out   = {s: d.get(s, floor) / total for s in vocab}
                return out
            total = sum(d.get(s, 0) for s in vocab)
            return {s: d[s] / total for s in vocab} if total > 0 else d

    # Fallback: uniform (should not be reached with current predictors)
    V = max(len(vocab), 1)
    return {s: 1.0 / V for s in vocab}


def log_loss_bits(dist: dict, actual) -> float:
    """Bits/symbol:  −log₂ P(actual).  Lower is better."""
    p = dist.get(actual, 1e-10)
    return -math.log2(max(p, 1e-10))


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def wilcoxon_p(x: list, y: list) -> float:
    """
    Two-tailed Wilcoxon signed-rank p-value comparing paired samples x and y.
    Uses scipy if available, otherwise falls back to a normal approximation.
    """
    diffs = [a - b for a, b in zip(x, y)]
    nonzero = [d for d in diffs if d != 0]
    if not nonzero:
        return 1.0
    if _SCIPY and len(nonzero) >= 10:
        try:
            _, p = _scipy_wilcoxon(nonzero, alternative='two-sided')
            return float(p)
        except Exception:
            pass
    # Normal approximation (valid for n >= 10)
    n = len(nonzero)
    ranked = sorted(enumerate(abs(d) for d in nonzero), key=lambda x: x[1])
    T_plus = sum(r for r, (orig, _) in enumerate(ranked, 1)
                 if nonzero[orig] > 0)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (T_plus - mu) / sigma if sigma > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return float(p)


def mean_ci_95(values: list) -> tuple:
    """Return (mean, half-width of 95% CI) assuming normal distribution."""
    n = len(values)
    if n < 2:
        return (values[0] if values else 0.0, 0.0)
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return mu, 1.96 * se


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_full(method, seq: list, train_frac: float = 0.8,
                   drift_at: int | None = None,
                   post_drift_window: int = 120) -> dict:
    """
    Fully online evaluation: predict → observe → feedback at every step.
    Returns accuracy, log-loss, and drift-recovery statistics.
    """
    n = len(seq)
    train_n = int(n * train_frac)
    vocab = set(seq)

    records = []
    for i, v in enumerate(seq):
        pred, conf = method.predict()
        dist = _get_dist(method, vocab)
        method.observe(v)
        method.feedback(v)
        if pred is not None:
            records.append({
                'step':    i,
                'correct': pred == v,
                'll':      log_loss_bits(dist, v),
                'phase':   'train' if i < train_n else 'test',
            })

    test_r = [r for r in records if r['phase'] == 'test']
    train_r = [r for r in records if r['phase'] == 'train']

    def _acc(recs):
        return sum(r['correct'] for r in recs) / len(recs) if recs else 0.0

    def _ll(recs):
        return sum(r['ll'] for r in recs) / len(recs) if recs else float('inf')

    # Q1 ramp: first quarter of test
    q1_r = test_r[:max(1, len(test_r) // 4)]

    # Post-drift window (train-side only)
    post_drift = None
    if drift_at is not None:
        dr = [r for r in records
              if drift_at <= r['step'] < drift_at + post_drift_window
              and r['step'] < train_n]
        post_drift = _acc(dr)

    # 4-block learning curve over the full sequence
    block_accs = []
    for b in range(4):
        lo = round(n * b / 4)
        hi = round(n * (b + 1) / 4)
        br = [r for r in records if lo <= r['step'] < hi]
        block_accs.append(_acc(br))

    return {
        'test_acc':          _acc(test_r),
        'test_ll':           _ll(test_r),
        'train_acc':         _acc(train_r),
        'q1_acc':            _acc(q1_r),
        'post_drift_acc':    post_drift,
        'block_accs':        block_accs,
        'n_test':            len(test_r),
        # Per-step binary correct list — used for per-step Wilcoxon significance test.
        'step_correct_test': [int(r['correct']) for r in test_r],
    }


def _time_and_nodes(method, seq: list) -> dict:
    """Run method and return wall-clock time + final node count."""
    t0 = time.perf_counter()
    for v in seq:
        method.predict()
        method.observe(v)
        method.feedback(v)
    elapsed = time.perf_counter() - t0
    nodes = len(getattr(method, '_nodes', []))
    return {'time_s': elapsed, 'n_nodes': nodes}


# ══════════════════════════════════════════════════════════════════════════════
# METHOD FACTORIES
# ══════════════════════════════════════════════════════════════════════════════

_FOREST_BASE = dict(n_trees=5, dropout=0.2, stagger=25, voting='adaptive',
                    heterogeneous_k=True, tree_lr=0.1, max_trees=20,
                    auto_grow=True, auto_prune=True,
                    grow_threshold=8, prune_window=50,
                    binary_correction_scale=0.05)


def _make_std_methods(k, sim, vig, forest_kw=None):
    fkw = dict(_FOREST_BASE)
    if forest_kw:
        fkw.update(forest_kw)
    return [
        ('Persistence', PersistencePredictor()),
        ('Majority',    MajorityPredictor()),
        ('N-gram(5)',   NgramPredictorDist(max_order=5)),
        ('PPM-D(5)',    PPMPredictor(max_order=5)),
        ('CTW(5)',      CTWPredictor(max_order=5)),
        ('Predictor',   UniversalPredictor(k, sim, learning_rate=0.08, vigilance=vig,
                                            adaptive_cap=True, binary_correction_scale=0.05,
                                            cred_max=6.05, lambda_power=0.65)),
        ('Forest',      PredictorForest(k, sim, learning_rate=0.1, vigilance=vig, **fkw)),
    ]


def _make_extended_methods(k, sim, vig, forest_kw=None):
    """Extended baselines: KN, PPM*, Online LSTM alongside Predictor and Forest."""
    if not _EXT:
        raise ImportError('baselines_extended.py not found')
    fkw = dict(_FOREST_BASE)
    if forest_kw:
        fkw.update(forest_kw)
    return [
        ('KN(5)',     KNPredictor(max_order=5)),
        ('PPM*(20)',  PPMStarPredictor(max_order=20)),
        ('LSTM(64)',  OnlineLSTMPredictor(hidden_size=64)),
        ('Predictor', UniversalPredictor(k, sim, learning_rate=0.08, vigilance=vig,
                                         adaptive_cap=True, binary_correction_scale=0.05,
                                         cred_max=6.05, lambda_power=0.65)),
        ('Forest',    PredictorForest(k, sim, learning_rate=0.1, vigilance=vig, **fkw)),
    ]


def _make_drift_methods(k, sim, vig):
    """For concept drift: cap N-gram/PPM/CTW to same max_order=k."""
    fkw = dict(_FOREST_BASE,
               heterogeneous_k=False, auto_grow=False, auto_prune=False)
    return [
        ('Persistence',        PersistencePredictor()),
        ('Majority',           MajorityPredictor()),
        (f'N-gram({k})',       NgramPredictorDist(max_order=k)),
        (f'PPM-D({k})',        PPMPredictor(max_order=k)),
        (f'CTW({k})',          CTWPredictor(max_order=k)),
        ('Predictor',          UniversalPredictor(k, sim, learning_rate=0.08, vigilance=vig,
                                                   adaptive_cap=True, binary_correction_scale=0.05,
                                                   cred_max=6.05, lambda_power=0.65)),
        ('Forest',             PredictorForest(k, sim, learning_rate=0.1, vigilance=vig, **fkw)),
    ]


def _make_ablation_methods(k, sim, vig):
    def _mk(variant):
        return AblatedPredictor(k, sim, learning_rate=0.1, vigilance=vig,
                                variant=variant)
    return [
        ('Full',          _mk('full')),
        ('−correction',   _mk('no_correction')),
        ('−exploration',  _mk('no_exploration')),
        ('−credibility',  _mk('flat_cred')),
        ('−coupling',     _mk('no_coupling')),
        ('Surface only',  _mk('surface_only')),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# LATEX HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(v, digits=1, pct=True):
    """Format a float as a percentage string."""
    s = f'{v * 100:.{digits}f}'
    return s + ('\\%' if pct else '')


def _cell(v, best, second, digits=1, higher_better=True):
    """Format a table cell with bold best / underline second best."""
    f = _fmt(v, digits)
    if higher_better:
        if abs(v - best) < 1e-9:
            return f'\\textbf{{{f}}}'
        if abs(v - second) < 1e-9:
            return f'\\underline{{{f}}}'
    else:  # lower is better (log-loss)
        if abs(v - best) < 1e-9:
            return f'\\textbf{{{f}}}'
        if abs(v - second) < 1e-9:
            return f'\\underline{{{f}}}'
    return f


def _sig(p):
    """Significance stars."""
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


_TEX_HEADER = r"""\usepackage{booktabs}
\usepackage{multirow}
\usepackage{siunitx}
"""


def _write_table(path: str, tex: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(tex)
    print(f'  Wrote {path}')


# ══════════════════════════════════════════════════════════════════════════════
# TABLE GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def _table_accuracy(all_ds, caption, label):
    """Table: test accuracy across datasets × methods."""
    methods = list(all_ds[0]['results'].keys())
    col_spec = 'l' + 'r' * len(methods)
    header = ' & '.join(['Dataset'] + [m.replace('(', '(').replace('−', '$-$')
                                        for m in methods])
    rows = []
    for ds in all_ds:
        name = ds['name']
        vals = {m: ds['results'][m]['test_acc'] for m in methods}
        sorted_v = sorted(vals.values(), reverse=True)
        best, second = sorted_v[0], (sorted_v[1] if len(sorted_v) > 1 else sorted_v[0])
        row = name + ' & ' + ' & '.join(
            _cell(vals[m], best, second) for m in methods)
        rows.append(row)

    body = ' \\\\\n'.join(rows)
    return (
        f'\\begin{{table}}[h]\n'
        f'\\centering\n'
        f'\\caption{{{caption}}}\n'
        f'\\label{{{label}}}\n'
        f'\\begin{{tabular}}{{{col_spec}}}\n'
        f'\\toprule\n'
        f'{header} \\\\\n'
        f'\\midrule\n'
        f'{body} \\\\\n'
        f'\\bottomrule\n'
        f'\\end{{tabular}}\n'
        f'\\end{{table}}\n'
    )


def _table_logloss(all_ds, caption, label):
    """Table: log-loss (bits/symbol) — lower is better."""
    methods = list(all_ds[0]['results'].keys())
    col_spec = 'l' + 'r' * len(methods)
    header = ' & '.join(['Dataset'] + [m for m in methods])
    rows = []
    for ds in all_ds:
        name = ds['name']
        vals = {m: ds['results'][m]['test_ll'] for m in methods}
        valid = [v for v in vals.values() if math.isfinite(v)]
        best   = min(valid) if valid else 0
        second = sorted(valid)[1] if len(valid) > 1 else best
        row = name + ' & ' + ' & '.join(
            _cell(vals[m], best, second, digits=2, higher_better=False)
            if math.isfinite(vals[m]) else 'n/a'
            for m in methods)
        rows.append(row)

    body = ' \\\\\n'.join(rows)
    return (
        f'\\begin{{table}}[h]\n'
        f'\\centering\n'
        f'\\caption{{{caption}}}\n'
        f'\\label{{{label}}}\n'
        f'\\begin{{tabular}}{{{col_spec}}}\n'
        f'\\toprule\n'
        f'{header} \\\\\n'
        f'\\midrule\n'
        f'{body} \\\\\n'
        f'\\bottomrule\n'
        f'\\end{{tabular}}\n'
        f'\\end{{table}}\n'
    )


def _table_ablation(ablation_results, caption, label):
    """Table: ablation study — each component removed."""
    datasets = list(ablation_results.keys())
    variants = list(ablation_results[datasets[0]].keys())
    col_spec = 'l' + 'r' * len(datasets)
    header = ' & '.join(['Variant'] + datasets)
    rows = []
    for v in variants:
        vals = {d: ablation_results[d][v]['test_acc'] for d in datasets}
        sorted_v = sorted(vals.values(), reverse=True)
        best, second = sorted_v[0], (sorted_v[1] if len(sorted_v) > 1 else sorted_v[0])
        row = v.replace('−', '$-$') + ' & ' + ' & '.join(
            _fmt(vals[d]) for d in datasets)
        rows.append(row)
    body = ' \\\\\n'.join(rows)
    return (
        f'\\begin{{table}}[h]\n'
        f'\\centering\n'
        f'\\caption{{{caption}}}\n'
        f'\\label{{{label}}}\n'
        f'\\begin{{tabular}}{{{col_spec}}}\n'
        f'\\toprule\n'
        f'{header} \\\\\n'
        f'\\midrule\n'
        f'{body} \\\\\n'
        f'\\bottomrule\n'
        f'\\end{{tabular}}\n'
        f'\\end{{table}}\n'
    )


def _table_significance(sig_results, caption, label):
    """Table: per-step Wilcoxon p-values (Predictor and Forest vs PPM-D)."""
    datasets = list(sig_results.keys())
    col_spec = 'l r rr rr'
    header = (r'Dataset & $n$ & \multicolumn{2}{c}{Predictor vs PPM-D} & '
              r'\multicolumn{2}{c}{Forest vs PPM-D} \\')
    sub_hdr = r' & & Acc. $\Delta$ & $p$ & Acc. $\Delta$ & $p$ \\'
    rows = []
    for ds in datasets:
        pred_d  = sig_results[ds]['pred_delta']
        pred_p  = sig_results[ds]['pred_p']
        fst_d   = sig_results[ds]['forest_delta']
        fst_p   = sig_results[ds]['forest_p']
        n_test  = sig_results[ds].get('n_test', '?')
        star_p  = _sig(pred_p)
        star_f  = _sig(fst_p)
        row = (f'{ds} & {n_test} & {pred_d:+.1f}pp$^{{{star_p}}}$ & {pred_p:.3f} & '
               f'{fst_d:+.1f}pp$^{{{star_f}}}$ & {fst_p:.3f}')
        rows.append(row)
    body = ' \\\\\n'.join(rows)
    return (
        f'\\begin{{table}}[h]\n'
        f'\\centering\n'
        f'\\caption{{{caption}}}\n'
        f'\\label{{{label}}}\n'
        f'\\begin{{tabular}}{{{col_spec}}}\n'
        f'\\toprule\n'
        f'{header}\n'
        f'\\midrule\n'
        f'{sub_hdr}\n'
        f'\\midrule\n'
        f'{body} \\\\\n'
        f'\\bottomrule\n'
        f'\\end{{tabular}}\n'
        f'\\end{{table}}\n'
    )


def _table_scaling(scaling_results, caption, label):
    """Table: accuracy and timing vs sequence length."""
    ns = sorted(scaling_results.keys())
    methods = list(scaling_results[ns[0]].keys())
    col_spec = 'r' + 'rr' * len(methods)
    m_header = ' & '.join(f'\\multicolumn{{2}}{{c}}{{{m}}}' for m in methods)
    sub_header = ' & '.join(['n'] + ['Acc.\\% & ms'] * len(methods))
    rows = []
    for n in ns:
        parts = [str(n)]
        for m in methods:
            acc = scaling_results[n][m].get('test_acc', 0)
            t_ms = scaling_results[n][m].get('time_s', 0) * 1000
            nodes = scaling_results[n][m].get('n_nodes', 0)
            parts.append(f'{acc*100:.1f} & {t_ms:.0f}')
        rows.append(' & '.join(parts))
    body = ' \\\\\n'.join(rows)
    return (
        f'\\begin{{table}}[h]\n'
        f'\\centering\n'
        f'\\caption{{{caption}}}\n'
        f'\\label{{{label}}}\n'
        f'\\begin{{tabular}}{{{col_spec}}}\n'
        f'\\toprule\n'
        f' & {m_header} \\\\\n'
        f'{sub_header} \\\\\n'
        f'\\midrule\n'
        f'{body} \\\\\n'
        f'\\bottomrule\n'
        f'\\end{{tabular}}\n'
        f'\\end{{table}}\n'
    )


# ══════════════════════════════════════════════════════════════════════════════
# DATASET LOADER
# ══════════════════════════════════════════════════════════════════════════════

def _load_standard_datasets():
    datasets = []

    raw = load_airline_passengers()
    datasets.append({
        'name': 'Airline', 'seq': discretize(normalize(raw), n_bins=8),
        'sim': gaussian(sigma=2.0), 'k': 4, 'vig': 0.3, 'forest_kw': {},
    })
    for loader, name, k, vig, fkw in [
        (lambda: load_gutenberg_text(n_chars=15_000),
         'Alice (15K)', 6, 0.7, {'n_trees': 3}),
        (lambda: load_moby_dick(n_chars=50_000),
         'Moby Dick (50K)', 6, 0.7, {'n_trees': 3}),
        (lambda: load_dna_sequence(n_bases=None),
         'DNA (full 48K)', 5, 0.3, {'n_trees': 2, 'heterogeneous_k': False}),
        (lambda: load_weather_events(n_days=None),
         'Weather', 3, 0.3, {}),
    ]:
        try:
            datasets.append({'name': name, 'seq': loader(),
                             'sim': hamming, 'k': k, 'vig': vig, 'forest_kw': fkw})
        except Exception as e:
            print(f'  [skip] {name}: {e}')

    datasets.append({
        'name': 'PRNG', 'seq': random_integers(n=500, low=0, high=9, seed=7),
        'sim': gaussian(sigma=1.0), 'k': 3, 'vig': 0.3, 'forest_kw': {},
    })

    try:
        elec = load_electricity()
        datasets.append({
            'name': 'Electricity', 'seq': elec,
            'sim': hamming, 'k': 4, 'vig': 0.7, 'forest_kw': {},
        })
    except Exception as e:
        print(f'  [skip] Electricity: {e}')

    return datasets


# ══════════════════════════════════════════════════════════════════════════════
# LEARNING-CURVE PLOT  (matplotlib, optional)
# ══════════════════════════════════════════════════════════════════════════════

def _plot_drift_curves(drift_datasets, out_dir='ieee_tables'):
    if not _MPL:
        return
    os.makedirs(out_dir, exist_ok=True)
    for ds in drift_datasets:
        name  = ds['name']
        seq   = ds['seq']
        cfg   = ds['cfg']
        drift_at = ds.get('drift_at')
        n_blocks = 20
        block_n  = len(seq) // n_blocks

        fig, ax = plt.subplots(figsize=(7, 3.5))
        methods = _make_drift_methods(cfg['k'], cfg['sim'], cfg['vig'])

        for mname, method in methods:
            block_accs = []
            correct = total = 0
            for i, v in enumerate(seq):
                pred, _ = method.predict()
                method.observe(v)
                method.feedback(v)
                if pred is not None:
                    correct += int(pred == v)
                    total += 1
                if (i + 1) % block_n == 0:
                    block_accs.append(correct / max(total, 1))
                    correct = total = 0

            xs = [(b + 0.5) * block_n for b in range(len(block_accs))]
            style = {'Predictor': dict(lw=2, color='tab:blue'),
                     'Forest':    dict(lw=2, color='tab:orange'),
                     }.get(mname, dict(lw=1, ls='--', alpha=0.6))
            ax.plot(xs, [a * 100 for a in block_accs],
                    label=mname, **style)

        if drift_at:
            ax.axvline(drift_at, color='red', ls=':', lw=1.5, label='Drift point')
        ax.axhline(100 / len(set(seq)), color='gray', ls=':', lw=1, alpha=0.5,
                   label='Baseline (random)')
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy (\\%)')
        ax.set_title(name)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f'fig_drift_{name.lower().replace(" ", "_")}.pdf')
        fig.savefig(path)
        plt.close(fig)
        print(f'  Wrote {path}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUT = 'ieee_tables'
    os.makedirs(OUT, exist_ok=True)

    # ── 1. STANDARD DATASETS ──────────────────────────────────────────────────
    print('\n[1/5] Standard datasets …')
    std_cfgs     = _load_standard_datasets()
    all_std_summary = []   # [{name, results: {method: {test_acc, test_ll, step_correct_test}}}]

    for cfg in std_cfgs:
        name = cfg['name']
        seq  = cfg['seq']
        print(f'  {name}  (n={len(seq)}, k={cfg["k"]}) …', end=' ', flush=True)
        methods = _make_std_methods(cfg['k'], cfg['sim'], cfg['vig'],
                                    cfg.get('forest_kw'))
        entry = {'name': name, 'seq': seq, 'results': {}}
        for mname, method in methods:
            r = _evaluate_full(method, seq)
            entry['results'][mname] = r
        all_std_summary.append(entry)
        print('done')

    # Write Table 1: accuracy
    tex1 = _table_accuracy(
        all_std_summary,
        caption=('Test accuracy (\\%) on standard benchmarks. '
                 '\\textbf{Bold} = best, \\underline{underline} = second best.'),
        label='tab:main_accuracy')
    _write_table(f'{OUT}/01_main_accuracy.tex', tex1)

    # Write Table 2: log-loss
    tex2 = _table_logloss(
        all_std_summary,
        caption=('Log-loss (bits/symbol) on standard benchmarks '
                 '(Laplace-smoothed, $\\alpha{=}0.1$). '
                 'Lower is better; \\textbf{bold} = best.'),
        label='tab:logloss')
    _write_table(f'{OUT}/02_logloss.tex', tex2)

    # ── 2. STATISTICAL SIGNIFICANCE (per-step Wilcoxon) ───────────────────────
    # Each test step is a paired binary observation (correct=1, wrong=0).
    # With n = |test partition| ≈ 100–300 steps per dataset, the test has
    # proper power.  This is the correct unit of analysis: every symbol
    # prediction is an independent trial given the history up to that point.
    print('\n[2/5] Statistical significance (per-step Wilcoxon signed-rank) …')
    sig_results = {}
    for ds in all_std_summary:
        name      = ds['name']
        ppm_steps  = ds['results']['PPM-D(5)']['step_correct_test']
        pred_steps = ds['results']['Predictor']['step_correct_test']
        fst_steps  = ds['results']['Forest']['step_correct_test']
        n = min(len(ppm_steps), len(pred_steps), len(fst_steps))
        pred_delta = (sum(pred_steps[:n]) - sum(ppm_steps[:n])) / max(n, 1) * 100
        fst_delta  = (sum(fst_steps[:n])  - sum(ppm_steps[:n])) / max(n, 1) * 100
        sig_results[name] = {
            'pred_delta':   pred_delta,
            'pred_p':       wilcoxon_p(pred_steps[:n], ppm_steps[:n]),
            'forest_delta': fst_delta,
            'forest_p':     wilcoxon_p(fst_steps[:n],  ppm_steps[:n]),
            'n_test':       n,
        }
    tex5 = _table_significance(
        sig_results,
        caption=('Wilcoxon signed-rank test (two-tailed): Predictor and Forest vs '
                 'PPM-D(5). Each paired observation is one test-step prediction '
                 '(correct$=$1, wrong$=$0); $n$ is the number of test steps. '
                 '$^*p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$.'),
        label='tab:significance')
    _write_table(f'{OUT}/05_significance.tex', tex5)

    # ── 2. CONCEPT-DRIFT DATASETS ─────────────────────────────────────────────
    print('\n[3/5] Concept-drift datasets …')
    drift_seqs = [
        {'name': 'Sudden (k=1)',
         'seq':  _sudden_drift_seq(),
         'cfg':  {'k': 1, 'sim': hamming, 'vig': 0.7},
         'drift_at': 500},
        {'name': 'Gradual (k=1)',
         'seq':  gradual_drift_seq(n=1200, drift_center=600, drift_width=200, seed=42),
         'cfg':  {'k': 1, 'sim': hamming, 'vig': 0.7},
         'drift_at': 600},
        {'name': 'Recurring (k=1)',
         'seq':  recurring_concepts_seq(n=1200, cycle_length=300, seed=42),
         'cfg':  {'k': 1, 'sim': hamming, 'vig': 0.7},
         'drift_at': 300},
        {'name': 'Fast (k=1)',
         'seq':  fast_drift_seq(n=1200, drift_interval=150, seed=42),
         'cfg':  {'k': 1, 'sim': hamming, 'vig': 0.7},
         'drift_at': 150},
    ]

    drift_summary = []
    for ds in drift_seqs:
        name = ds['name']
        seq  = ds['seq']
        cfg  = ds['cfg']
        drift_at = ds.get('drift_at')
        print(f'  {name} …', end=' ', flush=True)
        methods = _make_drift_methods(cfg['k'], cfg['sim'], cfg['vig'])
        entry = {'name': name, 'seq': seq, 'results': {}}
        for mname, method in methods:
            r = _evaluate_full(method, seq, drift_at=drift_at)
            entry['results'][mname] = r
        drift_summary.append(entry)
        print('done')

    tex3 = _table_accuracy(
        drift_summary,
        caption=('Test accuracy (\\%) on concept-drift streams. '
                 'All methods capped to order $k=1$ so the count-based '
                 'methods cannot escape the drift via higher-order backoff. '
                 'Sudden: reversal at step 500; '
                 'Gradual: 200-step ramp; '
                 'Recurring: $300$-step A$\\to$B$\\to$A cycles; '
                 'Fast: reversal every 150 steps.'),
        label='tab:drift_accuracy')
    _write_table(f'{OUT}/03_drift_accuracy.tex', tex3)

    # Post-drift recovery supplementary table
    _write_postdrift_table(drift_summary, OUT)

    # Optional: learning-curve plots
    _plot_drift_curves(drift_seqs, out_dir=OUT)

    # ── 3. ABLATION STUDY ─────────────────────────────────────────────────────
    print('\n[4/5] Ablation study …')
    ablation_datasets = [
        cfg for cfg in std_cfgs if cfg['name'] in ('Airline', 'Alice', 'DNA', 'Weather')
    ]
    # Also add the sudden-drift test
    ablation_datasets.append({
        'name': 'Drift', 'seq': _sudden_drift_seq(),
        'sim': hamming, 'k': 1, 'vig': 0.7,
        'forest_kw': {},
    })

    ablation_results = {}  # dataset_name → variant → result
    for cfg in ablation_datasets:
        name = cfg['name']
        seq  = cfg['seq']
        print(f'  {name} …', end=' ', flush=True)
        variants_res = {}
        methods = _make_ablation_methods(cfg['k'], cfg['sim'], cfg['vig'])
        for mname, method in methods:
            r = _evaluate_full(method, seq)
            variants_res[mname] = r
        ablation_results[name] = variants_res
        print('done')

    tex4 = _table_ablation(
        ablation_results,
        caption=('Ablation study: test accuracy (\\%) when each component '
                 'of the UniversalPredictor is disabled. '
                 'Full = complete system; '
                 '$-$correction = no correction nodes; '
                 '$-$exploration = no exploration nodes; '
                 '$-$credibility = flat node weights; '
                 '$-$coupling = inter-node communication disabled; '
                 'Surface only = pure surface similarity, no distributional blend.'),
        label='tab:ablation')
    _write_table(f'{OUT}/04_ablation.tex', tex4)

    # ── 4. SCALING ANALYSIS ───────────────────────────────────────────────────
    print('\n[5/5] Computational scaling analysis …')
    scale_ns = [250, 500, 1000, 2500, 5000]
    scale_methods = {
        'N-gram(5)':  lambda: NgramPredictorDist(max_order=5),
        'PPM-D(5)':   lambda: PPMPredictor(max_order=5),
        'CTW(5)':     lambda: CTWPredictor(max_order=5),
        'Predictor':  lambda: UniversalPredictor(3, hamming,
                                                  learning_rate=0.1, vigilance=0.7),
    }
    scaling_results = {}
    for n in scale_ns:
        seq = random_integers(n=n, low=0, high=9, seed=77)
        scaling_results[n] = {}
        for mname, factory in scale_methods.items():
            method = factory()
            timing = _time_and_nodes(method, seq)
            # also run proper evaluation for accuracy
            method2 = factory()
            r = _evaluate_full(method2, seq)
            scaling_results[n][mname] = {
                'test_acc': r['test_acc'],
                'time_s':   timing['time_s'],
                'n_nodes':  timing['n_nodes'],
            }
        print(f'  n={n} done')

    tex6 = _table_scaling(
        scaling_results,
        caption=('Accuracy (\\%) and wall-clock time (ms) vs sequence length $n$. '
                 'Predictor creates $O(n)$ nodes; '
                 'reported as node count at end of sequence.'),
        label='tab:scaling')
    _write_table(f'{OUT}/06_scaling.tex', tex6)

    # ── 5. ELECTRICITY: REAL-WORLD CONCEPT DRIFT  ─────────────────────────────
    print('\n[5b/5] Electricity real-world concept-drift benchmark …')
    try:
        elec_seq = load_electricity()
        _run_electricity_analysis(elec_seq, OUT)
    except Exception as e:
        print(f'  [skip] Electricity: {e}')

    # ── 6. EXTENDED BASELINE COMPARISON (KN, PPM*, Online LSTM) ─────────────
    if _EXT:
        print('\n[6/6] Extended baseline comparison (KN, PPM*, LSTM) …')
        ext_summary = []
        for cfg in std_cfgs:
            name = cfg['name']
            seq  = cfg['seq']
            print(f'  {name}  (n={len(seq)}) …', end=' ', flush=True)
            methods = _make_extended_methods(cfg['k'], cfg['sim'], cfg['vig'],
                                             cfg.get('forest_kw'))
            entry = {'name': name, 'seq': seq, 'results': {}}
            for mname, method in methods:
                r = _evaluate_full(method, seq)
                entry['results'][mname] = r
            ext_summary.append(entry)
            print('done')

        tex8 = _table_accuracy(
            ext_summary,
            caption=(
                'Extended baseline comparison: test accuracy (\\%) for '
                'Interpolated Kneser-Ney N-gram KN(5), '
                'PPM* with max order 20, and Online LSTM ($H{=}64$, '
                'BPTT-1, Adam) versus Predictor and Forest. '
                '\\textbf{Bold} = best, \\underline{underline} = second best.'
            ),
            label='tab:extended')
        _write_table(f'{OUT}/08_extended_comparison.tex', tex8)
    else:
        print('\n[6/6] Extended baselines skipped (baselines_extended.py not found)')

    # ── 7. MASTER INCLUDE FILE ────────────────────────────────────────────────
    _write_master_tex(OUT)
    print(f'\nAll tables written to {OUT}/')
    print('In LaTeX: \\input{ieee_tables/all_tables.tex}')


# ── helpers used in main ───────────────────────────────────────────────────────

def _sudden_drift_seq(n=1000, drift_at=500, noise=0.05, seed=99):
    """Identical to benchmark.py concept drift: k=1, reversed rule."""
    rng = random.Random(seed)
    vocab = [0, 1, 2]
    seq = [rng.choice(vocab)]
    for i in range(1, n):
        v = seq[-1]
        intended = (v + 1) % 3 if i < drift_at else (v - 1) % 3
        seq.append(rng.choice(vocab) if rng.random() < noise else intended)
    return seq


def _write_postdrift_table(drift_summary, out_dir):
    """Supplementary: post-drift recovery (first 120 steps after drift)."""
    methods = list(drift_summary[0]['results'].keys())
    col_spec = 'l' + 'r' * len(methods)
    header = ' & '.join(['Stream'] + methods)
    rows = []
    for ds in drift_summary:
        name = ds['name']
        vals = {m: ds['results'][m].get('post_drift_acc') or 0.0
                for m in methods}
        sorted_v = sorted(vals.values(), reverse=True)
        best   = sorted_v[0]
        second = sorted_v[1] if len(sorted_v) > 1 else best
        row = name + ' & ' + ' & '.join(
            _cell(vals[m], best, second) for m in methods)
        rows.append(row)
    body = ' \\\\\n'.join(rows)
    tex = (
        f'\\begin{{table}}[h]\n'
        f'\\centering\n'
        f'\\caption{{Post-drift recovery: accuracy in first 120 steps '
        f'after each concept change (within training partition). '
        f'Higher = faster adaptation.}}\n'
        f'\\label{{tab:postdrift}}\n'
        f'\\begin{{tabular}}{{{col_spec}}}\n'
        f'\\toprule\n'
        f'{header} \\\\\n'
        f'\\midrule\n'
        f'{body} \\\\\n'
        f'\\bottomrule\n'
        f'\\end{{tabular}}\n'
        f'\\end{{table}}\n'
    )
    _write_table(f'{out_dir}/03b_postdrift_recovery.tex', tex)


def _run_electricity_analysis(elec_seq: list, out_dir: str,
                               block_size: int = 1000) -> None:
    """
    Run all 7 methods on the full Electricity sequence (45K steps).
    Produce:
      07_electricity_accuracy.tex   — overall test accuracy table
      07b_electricity_blocks.tex    — per-1000-step rolling accuracy table
      fig_electricity_blocks.pdf    — learning curve (if matplotlib available)
    """
    vocab = set(elec_seq)
    n     = len(elec_seq)
    n_blocks = n // block_size
    k, vig = 4, 0.7

    methods = _make_drift_methods(k, hamming, vig)
    method_names = [m for m, _ in methods]

    # Prequential evaluation with block tracking
    block_accs: dict[str, list[float]] = {m: [] for m in method_names}
    overall:    dict[str, dict]        = {}

    for mname, method in methods:
        print(f'    {mname} …', end=' ', flush=True)
        block_correct = block_total = 0
        all_correct   = all_total   = 0
        test_correct  = test_total  = 0
        test_start    = int(n * 0.8)

        for i, v in enumerate(elec_seq):
            pred, _ = method.predict()
            dist     = _get_dist(method, vocab)
            method.observe(v)
            method.feedback(v)

            if pred is not None:
                hit = int(pred == v)
                block_correct += hit
                block_total   += 1
                all_correct   += hit
                all_total     += 1
                if i >= test_start:
                    test_correct += hit
                    test_total   += 1

            if block_total == block_size:
                block_accs[mname].append(block_correct / block_total)
                block_correct = block_total = 0

        overall[mname] = {
            'test_acc':  test_correct / max(test_total, 1),
            'train_acc': (all_correct - test_correct) / max(all_total - test_total, 1),
        }
        print(f'{overall[mname]["test_acc"]*100:.1f}%')

    # Table 7a: overall test accuracy
    vals = {m: overall[m]['test_acc'] for m in method_names}
    sv   = sorted(vals.values(), reverse=True)
    best, second = sv[0], (sv[1] if len(sv) > 1 else sv[0])
    row  = 'Electricity & ' + ' & '.join(
        _cell(vals[m], best, second) for m in method_names)
    col_spec = 'l' + 'r' * len(method_names)
    tex7 = (
        f'\\begin{{table}}[h]\n\\centering\n'
        f'\\caption{{Test accuracy (\\%) on the Electricity dataset '
        f'(Harries 1999; $n={n:,}$ steps, binary label: price UP/DOWN). '
        f'All methods use context order $k={k}$. '
        f'\\textbf{{Bold}} = best, \\underline{{underline}} = second best.}}\n'
        f'\\label{{tab:electricity_accuracy}}\n'
        f'\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n'
        f'Dataset & ' + ' & '.join(method_names) + ' \\\\\n'
        f'\\midrule\n{row} \\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n'
    )
    _write_table(f'{out_dir}/07_electricity_accuracy.tex', tex7)

    # Table 7b: per-block accuracy (first 10 blocks + last 10 blocks)
    n_show  = min(10, n_blocks)
    show_idx = list(range(n_show)) + (
        list(range(n_blocks - n_show, n_blocks)) if n_blocks > 2 * n_show else [])
    show_idx = sorted(set(show_idx))

    highlight = {'Predictor', 'Forest'}
    rows7b = []
    for bi in show_idx:
        step_lo = bi * block_size
        step_hi = step_lo + block_size
        parts = [f'{step_lo//1000}K--{step_hi//1000}K']
        bvals = {m: block_accs[m][bi] if bi < len(block_accs[m]) else 0.0
                 for m in method_names}
        sv_b  = sorted(bvals.values(), reverse=True)
        b_best, b_sec = sv_b[0], (sv_b[1] if len(sv_b) > 1 else sv_b[0])
        for m in method_names:
            parts.append(_cell(bvals[m], b_best, b_sec))
        rows7b.append(' & '.join(parts))

    if n_show < n_blocks:
        rows7b.insert(n_show, '\\multicolumn{' + str(len(method_names)+1) + '}{c}{$\\vdots$}')

    body7b = ' \\\\\n'.join(rows7b)
    tex7b = (
        f'\\begin{{table}}[h]\n\\centering\n'
        f'\\caption{{Per-{block_size}-step rolling accuracy (\\%) on Electricity. '
        f'First and last {n_show} blocks shown. '
        f'Drift is gradual throughout the 2-year stream.}}\n'
        f'\\label{{tab:electricity_blocks}}\n'
        f'\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n'
        f'Steps & ' + ' & '.join(method_names) + ' \\\\\n'
        f'\\midrule\n{body7b} \\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n'
    )
    _write_table(f'{out_dir}/07b_electricity_blocks.tex', tex7b)

    # Learning-curve plot
    if _MPL:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        xs = [(i + 0.5) * block_size for i in range(n_blocks)]
        for mname in method_names:
            ys = [a * 100 for a in block_accs[mname]]
            style = {
                'Predictor': dict(lw=2.5, color='tab:blue'),
                'Forest':    dict(lw=2.5, color='tab:orange'),
            }.get(mname, dict(lw=1, ls='--', alpha=0.55))
            ax.plot(xs[:len(ys)], ys, label=mname, **style)
        ax.axhline(50, color='gray', ls=':', lw=1, alpha=0.5, label='50\\% baseline')
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy (\\%)')
        ax.set_title('Electricity — rolling accuracy')
        ax.legend(fontsize=7, ncol=3)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = f'{out_dir}/fig_electricity_blocks.pdf'
        fig.savefig(path)
        plt.close(fig)
        print(f'  Wrote {path}')

    # Print quick analysis to console
    print(f'\n  === Electricity analysis (n={n:,}, k={k}) ===')
    ppm_key = f'PPM-D({k})'
    pred_key = 'Predictor'
    for m in method_names:
        delta = (overall[m]['test_acc'] - overall.get(ppm_key, {}).get('test_acc', 0)) * 100
        print(f'  {m:20s}  test={overall[m]["test_acc"]*100:.1f}%  '
              f'Δvs{ppm_key}={delta:+.1f}pp')

    # First-half vs second-half per method
    half = n_blocks // 2
    print(f'\n  First-half vs second-half accuracy (block 0..{half-1} vs {half}..{n_blocks-1}):')
    for m in method_names:
        first  = sum(block_accs[m][:half]) / max(half, 1) * 100
        second = sum(block_accs[m][half:]) / max(n_blocks - half, 1) * 100
        print(f'  {m:20s}  first={first:.1f}%  second={second:.1f}%  '
              f'drift_delta={second-first:+.1f}pp')


def _write_master_tex(out_dir):
    """Write a master .tex file that inputs all tables."""
    tables = [
        ('01_main_accuracy.tex',
         '% Table 1: Main accuracy comparison'),
        ('02_logloss.tex',
         '% Table 2: Log-loss (bits/symbol)'),
        ('03_drift_accuracy.tex',
         '% Table 3: Concept-drift test accuracy'),
        ('03b_postdrift_recovery.tex',
         '% Table 3b: Post-drift recovery'),
        ('04_ablation.tex',
         '% Table 4: Ablation study'),
        ('05_significance.tex',
         '% Table 5: Statistical significance'),
        ('06_scaling.tex',
         '% Table 6: Computational scaling'),
        ('07_electricity_accuracy.tex',
         '% Table 7: Electricity dataset accuracy'),
        ('07b_electricity_blocks.tex',
         '% Table 7b: Electricity per-block accuracy'),
        ('08_extended_comparison.tex',
         '% Table 8: Extended baseline comparison (KN, PPM*, LSTM)'),
    ]
    lines = [
        '% Auto-generated by ieee_benchmark.py',
        '% Place \\input{ieee_tables/all_tables.tex} in your IEEE article.',
        '',
        _TEX_HEADER,
        '',
    ]
    for fname, comment in tables:
        lines += [comment, f'\\input{{ieee_tables/{fname}}}', '']

    path = os.path.join(out_dir, 'all_tables.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  Wrote {path}')


if __name__ == '__main__':
    main()
