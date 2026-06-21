"""
Forest architectural tests — one claim per test.

  grow_mechanism       — spawns after grow_threshold unanimous-wrong steps
  grow_resets          — streak resets when prediction is not unanimous-wrong
  grow_requires_unan   — does NOT fire when trees predict different wrong answers
  prune_mechanism      — deactivates tree below credibility floor
  prune_resets         — counter resets when tree recovers above floor
  max_trees_cap        — _spawn_tree never exceeds max_trees
  min_active_floor     — prune never drops below 2 active trees
  adaptive_voting      — not catastrophically worse than mixture on either regime
  regression_output    — predict() returns float mean, confidence in [0,1]
  classification_eq    — task='classification' identical output to task='sequence'
"""

from forest import PredictorForest
from similarity import gaussian, hamming
from datasets import load_airline_passengers, random_integers
from run_experiments import discretize, normalize


# ── helpers ────────────────────────────────────────────────────────────────────

def _run(seq, sim_fn, k, voting='adaptive', task='sequence',
         n_trees=3, auto_grow=False, auto_prune=False, seed=42, **kw):
    n       = len(seq)
    train_n = int(n * 0.8)
    f = PredictorForest(k, sim_fn, n_trees=n_trees, voting=voting, task=task,
                        auto_grow=auto_grow, auto_prune=auto_prune, seed=seed, **kw)
    for i, v in enumerate(seq[:train_n]):
        if i >= k:
            f.predict()
        f.observe(v)
        if i >= k:
            f.feedback(v)
    correct = total = 0
    reg_preds = []
    for v in seq[train_n:]:
        pred, conf = f.predict()
        f.observe(v)
        f.feedback(v)
        if pred is not None:
            if task == 'regression':
                reg_preds.append((pred, conf, v))
            else:
                correct += int(pred == v)
            total += 1
    acc = correct / total if total > 0 and task != 'regression' else 0.0
    return acc, f, reg_preds


# ── grow / prune mechanism tests ───────────────────────────────────────────────

def test_grow_mechanism():
    """Spawns exactly one tree after grow_threshold unanimous-wrong steps."""
    f = PredictorForest(3, None, n_trees=2, grow_threshold=3, max_trees=10,
                        auto_grow=True, auto_prune=False, heterogeneous_k=False)
    f._last_preds = ['X', 'X']

    for _ in range(3):
        f._check_grow([0, 1], 'Y')

    assert len(f.trees) == 3,       f"Expected 3 trees, got {len(f.trees)}"
    assert f._n_spawned == 1
    assert f._corr_fail_str == 0,   "Streak must reset after spawn"
    assert f.trees[-1].k == 4,      f"New tree k should be 4, got {f.trees[-1].k}"


def test_grow_resets_on_non_unanimous():
    """Streak counter resets if any tree breaks the unanimous-wrong condition."""
    f = PredictorForest(3, None, n_trees=2, grow_threshold=5, max_trees=10,
                        auto_grow=True, auto_prune=False, heterogeneous_k=False)
    f._last_preds = ['X', 'X']

    f._check_grow([0, 1], 'Y')
    f._check_grow([0, 1], 'Y')
    assert f._corr_fail_str == 2

    f._last_preds[0] = 'Y'          # tree 0 now correct — not unanimous
    f._check_grow([0, 1], 'Y')
    assert f._corr_fail_str == 0,   f"Streak should reset, got {f._corr_fail_str}"
    assert len(f.trees) == 2


def test_grow_requires_unanimous():
    """Does NOT fire when trees predict different wrong answers."""
    f = PredictorForest(3, None, n_trees=2, grow_threshold=3, max_trees=10,
                        auto_grow=True, auto_prune=False, heterogeneous_k=False)
    f._last_preds = ['X', 'Z']      # both wrong but different

    for _ in range(10):
        f._check_grow([0, 1], 'Y')

    assert len(f.trees) == 2,       "Should not spawn on non-unanimous wrong predictions"
    assert f._n_spawned == 0


def test_prune_mechanism():
    """Deactivates tree after prune_window steps below credibility floor."""
    f = PredictorForest(3, None, n_trees=3, auto_grow=False, auto_prune=True,
                        prune_floor=0.5, prune_window=3)
    # tree 0 cred = 0.05; mean ≈ 0.68; floor × mean ≈ 0.34 — tree 0 well below
    f._tree_creds = [0.05, 1.0, 1.0]

    for _ in range(3):
        f._check_prune([0, 1, 2])

    assert 0 in f._inactive,        "Tree 0 should be deactivated"
    assert len(f._active) == 2,     f"Expected 2 active, got {len(f._active)}"


def test_prune_counter_resets_on_recovery():
    """Prune counter resets when the tree climbs back above the floor."""
    f = PredictorForest(3, None, n_trees=3, auto_grow=False, auto_prune=True,
                        prune_floor=0.5, prune_window=5)
    f._tree_creds = [0.05, 1.0, 1.0]

    f._check_prune([0, 1, 2])       # counter[0] = 1
    f._check_prune([0, 1, 2])       # counter[0] = 2
    assert f._prune_ctrs[0] == 2

    f._tree_creds[0] = 1.0          # recovery
    f._check_prune([0, 1, 2])
    assert f._prune_ctrs[0] == 0,   f"Counter should reset after recovery, got {f._prune_ctrs[0]}"
    assert 0 not in f._inactive


def test_max_trees_cap():
    """_spawn_tree never exceeds max_trees regardless of how many times called."""
    f = PredictorForest(2, None, n_trees=2, max_trees=4,
                        auto_grow=True, auto_prune=False)
    for _ in range(20):
        f._spawn_tree()
    assert len(f.trees) <= 4, f"Exceeded max_trees: {len(f.trees)}"


def test_min_active_floor():
    """auto_prune preserves at least 2 active trees."""
    f = PredictorForest(3, None, n_trees=3, auto_grow=False, auto_prune=True,
                        prune_floor=0.5, prune_window=1)

    # Prune tree 0 — drops to 2 active
    f._tree_creds = [0.01, 1.0, 1.0]
    for _ in range(3):
        active = f._active
        if len(active) > 2:
            f._check_prune(active)

    assert len(f._active) == 2

    # Now try to prune with only 2 active — should be blocked
    f._tree_creds[1] = 0.01
    for _ in range(5):
        f._check_prune(f._active)   # active=[1,2], len=2 → returns immediately

    assert len(f._active) == 2,     f"Floor violated: {len(f._active)} active trees"


# ── voting tests ───────────────────────────────────────────────────────────────

def test_adaptive_voting():
    """
    Adaptive is not catastrophically worse than mixture on either structured
    or random data.  On structured data it may exceed mixture (product-mode
    boost when trees are confident); on random data it should stay near baseline.
    """
    raw        = load_airline_passengers()
    structured = discretize(normalize(raw), n_bins=8)
    rand_seq   = random_integers(n=500, low=0, high=9, seed=7)

    results = {}
    for v in ('mixture', 'product', 'adaptive'):
        acc_s, _, _ = _run(structured, gaussian(sigma=2.0), 4, voting=v, n_trees=3)
        acc_r, _, _ = _run(rand_seq,   gaussian(sigma=1.0), 3, voting=v, n_trees=3)
        results[v] = (acc_s, acc_r)

    mix_s,  mix_r  = results['mixture']
    ada_s,  ada_r  = results['adaptive']
    baseline_r     = 1.0 / len(set(rand_seq))

    # Adaptive not catastrophically worse than mixture on structured data
    assert ada_s >= mix_s * 0.75, \
        f"Adaptive {ada_s:.3f} much worse than mixture {mix_s:.3f} on structured data"

    # Adaptive stays reasonable on random data (doesn't crater below half-baseline)
    assert ada_r >= baseline_r * 0.4, \
        f"Adaptive {ada_r:.3f} far below baseline {baseline_r:.3f} on random data"

    return results


# ── task type tests ────────────────────────────────────────────────────────────

def test_regression_output():
    """
    task='regression' returns float mean predictions; confidence stays in [0,1];
    MAE is no worse than 1.5× the grand-mean baseline.
    """
    raw = load_airline_passengers()
    seq = discretize(normalize(raw), n_bins=8)   # bins are ints, but regression returns float

    n       = len(seq)
    train_n = int(n * 0.8)

    f = PredictorForest(4, gaussian(sigma=2.0), n_trees=3, voting='adaptive',
                        task='regression', auto_grow=False, auto_prune=False, seed=42)
    for i, v in enumerate(seq[:train_n]):
        if i >= 4:
            f.predict()
        f.observe(v)
        if i >= 4:
            f.feedback(v)

    preds, actuals = [], []
    for v in seq[train_n:]:
        pred, conf = f.predict()
        f.observe(v)
        f.feedback(v)
        if pred is not None:
            assert isinstance(pred, float), f"Expected float, got {type(pred)}: {pred}"
            assert 0.0 <= conf <= 1.0,      f"Confidence {conf:.4f} outside [0,1]"
            preds.append(pred)
            actuals.append(v)

    assert preds, "No regression predictions made"

    mae          = sum(abs(p - a) for p, a in zip(preds, actuals)) / len(preds)
    grand_mean   = sum(seq[:train_n]) / len(seq[:train_n])
    baseline_mae = sum(abs(grand_mean - a) for a in actuals) / len(actuals)

    assert mae <= baseline_mae * 1.5, \
        f"Regression MAE {mae:.3f} much worse than baseline {baseline_mae:.3f}"

    return mae, baseline_mae


def test_classification_equals_sequence():
    """task='classification' and task='sequence' produce byte-identical outputs."""
    raw = load_airline_passengers()
    seq = discretize(normalize(raw), n_bins=8)

    f_cls = PredictorForest(4, gaussian(sigma=2.0), n_trees=3, voting='adaptive',
                             task='classification', auto_grow=False, auto_prune=False, seed=0)
    f_seq = PredictorForest(4, gaussian(sigma=2.0), n_trees=3, voting='adaptive',
                             task='sequence',        auto_grow=False, auto_prune=False, seed=0)

    for v in seq:
        p1, c1 = f_cls.predict()
        p2, c2 = f_seq.predict()
        assert p1 == p2,                   f"Prediction differs: cls={p1} seq={p2}"
        assert abs(c1 - c2) < 1e-10,      f"Confidence differs: cls={c1} seq={c2}"
        f_cls.observe(v);  f_seq.observe(v)
        f_cls.feedback(v); f_seq.feedback(v)


# ── end-to-end sizing comparison ───────────────────────────────────────────────

def _e2e_sizing(label, grow, prune, seq, k, sim_fn):
    f = PredictorForest(k, sim_fn, n_trees=3, voting='adaptive',
                        auto_grow=grow, auto_prune=prune,
                        grow_threshold=8, prune_window=20, max_trees=10, seed=42)
    n = len(seq); train_n = int(n * 0.8)
    for i, v in enumerate(seq[:train_n]):
        if i >= k: f.predict()
        f.observe(v)
        if i >= k: f.feedback(v)
    correct = total = 0
    for v in seq[train_n:]:
        pred, _ = f.predict()
        f.observe(v); f.feedback(v)
        if pred is not None:
            correct += int(pred == v); total += 1
    ns  = f.node_stats()
    acc = correct / total if total > 0 else 0.0
    return acc, ns['n_active'], ns['n_total'], ns['n_spawned'], ns['n_inactive']


# ── orchestrator ───────────────────────────────────────────────────────────────

def main():
    print("Forest Architectural Test Suite\n")

    unit_tests = [
        ("Grow: triggers after threshold",          test_grow_mechanism),
        ("Grow: resets on non-unanimous step",      test_grow_resets_on_non_unanimous),
        ("Grow: requires unanimous wrong answer",   test_grow_requires_unanimous),
        ("Prune: deactivates below-floor tree",     test_prune_mechanism),
        ("Prune: counter resets on recovery",       test_prune_counter_resets_on_recovery),
        ("Sizing: max_trees cap enforced",          test_max_trees_cap),
        ("Sizing: min 2 active trees preserved",    test_min_active_floor),
        ("Regression: returns float mean + conf",   test_regression_output),
        ("Classification: identical to sequence",   test_classification_equals_sequence),
    ]

    all_pass = True
    for name, fn in unit_tests:
        try:
            result = fn()
            extra = ""
            if isinstance(result, tuple) and len(result) == 2:
                extra = f"  (MAE {result[0]:.3f} vs baseline {result[1]:.3f})"
            print(f"  [PASS] {name}{extra}")
        except AssertionError as e:
            print(f"  [FAIL] {name}\n         {e}")
            all_pass = False
        except Exception as e:
            print(f"  [ERR ] {name}\n         {type(e).__name__}: {e}")
            all_pass = False

    # ── adaptive voting comparison ─────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  Adaptive voting comparison\n")
    try:
        res = test_adaptive_voting()
        print(f"  {'Voting':<10} {'Structured':>11} {'Random':>8}")
        for v, (acc_s, acc_r) in res.items():
            print(f"  {v:<10} {acc_s:>11.3f} {acc_r:>8.3f}")
        print(f"  [PASS] adaptive_voting")
    except AssertionError as e:
        print(f"  [FAIL] adaptive_voting\n         {e}")
        all_pass = False
    except Exception as e:
        print(f"  [ERR ] adaptive_voting\n         {type(e).__name__}: {e}")
        all_pass = False

    # ── end-to-end dynamic sizing ─────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  Dynamic sizing — Airline Passengers (end-to-end)\n")
    raw = load_airline_passengers()
    seq = discretize(normalize(raw), n_bins=8)
    sim = gaussian(sigma=2.0)

    print(f"  {'Config':<14} {'Acc':>5}  {'Active/Total':>13}  {'Spawned':>7}  {'Inactive':>8}")
    for label, grow, prune in [
        ("fixed",        False, False),
        ("auto_grow",    True,  False),
        ("auto_prune",   False, True),
        ("grow+prune",   True,  True),
    ]:
        acc, n_act, n_tot, n_sp, n_in = _e2e_sizing(label, grow, prune, seq, 4, sim)
        print(f"  {label:<14} {acc:>5.3f}  {n_act}/{n_tot:>2} active      {n_sp:>7}  {n_in:>8}")

    print(f"\n{'═'*62}")
    print("  All unit tests PASSED" if all_pass else "  Some tests FAILED")


if __name__ == "__main__":
    main()
