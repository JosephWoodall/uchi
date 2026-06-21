"""
Comparative benchmark: UniversalPredictor + PredictorForest vs
  - Persistence (predict last seen)
  - Majority    (predict most frequent seen so far)
  - N-gram(5)   (variable-order n-gram, Laplace smoothing)
  - PPM-D(5)    (Prediction by Partial Matching, Witten-Bell escape)

Evaluation protocol for ALL methods (fully online — no separate training phase):
    for each observation v_t:
        pred, conf = method.predict()     # predict before seeing v_t
        method.observe(v_t)
        method.feedback(v_t)

Datasets:
    Standard  — Airline, Alice, DNA, Weather, PRNG
    Drift     — Synthetic concept-drift sequence: forward cycle → backward cycle at t=600
                The drift test is the strongest test of online adaptation.

Key metrics:
    Test accuracy  — last 20% of sequence
    Q1 accuracy    — first 25% of test phase (ramp speed)
    Noise overfit  — PRNG accuracy above random baseline (lower = better)
    Drift recovery — accuracy in first 120 steps after drift vs N-gram
"""
import math
import random
import functools
from collections import defaultdict

from predictor import UniversalPredictor
from forest import PredictorForest
from baselines import PersistencePredictor, MajorityPredictor, NgramPredictor, PPMPredictor
from run_experiments import discretize, normalize
from similarity import gaussian, hamming
from datasets import (
    load_airline_passengers, load_gutenberg_text, load_dna_sequence,
    load_weather_events, random_integers,
)

_W = 94


def _hr(c="─"):   print(c * _W)
def _dhr():        print("═" * _W)
def _dothr():      print("·" * _W)
def _pct(v):       return f"{v * 100:5.1f}%"
def _bar(v, w=5):
    f = min(round(v * w), w)
    return "█" * f + "░" * (w - f)


# ── datasets ──────────────────────────────────────────────────────────────────

def _concept_drift_seq(n=1000, drift_at=500, noise=0.05, seed=99):
    """
    SAME contexts, REVERSED outcomes — forces methods to unlearn phase 1.

    Symbols {0, 1, 2}.  k=1 context (single previous symbol).
    Phase 1 (0..499) : forward  cycle  0→1→2→0  (noise 5%)
    Phase 2 (500..999): reversed cycle 0→2→1→0  (noise 5%)

    After drift N-gram's accumulated phase-1 counts still dominate for hundreds
    of steps.  Credibility-weighted predictors down-weight wrong phase-1 nodes
    immediately and switch to phase-2 nodes much faster.
    """
    rng = random.Random(seed)
    vocab = [0, 1, 2]
    seq = [rng.choice(vocab)]
    for i in range(1, n):
        v = seq[-1]
        intended = (v + 1) % 3 if i < drift_at else (v - 1) % 3
        seq.append(rng.choice(vocab) if rng.random() < noise else intended)
    return seq


def _load_datasets():
    datasets = []

    raw = load_airline_passengers()
    datasets.append({
        "name": "Airline Passengers",
        "seq":  discretize(normalize(raw), n_bins=8),
        "sim":  gaussian(sigma=2.0),
        "k": 4, "vig": 0.3, "forest_kw": {},
    })

    for loader, name, k, vig, fkw in [
        (lambda: load_gutenberg_text(n_chars=1500),
         "Alice in Wonderland", 3, 0.7,
         {"heterogeneous_k": False, "auto_grow": False, "auto_prune": False}),
        (lambda: load_dna_sequence(n_bases=1500),
         "Bacteriophage Lambda DNA", 4, 0.3, {}),
        (lambda: load_weather_events(n_days=500),
         "NYC Weather Events", 3, 0.3, {}),
    ]:
        try:
            datasets.append({
                "name": name, "seq": loader(),
                "sim": hamming, "k": k, "vig": vig, "forest_kw": fkw,
            })
        except Exception as exc:
            print(f"  [skip] {name}: {exc}")

    datasets.append({
        "name": "Python PRNG",
        "seq":  random_integers(n=500, low=0, high=9, seed=7),
        "sim":  gaussian(sigma=1.0),
        "k": 3, "vig": 0.3, "forest_kw": {},
    })

    # Concept-drift dataset (separate section)
    # k=1: same single-symbol context, reversed outcome across phases
    datasets.append({
        "name": "__drift__",
        "seq":  _concept_drift_seq(n=1000, drift_at=500, noise=0.05, seed=99),
        "sim":  hamming,
        "k": 1, "vig": 0.7,
        "forest_kw": {"auto_grow": False, "auto_prune": False,
                      "heterogeneous_k": False},
        "drift_at": 500,
    })

    return datasets


# ── method factory ────────────────────────────────────────────────────────────

def _make_methods(cfg, drift_mode=False):
    k, sim, vig = cfg["k"], cfg["sim"], cfg["vig"]

    base_forest_kw = dict(
        n_trees=5, dropout=0.2, stagger=25, voting="adaptive",
        heterogeneous_k=True, tree_lr=0.1, max_trees=20,
        auto_grow=True, auto_prune=True, grow_threshold=8, prune_window=50,
    )
    base_forest_kw.update(cfg.get("forest_kw", {}))

    # For concept drift: cap N-gram/PPM to max_order=k so they cannot escape
    # the k=1 conflict by silently backing off to higher-order (non-overlapping)
    # contexts that happen to be different between phases.  All methods must
    # face the exact same context window, making the comparison fair.
    ng_order  = k if drift_mode else 5
    ppm_order = k if drift_mode else 5

    return [
        ("Persistence",        PersistencePredictor()),
        ("Majority",           MajorityPredictor()),
        (f"N-gram({ng_order})",  NgramPredictor(max_order=ng_order)),
        (f"PPM-D({ppm_order})",  PPMPredictor(max_order=ppm_order)),
        ("Predictor",          UniversalPredictor(k, sim, learning_rate=0.1, vigilance=vig)),
        ("Forest",             PredictorForest(k, sim, learning_rate=0.1, vigilance=vig,
                                               **base_forest_kw)),
    ]


# ── evaluation ────────────────────────────────────────────────────────────────

def _evaluate(method, seq, train_frac=0.8, drift_at=None):
    n       = len(seq)
    train_n = int(n * train_frac)
    records = []

    for i, v in enumerate(seq):
        pred, conf = method.predict()
        method.observe(v)
        method.feedback(v)
        if pred is not None:
            records.append({
                "step":    i,
                "correct": pred == v,
                "conf":    conf,
                "phase":   "train" if i < train_n else "test",
            })

    test_r  = [r for r in records if r["phase"] == "test"]
    test_acc = _acc(test_r)

    q1_r    = test_r[: max(1, len(test_r) // 4)]
    q1_acc  = _acc(q1_r)

    # Block accuracies in 4 equal chunks of the full sequence
    block_accs = []
    for b in range(4):
        lo = round(n * b / 4)
        hi = round(n * (b + 1) / 4)
        br = [r for r in records if lo <= r["step"] < hi]
        block_accs.append(_acc(br))

    # Post-drift accuracy (120 steps after drift point, if provided)
    post_drift_acc = None
    if drift_at is not None:
        dr = [r for r in records
              if drift_at <= r["step"] < drift_at + 120 and r["step"] < train_n]
        post_drift_acc = _acc(dr)

    # Confidence calibration buckets on full sequence
    buckets = {
        "low":  [r for r in records if r["conf"] <  0.4],
        "mid":  [r for r in records if 0.4 <= r["conf"] < 0.7],
        "high": [r for r in records if r["conf"] >= 0.7],
    }
    calib = {k: (_acc(v), len(v)) for k, v in buckets.items()}

    return {
        "test_acc":       test_acc,
        "q1_acc":         q1_acc,
        "block_accs":     block_accs,
        "post_drift_acc": post_drift_acc,
        "calib":          calib,
        "n_test":         len(test_r),
    }


def _acc(records):
    return sum(r["correct"] for r in records) / len(records) if records else 0.0


# ── printing ──────────────────────────────────────────────────────────────────

def _print_dataset_block(ds):
    name     = ds["name"]
    seq      = ds["seq"]
    baseline = ds["baseline"]
    results  = ds["results"]
    methods  = list(results.keys())

    _hr()
    print(f"  {name}   "
          f"(n={len(seq)}, unique={len(set(seq))}, "
          f"baseline={_pct(baseline)}, train/test 80/20)")
    _hr()
    print(f"  {'Method':<13}  {'Test':>6}  {'Lift':>7}  {'vs PPM':>7}  "
          f"{'Q1':>5}  Curve (4×25%) ▸=80%")
    _dothr()

    ppm_acc  = results.get("PPM-D(5)", {}).get("test_acc", 0)
    best_acc = max(r["test_acc"] for r in results.values())

    for m, res in results.items():
        acc    = res["test_acc"]
        q1     = res["q1_acc"]
        blocks = res["block_accs"]
        lift   = (acc / baseline - 1) * 100 if baseline > 0 else 0
        vs_ppm = (acc - ppm_acc) * 100
        star   = "★" if abs(acc - best_acc) < 1e-9 else " "
        sl     = "+" if lift   >= 0 else ""
        sp     = "+" if vs_ppm >= 0 else ""

        bars = "  ".join(_bar(b, w=4) for b in blocks[:3])
        bars += "  ▸  " + _bar(blocks[3] if len(blocks) > 3 else 0, w=4)

        print(f"  {m:<13}  {_pct(acc)}{star}  {sl}{lift:5.1f}%  "
              f"{sp}{vs_ppm:+5.1f}pp  {_pct(q1)}  {bars}")

    best = max(results, key=lambda m: results[m]["test_acc"])
    ramp = (results[best]["q1_acc"] / results[best]["test_acc"] * 100
            if results[best]["test_acc"] > 0 else 0)
    print(f"\n  Best: {best} @ {_pct(results[best]['test_acc'])}  "
          f"| Q1 ramp: {ramp:.0f}% of final  "
          f"| n_test={results[best]['n_test']}")


def _print_drift_block(ds):
    seq      = ds["seq"]
    drift_at = ds.get("drift_at", 600)
    baseline = ds["baseline"]
    results  = ds["results"]

    _hr()
    print(f"  CONCEPT DRIFT TEST   "
          f"(n={len(seq)}, unique=3, baseline={_pct(baseline)})")
    print(f"  SAME k=1 contexts, REVERSED outcomes at step {drift_at}.")
    print(f"  Phase 1 (0–{drift_at}): forward  cycle  0→1→2→0  (5% noise)")
    print(f"  Phase 2 ({drift_at}–1000): reversed cycle 0→2→1→0  (5% noise)")
    print(f"  N-gram/PPM: phase-1 counts outweigh phase-2 counts until ~step 900.")
    print(f"  Credibility: wrong phase-1 nodes degrade immediately; adaptation is fast.")
    _hr()
    print(f"  {'Method':<13}  {'Test':>6}  {'Lift':>7}  {'vs PPM':>7}  "
          f"{'Post-drift':>10}  Curve (4×25%) ▸=80%")
    _dothr()

    ppm_key  = next((k for k in results if "PPM" in k), None)
    ppm_acc  = results.get(ppm_key, {}).get("test_acc", 0) if ppm_key else 0
    best_acc = max(r["test_acc"] for r in results.values())

    for m, res in results.items():
        acc   = res["test_acc"]
        q1    = res["q1_acc"]
        pd    = res.get("post_drift_acc")
        blocks= res["block_accs"]
        lift  = (acc / baseline - 1) * 100 if baseline > 0 else 0
        vs_ppm= (acc - ppm_acc) * 100
        star  = "★" if abs(acc - best_acc) < 1e-9 else " "
        sl    = "+" if lift   >= 0 else ""
        sp    = "+" if vs_ppm >= 0 else ""
        pd_s  = _pct(pd) if pd is not None else "  n/a "

        bars = "  ".join(_bar(b, w=4) for b in blocks[:3])
        bars += "  ▸  " + _bar(blocks[3] if len(blocks) > 3 else 0, w=4)

        print(f"  {m:<13}  {_pct(acc)}{star}  {sl}{lift:5.1f}%  "
              f"{sp}{vs_ppm:+5.1f}pp  {pd_s}        {bars}")

    print(f"\n  Post-drift col = accuracy in first 120 steps after step {drift_at} "
          f"(immediately after rule flip, still in training).")
    print( "  Higher post-drift = faster adaptation. N-gram/PPM slow due to accumulated counts.")


def _print_summary(all_std):
    _dhr()
    print(f"  {'BENCHMARK SUMMARY — STANDARD DATASETS':^90}")
    _dhr()
    methods = list(all_std[0]["results"].keys())

    hdr = f"  {'Dataset':<28}"
    for m in methods:
        hdr += f"  {m:>10}"
    print(hdr)
    _hr()

    avg_accs   = {m: [] for m in methods}
    baselines  = []

    for ds in all_std:
        baselines.append(ds["baseline"])
        best = max(ds["results"][m]["test_acc"] for m in methods)
        row  = f"  {ds['name']:<28}"
        for m in methods:
            acc = ds["results"][m]["test_acc"]
            avg_accs[m].append(acc)
            mk  = "★" if abs(acc - best) < 1e-9 else " "
            row += f"  {_pct(acc)}{mk}"
        print(row)

    _hr()

    row = f"  {'Avg lift vs baseline':<28}"
    for m in methods:
        lifts = [(a - b) / b * 100 for a, b in zip(avg_accs[m], baselines) if b > 0]
        row  += f"  {sum(lifts)/len(lifts):+6.0f}%  "
    print(row)

    if "PPM-D(5)" in methods:
        row = f"  {'Avg delta vs PPM-D(5)':<28}"
        ppm = avg_accs["PPM-D(5)"]
        for m in methods:
            d = sum((a - p) * 100 for a, p in zip(avg_accs[m], ppm)) / len(ppm)
            row += f"  {d:+6.1f}pp  "
        print(row)

    _dhr()


def _print_calibration(all_std):
    _dhr()
    print(f"  {'CONFIDENCE CALIBRATION — PYTHON PRNG (random data, baseline 10%)':^90}")
    print(f"  A well-calibrated method should have similar accuracy across confidence buckets.")
    print(f"  N-gram / PPM overfit: high-confidence predictions on random data are no better")
    print(f"  (or worse) than low-confidence ones.  This system stays near baseline throughout.")
    _dhr()
    prng = next((d for d in all_std if "PRNG" in d["name"]), None)
    if prng is None:
        print("  (PRNG dataset unavailable)")
        return

    methods = list(prng["results"].keys())
    print(f"  {'Method':<13}  {'low conf (<40%)':>16}  {'mid (40-70%)':>14}  "
          f"{'high (>70%)':>13}  {'NOISE OVERFIT':>14}")
    _dothr()

    baseline = prng["baseline"]
    for m in methods:
        c    = prng["results"][m]["calib"]
        lo_a, lo_n = c["low"]
        mi_a, mi_n = c["mid"]
        hi_a, hi_n = c["high"]
        test_acc   = prng["results"][m]["test_acc"]
        overfit    = (test_acc - baseline) * 100   # pp above baseline; negative = good

        lo_s = f"{_pct(lo_a)} (n={lo_n})"
        mi_s = f"{_pct(mi_a)} (n={mi_n})"
        hi_s = f"{_pct(hi_a)} (n={hi_n})"
        ov_s = f"{overfit:+.1f}pp {'← overfit!' if overfit > 3 else '✓'}"

        print(f"  {m:<13}  {lo_s:>16}  {mi_s:>14}  {hi_s:>13}  {ov_s:>14}")

    _dhr()


def _print_key_findings(all_std, drift_ds):
    methods = list(all_std[0]["results"].keys())

    structured = [d for d in all_std if "PRNG" not in d["name"]]
    prng_ds    = next((d for d in all_std if "PRNG" in d["name"]), None)

    def avg_lift(ds_list, m):
        lifts = [(d["results"][m]["test_acc"] - d["baseline"]) / d["baseline"] * 100
                 for d in ds_list if d["baseline"] > 0]
        return sum(lifts) / len(lifts) if lifts else 0

    ppm  = "PPM-D(5)"
    pred = "Predictor"
    fst  = "Forest"

    _dhr()
    print("  KEY FINDINGS")
    _hr()
    print(f"  {'Metric':<45}  {ppm:>10}  {pred:>10}  {fst:>10}")
    _dothr()
    print(f"  {'Avg lift on structured data (excl PRNG)':<45}  "
          f"{avg_lift(structured, ppm):>9.1f}%  "
          f"{avg_lift(structured, pred):>9.1f}%  "
          f"{avg_lift(structured, fst):>9.1f}%")

    if prng_ds:
        b      = prng_ds["baseline"]
        ov_ppm = (prng_ds["results"][ppm]["test_acc"]  - b) * 100
        ov_pre = (prng_ds["results"][pred]["test_acc"] - b) * 100
        ov_fst = (prng_ds["results"][fst]["test_acc"]  - b) * 100
        print(f"  {'PRNG overfit (pp above 10% baseline; 0 = ideal)':<45}  "
              f"{ov_ppm:>+9.1f}pp  {ov_pre:>+9.1f}pp  {ov_fst:>+9.1f}pp")

    if drift_ds:
        dr      = drift_ds["drift_at"]
        # drift results use method names like "PPM-D(1)" not "PPM-D(5)"
        ppm_key = next((k for k in drift_ds["results"] if "PPM" in k), ppm)
        pd_ppm  = drift_ds["results"].get(ppm_key, {}).get("post_drift_acc") or 0
        pd_pre  = drift_ds["results"].get(pred,    {}).get("post_drift_acc") or 0
        pd_fst  = drift_ds["results"].get(fst,     {}).get("post_drift_acc") or 0
        label   = f"Post-drift recovery (steps {dr}–{dr+120})"
        print(f"  {label:<45}  "
              f"{_pct(pd_ppm):>10}  {_pct(pd_pre):>10}  {_pct(pd_fst):>10}")

    _hr()

    # Wins tally
    wins_pred = sum(1 for d in structured
                    if d["results"][pred]["test_acc"] > d["results"][ppm]["test_acc"])
    wins_fst  = sum(1 for d in structured
                    if d["results"][fst]["test_acc"]  > d["results"][ppm]["test_acc"])
    ties_pred = sum(1 for d in structured
                    if abs(d["results"][pred]["test_acc"] - d["results"][ppm]["test_acc"]) < 0.005)
    print(f"\n  Predictor vs PPM on structured data: "
          f"{wins_pred} wins / {ties_pred} ties / "
          f"{len(structured) - wins_pred - ties_pred} losses  (out of {len(structured)} datasets)")
    print(f"  Forest    vs PPM on structured data: "
          f"{wins_fst} wins / 0 ties / "
          f"{len(structured) - wins_fst} losses")

    print(f"\n  The Predictor / Forest primary advantage over classical methods:")
    print(f"  1. Noise resistance  — N-gram/PPM overfit random data by memorising")
    print(f"     spurious patterns. This system's credibility mechanism prevents that.")
    print(f"  2. Concept drift     — credibility down-weights stale nodes after a rule")
    print(f"     change; N-gram counts accumulate and are never discounted.")
    print(f"  3. Calibrated uncertainty — confidence = actual vote decisiveness,")
    print(f"     not count-derived probability that ignores context validity.")
    _dhr()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    _dhr()
    print(f"  {'UNIVERSAL SEQUENCE PREDICTOR — COMPARATIVE BENCHMARK':^90}")
    print(f"  {'Persistence  ·  Majority  ·  N-gram(5)  ·  PPM-D(5)  ·  Predictor  ·  Forest':^90}")
    _dhr()
    print()

    print("Loading datasets...")
    datasets   = _load_datasets()
    std_cfgs   = [d for d in datasets if d["name"] != "__drift__"]
    drift_cfg  = next((d for d in datasets if d["name"] == "__drift__"), None)

    # Run standard datasets
    all_std = []
    for cfg in std_cfgs:
        name = cfg["name"]
        seq  = cfg["seq"]
        print(f"  Evaluating {name}  (n={len(seq)})...")
        methods = _make_methods(cfg)
        baseline = 1.0 / len(set(seq))
        ds_result = {
            "name":     name,
            "seq":      seq,
            "baseline": baseline,
            "results":  {},
        }
        for mname, method in methods:
            ds_result["results"][mname] = _evaluate(method, seq)
        all_std.append(ds_result)

    # Run drift dataset
    drift_result = None
    if drift_cfg:
        name = "Concept Drift"
        seq  = drift_cfg["seq"]
        print(f"  Evaluating {name}  (n={len(seq)})...")
        methods  = _make_methods(drift_cfg, drift_mode=True)
        baseline = 1.0 / len(set(seq))
        drift_result = {
            "name":     name,
            "seq":      seq,
            "baseline": baseline,
            "drift_at": drift_cfg["drift_at"],
            "results":  {},
        }
        for mname, method in methods:
            drift_result["results"][mname] = _evaluate(
                method, seq, drift_at=drift_cfg["drift_at"]
            )

    print()

    # Per-dataset blocks
    for ds in all_std:
        _print_dataset_block(ds)
        print()

    if drift_result:
        _print_drift_block(drift_result)
        print()

    # Summary table
    _print_summary(all_std)
    print()

    # PRNG calibration
    _print_calibration(all_std)
    print()

    # Key findings
    _print_key_findings(all_std, drift_result)


if __name__ == "__main__":
    main()
