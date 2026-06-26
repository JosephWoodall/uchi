"""
Tests that probe the architecture's specific claims, not just end-to-end accuracy.

  calibration        — does confidence predict accuracy?
  hypothesis_quality — accuracy when venturing into unflagged territory
  credibility_gini   — are credibilities differentiated or flat?
  node_efficiency    — nodes needed to reach accuracy milestones
"""

import statistics
from collections import defaultdict

from uchi.predictor import UniversalPredictor
from similarity import gaussian, hamming
from uchi_datasets import (
    load_airline_passengers,
    load_gutenberg_text,
    load_dna_sequence,
    load_weather_events,
    random_integers,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def discretize(seq, n_bins=8):
    lo, hi = min(seq), max(seq)
    w = (hi - lo) / n_bins
    return [min(int((v - lo) / w), n_bins - 1) for v in seq]

def normalize(seq):
    mu = statistics.mean(seq)
    sd = statistics.stdev(seq) or 1.0
    return [(v - mu) / sd for v in seq]

def gini(values):
    n = len(values)
    if n < 2:
        return 0.0
    s = sorted(values)
    total = sum(s) or 1e-12
    numer = sum((2 * i - n - 1) * v for i, v in enumerate(s, 1))
    return numer / (n * total)


# ── core evaluation loop ──────────────────────────────────────────────────────

def _run(seq, sim_fn, context_length, learning_rate=0.1, vigilance=0.7):
    """Returns predictor + per-prediction records after full train+eval pass."""
    n       = len(seq)
    train_n = int(n * 0.8)

    predictor = UniversalPredictor(
        context_length, sim_fn,
        learning_rate=learning_rate,
        vigilance=vigilance,
    )

    for i, v in enumerate(seq[:train_n]):
        if i >= context_length:
            predictor.predict()
        predictor.observe(v)
        if i >= context_length:
            predictor.feedback(v)

    records = []
    for v in seq[train_n:]:
        pred, conf = predictor.predict()
        predictor.observe(v)
        predictor.feedback(v)
        if pred is not None:
            records.append({
                'correct':       int(pred == v),
                'confidence':    conf,
                'max_sim':       predictor._last_max_sim,
                'is_hypothesis': predictor._last_max_sim < vigilance,
                'node_count':    len(predictor._nodes),
            })

    return predictor, records


# ── individual tests ──────────────────────────────────────────────────────────

def test_calibration(records):
    buckets = defaultdict(list)
    for r in records:
        bucket = min(int(r['confidence'] * 10), 9)
        buckets[bucket].append(r['correct'])

    rows = []
    errors = []
    for b in range(10):
        if not buckets[b]:
            continue
        mid_conf = (b + 0.5) / 10
        acc      = sum(buckets[b]) / len(buckets[b])
        errors.append(abs(acc - mid_conf))
        rows.append((f'{b*10}-{b*10+10}%', mid_conf, acc, len(buckets[b])))

    score = 1.0 - (sum(errors) / len(errors)) if errors else 0.0
    return score, rows


def test_hypothesis_quality(records, baseline):
    hyp = [r for r in records if r['is_hypothesis']]
    std = [r for r in records if not r['is_hypothesis']]
    hyp_acc = sum(r['correct'] for r in hyp) / len(hyp) if hyp else None
    std_acc = sum(r['correct'] for r in std) / len(std) if std else None
    return hyp_acc, std_acc, len(hyp), len(std)


def test_credibility_gini(predictor):
    creds = [n.node_cred for n in predictor._nodes]
    return gini(creds) if creds else 0.0


def test_node_efficiency(seq, sim_fn, context_length, vigilance=0.7):
    n       = len(seq)
    train_n = int(n * 0.8)

    predictor = UniversalPredictor(context_length, sim_fn, vigilance=vigilance)

    for i, v in enumerate(seq[:train_n]):
        if i >= context_length:
            predictor.predict()
        predictor.observe(v)
        if i >= context_length:
            predictor.feedback(v)

    hits    = []
    nc      = []
    correct = total = 0

    for v in seq[train_n:]:
        pred, _ = predictor.predict()
        predictor.observe(v)
        predictor.feedback(v)
        if pred is not None:
            hits.append(int(pred == v))
            nc.append(len(predictor._nodes))
            correct += hits[-1]
            total   += 1

    final_acc = correct / total if total > 0 else 0.0

    milestone_50 = milestone_75 = None
    running = 0
    for i, (h, nodes) in enumerate(zip(hits, nc)):
        running += h
        rolling  = running / (i + 1)
        if milestone_50 is None and rolling >= 0.5 * final_acc:
            milestone_50 = nodes
        if milestone_75 is None and rolling >= 0.75 * final_acc:
            milestone_75 = nodes

    return final_acc, milestone_50, milestone_75, nc[-1] if nc else 0


# ── orchestrator ──────────────────────────────────────────────────────────────

def run_suite(name, seq, sim_fn, context_length=3, vigilance=None):
    if vigilance is None:
        vigilance = (context_length - 1) / context_length

    unique   = len(set(seq))
    baseline = 1.0 / unique

    predictor, records = _run(seq, sim_fn, context_length, vigilance=vigilance)

    if not records:
        print(f"\n  {name} — no predictions recorded")
        return

    overall_acc = sum(r['correct'] for r in records) / len(records)

    cal_score, cal_rows          = test_calibration(records)
    hyp_acc, std_acc, n_hyp, n_std = test_hypothesis_quality(records, baseline)
    cred_gini                    = test_credibility_gini(predictor)
    fin_acc, m50, m75, final_nodes = test_node_efficiency(
                                       seq, sim_fn, context_length, vigilance)

    lift = (overall_acc / baseline - 1) * 100
    prev = PREV.get(name, {})

    print(f"\n{'═'*62}")
    print(f"  {name}")
    print(f"  Baseline: {baseline:.3f}   Accuracy: {overall_acc:.3f}"
          f"   Lift: {lift:+.1f}%"
          + (f"  [prev {prev['lift']:+.1f}%]" if prev else ""))

    print(f"\n  [1] Calibration  (does confidence predict accuracy?)")
    print(f"      Score: {cal_score:.3f}  (1.0 = perfect, 0.0 = useless)")
    for conf_range, mid, acc, cnt in cal_rows:
        bar = '█' * int(acc * 20)
        gap = '░' * int(mid * 20)
        print(f"      {conf_range:>8s}  expected {mid:.2f}  actual {acc:.2f}"
              f"  n={cnt:3d}  {bar if acc >= mid else gap}")

    print(f"\n  [2] Hypothesis quality  (in unexplored territory)")
    if hyp_acc is not None:
        print(f"      Hypothesis predictions : {n_hyp}   accuracy: {hyp_acc:.3f}"
              f"  (baseline: {baseline:.3f})")
        print(f"      Standard predictions   : {n_std}   accuracy: {std_acc:.3f}")
        verdict = "above baseline ✓" if hyp_acc > baseline else "at/below baseline"
        print(f"      Verdict : {verdict}")
    else:
        print(f"      No hypothesis predictions made (all contexts were covered)")

    print(f"\n  [3] Credibility Gini  (has feedback differentiated nodes?)")
    prev_gini  = prev.get('gini')
    gini_delta = f"  [prev {prev_gini:.3f}]" if prev_gini else ""
    print(f"      Gini: {cred_gini:.3f}{gini_delta}  "
          f"({'strong differentiation' if cred_gini > 0.3 else 'moderate' if cred_gini > 0.15 else 'weak'})")

    print(f"\n  [4] Node efficiency  (topology quality)")
    print(f"      Final nodes    : {final_nodes}")
    print(f"      Nodes at 50% acc milestone : {m50 if m50 else 'not reached'}")
    print(f"      Nodes at 75% acc milestone : {m75 if m75 else 'not reached'}")


PREV = {
    "Airline Passengers":       dict(lift=37.9, gini=0.109, cal=0.843),
    "Alice in Wonderland":      dict(lift=-8.3, gini=0.154, cal=0.833),
    "Bacteriophage Lambda DNA": dict(lift=14.7, gini=0.084, cal=0.779),
    "NYC Weather Events":       dict(lift=44.0, gini=0.368, cal=0.749),
    "Python PRNG":              dict(lift=40.0, gini=0.085, cal=0.876),
}


def _datasets():
    raw = load_airline_passengers()
    airline = discretize(normalize(raw), n_bins=8)
    try:
        alice = load_gutenberg_text(n_chars=1500)
    except Exception:
        alice = None
    try:
        dna = load_dna_sequence(n_bases=1500)
    except Exception:
        dna = None
    try:
        weather = load_weather_events(n_days=500)
    except Exception:
        weather = None
    prng = random_integers(n=500, low=0, high=9)
    return [
        ("Airline Passengers",       airline,  gaussian(sigma=2.0), 4),
        ("Alice in Wonderland",      alice,    hamming,             3),
        ("Bacteriophage Lambda DNA", dna,      hamming,             4),
        ("NYC Weather Events",       weather,  hamming,             3),
        ("Python PRNG",              prng,     gaussian(sigma=1.0), 3),
    ]


def main():
    print("Architecture Test Suite\n")
    datasets = _datasets()

    for idx, (name, seq, sim_fn, ctx_len) in enumerate(datasets, 1):
        print(f"[{idx}/5] {name}...")
        if seq is None:
            print("  Failed: dataset unavailable")
            continue
        run_suite(name, seq, sim_fn, context_length=ctx_len)

    print(f"\n{'═'*62}")
    print("What good results look like:")
    print("  Calibration → > 0.7 | Gini → > 0.3")
    print("  Node eff → milestones well before final count\n")


if __name__ == "__main__":
    main()
