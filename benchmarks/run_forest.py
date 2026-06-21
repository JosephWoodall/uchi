import functools

from forest import PredictorForest
from run_experiments import run, discretize, normalize
from similarity import gaussian, hamming
from datasets import (
    load_airline_passengers,
    load_gutenberg_text,
    load_dna_sequence,
    load_weather_events,
    random_integers,
)

N_TREES        = 5
DROPOUT        = 0.2
STAGGER        = 25
VOTING         = 'adaptive'
MAX_TREES      = 20
AUTO_GROW      = True
AUTO_PRUNE     = True
GROW_THRESHOLD = 8
PRUNE_WINDOW   = 50

forest_cls = functools.partial(
    PredictorForest,
    n_trees=N_TREES,
    dropout=DROPOUT,
    stagger=STAGGER,
    voting=VOTING,
    heterogeneous_k=True,
    tree_lr=0.1,
    max_trees=MAX_TREES,
    auto_grow=AUTO_GROW,
    auto_prune=AUTO_PRUNE,
    grow_threshold=GROW_THRESHOLD,
    prune_window=PRUNE_WINDOW,
)


def main() -> None:
    print("Universal Sequence Predictor — Forest Experiment Suite")
    print(f"  {N_TREES} initial trees  |  dropout {DROPOUT}  |  stagger {STAGGER}"
          f"  |  k heterogeneous  |  voting: {VOTING}")
    print(f"  Dynamic sizing: grow≤{MAX_TREES} trees  |  "
          f"auto_grow (thresh={GROW_THRESHOLD})  |  auto_prune (window={PRUNE_WINDOW})\n")

    print("[1/5] Airline passengers (numeric time series)...")
    raw = load_airline_passengers()
    seq = discretize(normalize(raw), n_bins=8)
    run("Airline Passengers", seq, gaussian(sigma=2.0), context_length=4,
        predictor_cls=forest_cls)

    print("\n[2/5] Alice in Wonderland (character-level text)...")
    try:
        seq = load_gutenberg_text(n_chars=1500)
        # Text at k=3 with 25-char alphabet: k=4+ contexts are too sparse (25^4 >> 1200 samples).
        # All trees run at k=3 so diversity comes from dropout + stagger only.
        # auto_grow disabled: spawning trees at k=4+ on this corpus would just add noise.
        forest_text_cls = functools.partial(forest_cls, heterogeneous_k=False,
                                            auto_grow=False, auto_prune=False)
        # ρ=0.7: most English trigrams are rare at 1500 chars, so high vigilance
        # is needed to trigger exploration nodes that anchor novel territory.
        run("Alice in Wonderland", seq, hamming, context_length=3, vigilance=0.7,
            predictor_cls=forest_text_cls)
    except Exception as exc:
        print(f"  Failed: {exc}")

    print("\n[3/5] Bacteriophage lambda genome (DNA)...")
    try:
        seq = load_dna_sequence(n_bases=1500)
        run("Bacteriophage Lambda DNA", seq, hamming, context_length=4,
            predictor_cls=forest_cls)
    except Exception as exc:
        print(f"  Failed: {exc}")

    print("\n[4/5] NYC daily weather codes (categorical events)...")
    try:
        seq = load_weather_events(n_days=500)
        run("NYC Weather Events", seq, hamming, context_length=3,
            predictor_cls=forest_cls)
    except Exception as exc:
        print(f"  Failed: {exc}")

    print("\n[5/5] Python PRNG (unpredictable baseline)...")
    seq = random_integers(n=500, low=0, high=9)
    run("Python PRNG (random)", seq, gaussian(sigma=1.0), context_length=3,
        predictor_cls=forest_cls)

    print(f"\n{'═'*62}")
    print(f"Initial config: {N_TREES} trees, k = base+0 … base+{N_TREES-1}")
    print(f"  Dropout {DROPOUT} | Stagger {STAGGER} steps | Voting: {VOTING}")
    print(f"  Trees grow (max {MAX_TREES}) on unanimous wrong streak ≥ {GROW_THRESHOLD}")
    print(f"  Trees prune after credibility below floor for {PRUNE_WINDOW} steps")
    print("  node_stats() n_active/n_total/n_spawned show forest evolution\n")


if __name__ == "__main__":
    main()
