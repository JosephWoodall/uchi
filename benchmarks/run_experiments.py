import statistics

from predictor import UniversalPredictor
from similarity import gaussian, hamming
from datasets import (
    load_airline_passengers,
    load_gutenberg_text,
    load_dna_sequence,
    load_weather_events,
    random_integers,
)


def discretize(seq: list[float], n_bins: int = 8) -> list[int]:
    lo, hi = min(seq), max(seq)
    width  = (hi - lo) / n_bins
    return [min(int((v - lo) / width), n_bins - 1) for v in seq]


def normalize(seq: list[float]) -> list[float]:
    mu    = statistics.mean(seq)
    sigma = statistics.stdev(seq) or 1.0
    return [(v - mu) / sigma for v in seq]


def run(name: str, seq: list, sim_fn, context_length: int = 3,
        learning_rate: float = 0.1, vigilance: float = 0.3,
        predictor_cls=None) -> None:
    if predictor_cls is None:
        predictor_cls = UniversalPredictor
    n       = len(seq)
    train_n = int(n * 0.8)

    predictor = predictor_cls(
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

    correct = total = 0
    conf_total = 0.0

    for v in seq[train_n:]:
        pred, conf = predictor.predict()
        predictor.observe(v)
        predictor.feedback(v)
        if pred is not None:
            correct    += int(pred == v)
            conf_total += conf
            total      += 1

    unique   = len(set(seq))
    baseline = 1.0 / unique
    accuracy = correct / total if total > 0 else 0.0
    avg_conf = conf_total / total if total > 0 else 0.0
    lift     = (accuracy / baseline - 1.0) * 100.0 if baseline > 0 else 0.0
    ns       = predictor.node_stats()

    print(f"\n{'─'*62}")
    print(f"  {name}")
    print(f"  Obs: {n}  |  Unique: {unique}  |  k={context_length}")
    print(f"  Baseline : {baseline:.3f}")
    print(f"  Accuracy : {accuracy:.3f}   Lift: {lift:+.1f}%")
    print(f"  Avg conf : {avg_conf:.3f}   Nodes: {ns['total_nodes']}")


def main() -> None:
    print("Universal Sequence Predictor — Experiment Suite\n")

    print("[1/5] Airline passengers...")
    raw = load_airline_passengers()
    seq = discretize(normalize(raw), n_bins=8)
    run("Airline Passengers", seq, gaussian(sigma=2.0), context_length=4)

    print("\n[2/5] Alice in Wonderland...")
    try:
        seq = load_gutenberg_text(n_chars=1500)
        run("Alice in Wonderland", seq, hamming, context_length=3)
    except Exception as exc:
        print(f"  Failed: {exc}")

    print("\n[3/5] Bacteriophage lambda genome (DNA)...")
    try:
        seq = load_dna_sequence(n_bases=1500)
        run("Bacteriophage Lambda DNA", seq, hamming, context_length=4)
    except Exception as exc:
        print(f"  Failed: {exc}")

    print("\n[4/5] NYC daily weather codes...")
    try:
        seq = load_weather_events(n_days=500)
        run("NYC Weather Events", seq, hamming, context_length=3)
    except Exception as exc:
        print(f"  Failed: {exc}")

    print("\n[5/5] Python PRNG (noise floor)...")
    seq = random_integers(n=500, low=0, high=9)
    run("Python PRNG (random)", seq, gaussian(sigma=1.0), context_length=3)

    print(f"\n{'═'*62}\n")


if __name__ == "__main__":
    main()
