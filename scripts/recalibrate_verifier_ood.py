"""recalibrate_verifier_ood.py -- refit + properly calibrate the verifier's
OOD detector against a SPECIFIC checkpoint, using real held-out validation
data, not training data and not the reused default threshold.

Why this exists (found directly while inspecting verifier_train.py, not
assumed): the training script's automatic end-of-run OOD fit has two real
problems for whichever checkpoint actually gets promoted:
  1. It fits on a *training* sample (data leakage into the "is this normal"
     estimate), not held-out data.
  2. It calls OODDetector() with no threshold override, which defaults to
     3.0 -- the same value already confirmed wrong on the first attempt
     (real distances were 15.5-18.1 against that threshold).
  3. It fits on whatever model weights are in memory at the very end of
     training (the last epoch), which is NOT necessarily the same weights
     as whichever checkpoint (by epoch) is actually chosen for promotion --
     confirmed the two can be out of sync when a later epoch's val loss
     doesn't win the "best" slot.

This script fits and calibrates fresh, against the exact checkpoint you're
about to promote, using the same held-out validation split the training
script itself carved out (same loader, same seed, so it's the identical
30K held-out examples at eval_frac=0.05 of 600K) -- but split further so the
Gaussian is fit on one half and the threshold is calibrated by percentile on
the other, so the same examples aren't used for both.

Usage:
    .venv/bin/python -m scripts.recalibrate_verifier_ood \
        --checkpoint uchi/flux/checkpoints/verifier/verifier_epoch5.pt \
        --percentile 95
"""
import argparse
import torch

from uchi.flux.verifier_model import EntailmentClassifier, OODDetector
from uchi.flux.verifier_train import load_verifier_examples, make_batches, DEFAULTS
from uchi.flux.tokenizer_v2 import TikTokenHybridTokenizer


def load_raw_classifier(checkpoint: str, device: str):
    obj = torch.load(checkpoint, map_location=device, weights_only=True)
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    emb = sd["embedding.weight"]
    vocab_size, d_model = int(emb.shape[0]), int(emb.shape[1])
    import re as _re
    layer_ids = {int(m.group(1)) for k in sd if (m := _re.match(r"layers\.(\d+)\.", k))}
    n_layers = (max(layer_ids) + 1) if layer_ids else 8
    for d_state in (32, 16, 64, 128):
        try:
            m = EntailmentClassifier(vocab_size=vocab_size, d_model=d_model,
                                      n_layers=n_layers, d_state=d_state).to(device)
            m.load_state_dict(sd, strict=False)
            return m
        except Exception:
            continue
    raise RuntimeError(f"could not load classifier from {checkpoint}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max-examples", type=int, default=600000)
    ap.add_argument("--seq-len", type=int, default=192)
    ap.add_argument("--percentile", type=float, default=95.0)
    ap.add_argument("--out", default=None,
                     help="output .ood.pt path; defaults to <checkpoint>.ood.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = TikTokenHybridTokenizer()

    print(f"Loading classifier from {args.checkpoint} ...")
    model = load_raw_classifier(args.checkpoint, device)
    model.eval()

    print("Reproducing the exact held-out validation split verifier_train.py used ...")
    data = load_verifier_examples(tokenizer, args.seq_len, args.max_examples)
    n_val = max(10, int(len(data) * DEFAULTS["eval_frac"]))
    val_data = data[:n_val]
    print(f"  Held-out validation size: {len(val_data):,}")

    half = len(val_data) // 2
    fit_data, calib_data = val_data[:half], val_data[half:]
    print(f"  Fit split: {len(fit_data):,}  Calibration split: {len(calib_data):,}")

    def pooled_reps(examples):
        reps = []
        with torch.no_grad():
            for X, mask, _ in make_batches(examples, 32, shuffle=False):
                X, mask = X.to(device), mask.to(device)
                reps.append(model.encode(X, mask).cpu())
        return torch.cat(reps, dim=0) if reps else torch.empty(0)

    print("Fitting OOD Gaussian on held-out fit split ...")
    fit_reps = pooled_reps(fit_data)
    ood = OODDetector()
    ood.fit(fit_reps)

    print(f"Calibrating threshold at {args.percentile}th percentile of held-out calibration-split distances ...")
    calib_reps = pooled_reps(calib_data)
    distances = sorted(ood.distance(v) for v in calib_reps)
    idx = min(len(distances) - 1, int(len(distances) * args.percentile / 100.0))
    ood.threshold = distances[idx]
    print(f"  Calibrated threshold: {ood.threshold:.2f}  "
          f"(min={distances[0]:.2f}, median={distances[len(distances)//2]:.2f}, max={distances[-1]:.2f})")

    out_path = args.out or (args.checkpoint.rsplit(".", 1)[0] + ".ood.pt")
    torch.save(ood.state_dict(), out_path)
    print(f"Saved recalibrated OOD detector -> {out_path}")


if __name__ == "__main__":
    main()
