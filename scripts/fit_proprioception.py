"""fit_proprioception.py -- build and calibrate FluxProprioception on
real questions from FLUX's actual CoT training sources (GSM8K/OpenOrca/
Magicoder/CommitPackFT), matching the exact raw-question prompt shape
FLUX is actually given at inference (think=True path, no context
wrapper -- see uchi/proprioception.py's docstring for why this
correction mattered).

Not wired into the live pipeline yet -- this is the fit-and-calibrate
step, analogous to what verifier_train.py does for the entailment
classifier's own OOD detector, but as a standalone script since this
doesn't require any gradient-based training (FLUX's weights are frozen
throughout).

Usage:
    .venv/bin/python -m scripts.fit_proprioception
"""
import argparse
import random

import torch

from uchi.flux.cot_distill import (
    generate_synthetic_cot, generate_openorca_cot,
    generate_magicoder_cot, generate_commitpackft_cot,
)
from uchi.proprioception import (
    FluxProprioception, load_flux_for_proprioception, pooled_hidden_for_question,
)


def load_flux(checkpoint: str):
    model, tokenizer = load_flux_for_proprioception(checkpoint)
    assert model is not None, f"could not load FLUX from {checkpoint}"
    return model, tokenizer


def real_questions(n_per_source: int) -> list[str]:
    """Real questions from FLUX's actual CoT training sources, same fair
    per-source sampling discipline used everywhere else this session."""
    sources = [generate_synthetic_cot, generate_openorca_cot,
               generate_magicoder_cot, generate_commitpackft_cot]
    questions = []
    for src in sources:
        try:
            examples = src(n_per_source)
            questions.extend(ex["question"] for ex in examples if ex.get("question"))
        except Exception as e:
            print(f"  [!] {src.__name__} failed: {e}")
    return questions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="uchi/flux/checkpoints/flux_best.pt")
    ap.add_argument("--n-per-source", type=int, default=150)
    ap.add_argument("--percentile", type=float, default=95.0)
    ap.add_argument("--out", default="uchi/flux/checkpoints/proprioception.pt")
    args = ap.parse_args()

    print("Loading FLUX ...")
    model, tokenizer = load_flux(args.checkpoint)

    print(f"Sampling real questions from CoT training sources ({args.n_per_source}/source) ...")
    all_questions = real_questions(args.n_per_source)
    random.Random(42).shuffle(all_questions)
    print(f"  Total: {len(all_questions)} real questions")

    n_val = max(20, int(len(all_questions) * 0.2))
    fit_questions, held_out_questions = all_questions[n_val:], all_questions[:n_val]
    print(f"  Fit set: {len(fit_questions)}  Held-out calibration set: {len(held_out_questions)}")

    prop = FluxProprioception()
    print("Fitting on real question representations ...")
    prop.fit(model, tokenizer, fit_questions)

    print(f"Calibrating threshold at {args.percentile}th percentile of held-out in-distribution distances ...")
    threshold = prop.calibrate_threshold(model, tokenizer, held_out_questions, percentile=args.percentile)
    print(f"  Calibrated threshold: {threshold:.2f}")

    # Sanity check: genuinely OOD prompts should still separate clearly
    # at this calibrated threshold.
    genuinely_ood = [
        "这是一个关于量子物理学的复杂讨论,涉及粒子纠缠和多维空间理论。",
        "xkq739 flibbertigibbet zqor plandorf mmm vex quorlth stanthex",
        "!@#$%^&*()_+ === >>> <<< ||| [[[ ]]] {{{ }}}",
    ]
    print("\n-- held-out in-distribution (should mostly be familiar) --")
    for q in held_out_questions[:5]:
        dist = prop.detector.distance(pooled_hidden_for_question(model, tokenizer, q))
        print(f"  dist={dist:7.2f}  unfamiliar={prop.is_unfamiliar(model, tokenizer, q)}   {q[:60]!r}")
    print("-- genuinely OOD --")
    for q in genuinely_ood:
        dist = prop.detector.distance(pooled_hidden_for_question(model, tokenizer, q))
        print(f"  dist={dist:7.2f}  unfamiliar={prop.is_unfamiliar(model, tokenizer, q)}   {q[:60]!r}")

    torch.save(prop.state_dict(), args.out)
    print(f"\nSaved fitted proprioception detector -> {args.out}")


if __name__ == "__main__":
    main()
