"""verifier_adversarial_validation.py -- held-out adversarial validation
for EntailmentClassifier (0.4.0 Item 17, exit criterion #7). Hard gate:
nothing downstream trusts the verifier live until this passes.

Deliberately does NOT reuse the Eiffel Tower / Building A-B-C examples
used throughout this project's demos and discussions -- testing on the
motivating examples themselves would prove nothing. Every case here uses
different entities/topics, constructed independently of training data or
the synthetic multi-hop generator's own entity/relation pools.

Three categories, matching what Item 17 was actually built to catch:
1. Numeric substitution (the "330m vs 500m" class -- same evidence,
   correct number vs a wrong one, high word-overlap either way)
2. Semantic negation (opposite meaning, historically missed by pure
   word-overlap)
3. Held-out multi-hop transitive chains (new entities/relations, not from
   generate_multihop_examples()'s own template pools)

Usage:
    .venv/bin/python -m scripts.verifier_adversarial_validation \
        --checkpoint uchi/flux/checkpoints/verifier/verifier_best.pt
"""
import argparse

from uchi.flux.verifier_model import EntailmentChecker

# (premise/evidence, claim, expected: True=should flag as contradiction, False=should NOT flag)
CASES = [
    # --- Category 1: numeric substitution ---
    ("The Golden Gate Bridge spans 2,737 meters and was completed in 1937.",
     "The Golden Gate Bridge spans 2,737 meters.", False),
    ("The Golden Gate Bridge spans 2,737 meters and was completed in 1937.",
     "The Golden Gate Bridge spans 4,200 meters.", True),
    ("The population of Iceland is approximately 370,000 people.",
     "The population of Iceland is approximately 370,000 people.", False),
    ("The population of Iceland is approximately 370,000 people.",
     "The population of Iceland is approximately 9 million people.", True),
    ("The battery lasts 14 hours on a full charge according to the manual.",
     "The battery lasts 14 hours on a full charge.", False),
    ("The battery lasts 14 hours on a full charge according to the manual.",
     "The battery lasts 40 hours on a full charge.", True),

    # --- Category 2: semantic negation (not just number swaps) ---
    ("The board of directors voted to approve the merger unanimously.",
     "The board of directors approved the merger.", False),
    ("The board of directors voted to approve the merger unanimously.",
     "The board of directors rejected the merger.", True),
    ("The new policy increased funding for public schools by 12 percent.",
     "The new policy increased school funding.", False),
    ("The new policy increased funding for public schools by 12 percent.",
     "The new policy cut school funding.", True),
    ("Clinical trials showed the drug significantly reduced symptoms in patients.",
     "The drug reduced symptoms in the trial.", False),
    ("Clinical trials showed the drug significantly reduced symptoms in patients.",
     "The drug had no effect on symptoms in the trial.", True),

    # --- Category 3: held-out multi-hop transitive chains (new entities/
    # relations vs. generate_multihop_examples()'s own pools) ---
    ("The Andes mountain range is longer than the Rockies. The Rockies are longer than the Alps.",
     "The Andes are longer than the Alps.", False),
    ("The Andes mountain range is longer than the Rockies. The Rockies are longer than the Alps.",
     "The Alps are longer than the Andes.", True),
    ("Comet A orbits the sun faster than Comet B. Comet B orbits faster than Comet C.",
     "Comet A orbits faster than Comet C.", False),
    ("Comet A orbits the sun faster than Comet B. Comet B orbits faster than Comet C.",
     "Comet C orbits faster than Comet A.", True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="uchi/flux/checkpoints/verifier/verifier_best.pt")
    args = ap.parse_args()

    checker = EntailmentChecker.load(args.checkpoint)
    assert checker is not None, f"verifier checkpoint failed to load from {args.checkpoint}"

    correct = 0
    results = []
    for premise, claim, expected in CASES:
        got = checker.is_contradiction(premise, claim)
        ok = (got == expected)
        correct += ok
        results.append((premise, claim, expected, got, ok))

    print(f"{'EXPECTED':9s} {'GOT':6s} {'OK':3s}  CLAIM")
    print("-" * 90)
    for premise, claim, expected, got, ok in results:
        print(f"{str(expected):9s} {str(got):6s} {'v' if ok else 'x':3s}  {claim!r}")

    print()
    print(f"Accuracy: {correct}/{len(CASES)} = {correct/len(CASES):.1%}")
    contra_cases = [r for r in results if r[2] is True]
    contra_caught = sum(1 for r in contra_cases if r[4])
    print(f"Real contradiction cases caught: {contra_caught}/{len(contra_cases)} "
          f"(this is the number that actually matters -- overall accuracy can look "
          f"fine even if every real contradiction is missed, if the classifier just "
          f"never predicts contradiction at all)")


if __name__ == "__main__":
    main()
