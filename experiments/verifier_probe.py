"""
verifier_probe.py — the make-or-break de-risk for Generate-and-Ground.

Question: can the TRIE act as a grounding oracle — assign higher "brain support"
to a grounded (correct) claim than to a plausible-but-false (distractor) one?

If yes, the architecture stands: a generator proposes, the trie vetoes hallucinated
claims. If no, "the trie keeps it honest" fails and we must rethink the oracle.

Grounding score of an answer A to question Q =
    mean over answer tokens of  log P(token | question-context)  under the trie
    (peek_distribution). Higher = the brain more strongly expects this continuation.

Two metrics on held-out MCQ (plausible distractors = plausible hallucinations):
  - pairwise: % of (correct vs one distractor) where grounded scores higher.
  - argmax:  % where correct is the single most-grounded of all options.

Usage:
    .venv/bin/python experiments/verifier_probe.py --dataset arc_easy --sample 300
"""
from __future__ import annotations
import argparse, gzip, math, os, pickle, random, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
_LETTERS = {"A", "B", "C", "D"}


def _load_router(p):
    from uchi.omni_router import OmniRouter
    if os.path.exists(p):
        try:
            with gzip.open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            with open(p, "rb") as f:
                return pickle.load(f)
    return OmniRouter(use_bpe=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="brain.uchi")
    ap.add_argument("--dataset", choices=["arc_easy", "arc_challenge"], default="arc_easy")
    ap.add_argument("--sample", type=int, default=300)
    args = ap.parse_args()
    random.seed(0)

    from datasets import load_dataset
    router = _load_router(args.brain)
    tok = router.tokenizer
    pred = router.predictor

    def toks(s):
        try:
            return list(tok.tokenize(s, is_inference=True))
        except Exception:
            return s.split()

    def ground_score(q_prompt, answer):
        """mean log P(answer token | trie context). Higher = more brain support."""
        ctx = ["<|user|>"] + toks(q_prompt) + ["<|assistant|>"]
        ans = toks(answer)
        if not ans:
            return -50.0
        lp, hist = 0.0, list(ctx)
        for t in ans:
            try:
                dist = pred.peek_distribution(hist)
            except Exception:
                dist = {}
            lp += math.log(max(dist.get(t, 1e-6), 1e-9))
            hist.append(t)
        return lp / len(ans)

    cfg = "ARC-Easy" if args.dataset == "arc_easy" else "ARC-Challenge"
    ds = load_dataset("allenai/ai2_arc", cfg, split="test")
    rows = []
    for r in ds:
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        ans = r["answerKey"].strip().upper()
        if set(labels).issubset(_LETTERS) and ans in labels:
            correct = texts[labels.index(ans)]
            distr = [t for l, t in zip(labels, texts) if l != ans]
            rows.append((r["question"], correct, distr))
    random.shuffle(rows); rows = rows[:args.sample]
    print(f"[*] {cfg}: {len(rows)} questions")

    def fmt(q):
        return "The following is a multiple choice question.\n\n" + q + "\n\nAnswer:"

    pair_hit = pair_n = arg_hit = arg_n = 0
    for q, correct, distr in rows:
        qp = fmt(q)
        sc = ground_score(qp, correct)
        sds = [ground_score(qp, d) for d in distr]
        if sds:
            pair_n += 1
            pair_hit += int(sc > sds[0])       # vs first distractor
            arg_n += 1
            arg_hit += int(sc > max(sds))      # vs all distractors

    print("\n" + "─" * 56)
    print(f"  TRIE GROUNDING ORACLE — {cfg}")
    print(f"  pairwise (grounded > 1 distractor) : {pair_hit/max(pair_n,1)*100:5.1f}%   (chance 50%)")
    print(f"  argmax  (grounded = most grounded) : {arg_hit/max(arg_n,1)*100:5.1f}%   (chance 25%)")
    print("─" * 56)
    if pair_hit / max(pair_n, 1) > 0.65:
        print("  VERDICT: trie separates grounded from fabricated → oracle viable.")
    else:
        print("  VERDICT: trie CANNOT reliably ground → rethink the oracle (retrieval/entailment).")


if __name__ == "__main__":
    main()
