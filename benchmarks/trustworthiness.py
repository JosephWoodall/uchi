"""
trustworthiness.py — the benchmark suite for Uchi's Generate-and-Ground architecture.

The OLD benchmarks (MMLU/ARC/SWE accuracy) measured LLM-style raw reasoning — the
axis a no-LLM system loses on. This suite measures what Uchi is actually FOR:
TRUSTWORTHINESS. Not "how smart," but: when it answers, is it right; does it know
what it doesn't know; does it ever confabulate.

Headline: SQuAD 2.0 (answerable + unanswerable questions). We index the contexts
as the brain's knowledge, then for each question run the full loop and record:

  KPIs
  ----
  coverage            % of ANSWERABLE questions the system chose to answer
  precision@answered  % of answered questions that are correct (the trust metric)
  honest-abstention   % of UNANSWERABLE questions correctly abstained on
  hallucination-rate  % of answers emitted that are wrong (target: near zero)

A trustworthy assistant maximises precision@answered and honest-abstention while
keeping hallucination-rate near zero — it would rather stay silent than lie.

Usage:
    .venv/bin/python benchmarks/trustworthiness.py --sample 800
"""
from __future__ import annotations
import argparse, os, re, sys, json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from uchi.retrieval import SemanticIndex           # noqa: E402
from uchi.oracle import FactCheckOracle             # noqa: E402
from uchi.generate_and_ground import GenerateAndGround  # noqa: E402

_ABSTAIN_MARK = "don't have grounded knowledge"
_EMB = os.path.join(os.path.dirname(__file__), "..", "uchi", "data", "skipgram_emb.pt")
_DECODER = os.path.join(os.path.dirname(__file__), "..", "uchi", "data", "decoder.pt")


def _correct(answer: str, golds: list[str]) -> bool:
    a = answer.lower()
    return any(g.strip() and g.lower() in a for g in golds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=800)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "trustworthiness_results.json"))
    args = ap.parse_args()

    from datasets import load_dataset
    print("[*] loading SQuAD 2.0 …")
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    import random
    random.seed(0)
    idx_rows = list(range(len(ds)))
    random.shuffle(idx_rows)
    rows = [ds[i] for i in idx_rows[:args.sample]]

    # Build the brain's knowledge index from the (unique) contexts.
    print("[*] building retrieval index from contexts …")
    index = SemanticIndex.from_embeddings_file(_EMB)
    seen = set()
    for r in rows:
        c = r["context"]
        if c not in seen:
            seen.add(c)
            index.build_from_corpus(c)
    print(f"[*] indexed {len(index):,} passages from {len(seen)} contexts")

    decoder = None
    try:
        from uchi.decoder import NeuralDecoder
        if NeuralDecoder.exists(_DECODER):
            decoder = NeuralDecoder.load(_DECODER)
            print("[*] neural decoder loaded")
    except Exception as e:
        print(f"[!] decoder load skipped: {e}")

    answerability = None
    try:
        from uchi.answerability import AnswerabilityChecker
        ap = os.path.join(os.path.dirname(__file__), "..", "uchi", "data", "answerability.pt")
        if AnswerabilityChecker.exists(ap):
            answerability = AnswerabilityChecker.load(ap)
            print("[*] answerability checker loaded")
    except Exception as e:
        print(f"[!] answerability load skipped: {e}")

    gg = GenerateAndGround(index, oracle=FactCheckOracle(), decoder=decoder,
                           answerability=answerability, min_sim=0.35, min_answerable=0.5)

    # One pass: record (answerable, correct, answerability_prob, base_abstained),
    # then sweep the answerability threshold to trace the trustworthiness curve.
    gg.answerability = None      # disable the gate in the loop; we sweep post-hoc
    recs = []
    for i, r in enumerate(rows):
        golds = r["answers"]["text"]
        answerable = len(golds) > 0
        q = r["question"]
        ev = index.retrieve(q, gg.retrieve_k)
        ans_prob = (answerability.prob(q, ev[0][0]) if (answerability and ev) else 1.0)
        reply = gg.answer(q)
        base_abstained = _ABSTAIN_MARK in reply
        correct = (not base_abstained) and answerable and _correct(reply, golds)
        recs.append((answerable, correct, ans_prob, base_abstained))
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(rows)}] processed")

    def kpis(th):
        ans_t = ans_a = ans_c = un_t = un_ab = emit = emit_wrong = 0
        for answerable, correct, ap, base_ab in recs:
            emitted = (not base_ab) and (ap >= th)
            if answerable:
                ans_t += 1
                if emitted:
                    ans_a += 1; emit += 1; ans_c += int(correct); emit_wrong += int(not correct)
            else:
                un_t += 1
                if emitted:
                    emit += 1; emit_wrong += 1
                else:
                    un_ab += 1
        return {
            "min_answerable": th,
            "coverage": round(ans_a / max(ans_t, 1), 4),
            "precision_at_answered": round(ans_c / max(ans_a, 1), 4),
            "honest_abstention": round(un_ab / max(un_t, 1), 4),
            "hallucination_rate": round(emit_wrong / max(emit, 1), 4),
        }

    ans_t = sum(1 for a, *_ in recs if a); un_t = len(recs) - ans_t
    W = 78
    print("\n" + "─" * W)
    print(f"  UCHI TRUSTWORTHINESS — SQuAD 2.0  ({len(recs)}q: {ans_t} answerable / {un_t} unanswerable)")
    print("─" * W)
    print(f"  {'threshold':>10} {'coverage':>10} {'precision':>10} {'honest-abst':>12} {'HALLUC':>9}")
    curve = [kpis(th) for th in (0.0, 0.5, 0.7, 0.85, 0.95)]
    for k in curve:
        print(f"  {k['min_answerable']:>10.2f} {k['coverage']*100:>9.1f}% {k['precision_at_answered']*100:>9.1f}% "
              f"{k['honest_abstention']*100:>11.1f}% {k['hallucination_rate']*100:>8.1f}%")
    print("─" * W)
    print("  threshold 0.0 = no answerability gate (word-overlap oracle only).")
    print("  Higher threshold → more abstention → lower hallucination, less coverage.")

    with open(args.out, "w") as f:
        json.dump({"n": len(recs), "answerable": ans_t, "unanswerable": un_t,
                   "indexed_passages": len(index), "curve": curve}, f, indent=2)
    print(f"  results → {args.out}")


if __name__ == "__main__":
    main()
