"""
mcq_discrimination.py  —  Phase 0 de-risking spike
==================================================
Question: can Uchi's 256D manifold discriminate the correct MCQ answer from its
distractors on OUT-OF-DISTRIBUTION questions, using ONLY the SSM (no trie, no
generation)?

If yes (> 25% on 4-choice), the manifold has latent OOD signal and the readout
plumbing (Phase 2) is worth building. If no (~25%), the manifold is untrained for
this task and Phase 1 (distractor-contrastive training) / capacity escalation is
mandatory before any readout work.

This is intentionally minimal and offline:
  question state := ssm.get_state(["<|user|>"] + tokens(prompt) + ["<|assistant|>"])
  Scorer P (policy): argmax_L  ssm.get_policy_score(state, L)
  Scorer V (value) : argmax_L  ssm.value(get_state(... ["<|assistant|>", L])).item()

ARC-Challenge TEST split is genuinely held out: the brain ingested the ARC TRAIN
split, so test questions are unseen. Random baseline = 25%.

Usage:
    .venv/bin/python experiments/mcq_discrimination.py --sample 200
    .venv/bin/python experiments/mcq_discrimination.py --dataset mmlu --sample 200
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import random
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_LETTERS = {"A", "B", "C", "D"}


def _format_prompt(question: str, labels: list[str], texts: list[str]) -> str:
    lines = ["The following is a multiple choice question.", "", question, ""]
    for label, text in zip(labels, texts):
        lines.append(f"{label}. {text}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def _load_router(brain_path: str):
    from uchi.omni_router import OmniRouter
    if os.path.exists(brain_path):
        try:
            with gzip.open(brain_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            with open(brain_path, "rb") as f:
                return pickle.load(f)
    OmniRouter._bootstrap_knowledge = lambda self, *a, **kw: None
    OmniRouter._bootstrap_persona = lambda self, *a, **kw: None
    return OmniRouter(use_bpe=False)


def _load_rows(dataset: str, sample: int):
    from datasets import load_dataset
    rows = []
    if dataset == "arc":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        for r in ds:
            labels = r["choices"]["label"]
            if not set(labels).issubset(_LETTERS):
                continue
            rows.append((r["question"], labels, r["choices"]["text"],
                         r["answerKey"].strip().upper()))
    else:  # mmlu
        ds = load_dataset("cais/mmlu", "all", split="test")
        idx2letter = ["A", "B", "C", "D"]
        for r in ds:
            choices = r["choices"]
            if len(choices) != 4:
                continue
            labels = idx2letter
            ans = idx2letter[int(r["answer"])]
            rows.append((r["question"], labels, choices, ans))
    if sample and sample < len(rows):
        rows = random.sample(rows, sample)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["arc", "mmlu"], default="arc")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--brain", default="brain.uchi")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    import torch
    from uchi.neuro_symbolic import get_ssm

    print(f"[*] loading brain {args.brain} …")
    router = _load_router(args.brain)
    tok = router.tokenizer
    ssm = get_ssm()
    ssm.eval()

    print(f"[*] loading {args.dataset} held-out rows …")
    rows = _load_rows(args.dataset, args.sample)
    print(f"[*] {len(rows)} questions  (random baseline = 25%)\n")

    import torch.nn.functional as F

    def _tok(text: str) -> list:
        try:
            return list(tok.tokenize(text, is_inference=True))
        except Exception:
            return text.split()

    n = 0
    hit_letter = 0   # letter probe (degenerate baseline — memorized Q→letter)
    hit_cos = 0      # SEMANTIC: cos(question_state, option_text_state)
    hit_valc = 0     # value head on (question + option_text)
    from collections import Counter
    letter_pred_dist = Counter()
    cos_pred_dist = Counter()
    cos_gap_sum = 0.0   # mean (best - 2nd best) cosine — is there ANY separation?
    t0 = time.time()

    with torch.no_grad():
        for i, (question, labels, texts, answer) in enumerate(rows):
            if answer not in labels:
                continue
            prompt = _format_prompt(question, labels, texts)
            q_tokens = _tok(prompt)
            base = ["<|user|>"] + list(q_tokens) + ["<|assistant|>"]
            try:
                q_state = ssm.get_state(base)                 # (1, d)
                qv = F.normalize(q_state, p=2, dim=-1)
                # pure-question encoding (no option list) for semantic proximity
                qonly = F.normalize(ssm.get_state(_tok(question)), p=2, dim=-1)
            except Exception:
                continue

            letter_scores, cos_scores, valc_scores = {}, {}, {}
            for L, txt in zip(labels, texts):
                # letter probe
                try:
                    letter_scores[L] = ssm.get_policy_score(q_state, L)
                except Exception:
                    letter_scores[L] = float("-inf")
                # semantic: does the OPTION TEXT embed near the question?
                try:
                    ov = F.normalize(ssm.get_state(_tok(txt)), p=2, dim=-1)
                    cos_scores[L] = float((qonly * ov).sum().item())
                except Exception:
                    cos_scores[L] = float("-inf")
                # value of question followed by the answer content
                try:
                    valc_scores[L] = ssm.value(ssm.get_state(base + _tok(txt))).item()
                except Exception:
                    valc_scores[L] = float("-inf")

            pred_letter = max(letter_scores, key=letter_scores.get)
            pred_cos = max(cos_scores, key=cos_scores.get)
            pred_valc = max(valc_scores, key=valc_scores.get)

            sc = sorted(cos_scores.values(), reverse=True)
            if len(sc) >= 2:
                cos_gap_sum += sc[0] - sc[1]

            n += 1
            hit_letter += int(pred_letter == answer)
            hit_cos += int(pred_cos == answer)
            hit_valc += int(pred_valc == answer)
            letter_pred_dist[pred_letter] += 1
            cos_pred_dist[pred_cos] += 1

            if (i + 1) % 50 == 0:
                print(f"  [{i+1:>4}/{len(rows)}]  letter={hit_letter/n:.3f}  "
                      f"cos={hit_cos/n:.3f}  valc={hit_valc/n:.3f}  "
                      f"({time.time()-t0:.0f}s)")

    print("\n" + "─" * 60)
    print(f"  MCQ discrimination — {args.dataset} held-out, {n} questions")
    print("─" * 60)
    print(f"  Random baseline        : 25.0%")
    print(f"  Letter probe (policy)  : {hit_letter/n*100:.1f}%   "
          f"(pred dist: {dict(letter_pred_dist)})")
    print(f"  SEMANTIC cos(Q, optTxt): {hit_cos/n*100:.1f}%   "
          f"(pred dist: {dict(cos_pred_dist)})")
    print(f"  Value(Q + optTxt)      : {hit_valc/n*100:.1f}%")
    print(f"  mean cosine gap (1st-2nd): {cos_gap_sum/n:.4f}   "
          f"(near 0 = manifold can't separate options)")
    print("─" * 60)
    best = max(hit_cos, hit_valc) / n
    if best > 0.32:
        print("  VERDICT: manifold HAS semantic OOD signal → build Phase 2 readout.")
    elif best > 0.27:
        print("  VERDICT: weak signal → Phase 1 contrastive training to sharpen it.")
    else:
        print("  VERDICT: ~random → manifold geometry untrained for QA; Phase 1 "
              "mandatory, consider capacity escalation.")


if __name__ == "__main__":
    main()
