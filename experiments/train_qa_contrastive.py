"""
train_qa_contrastive.py  —  Phase 1: bind questions to their correct answers
============================================================================
Additive, conservative fine-tune of the SSM. Loads the existing GRPO-trained
ssm_dynamics.pt and continues training with:

    loss = compute_loss(question + correct_answer, reward=1.0)   # regularizers
         + W * qa_contrastive_loss(question, correct, distractors)  # NEW

The trie is NEVER touched (recall is safe). Trains the exact geometry the
discrimination spike measures: cos(question, correct) > cos(question, distractor).

Data: ARC-Challenge TRAIN + MMLU auxiliary_train (held-IN; test splits are never
used here, so the spike on the test split remains a true OOD measurement).

The previous checkpoint is backed up to ssm_dynamics.pre_qa.pt before saving.

Usage:
    .venv/bin/python experiments/train_qa_contrastive.py --limit 2000 --epochs 2
    .venv/bin/python experiments/train_qa_contrastive.py --weight 0.5 --batch 8
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_LETTERS = {"A", "B", "C", "D"}
_CKPT = "ssm_dynamics.pt"
_BACKUP = "ssm_dynamics.pre_qa.pt"


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


def _load_tuples(limit: int):
    """Return list of (question, correct_text, [distractor_texts])."""
    from datasets import load_dataset
    tuples = []

    # ARC-Challenge train
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        for r in ds:
            labels, texts = r["choices"]["label"], r["choices"]["text"]
            ans = r["answerKey"].strip().upper()
            if not set(labels).issubset(_LETTERS) or ans not in labels:
                continue
            correct = texts[labels.index(ans)]
            distractors = [t for l, t in zip(labels, texts) if l != ans]
            if correct and distractors:
                tuples.append((r["question"], correct, distractors))
    except Exception as e:
        print(f"  [!] ARC train load failed: {e}")

    # MMLU auxiliary_train — each row is nested under a "train" key.
    try:
        ds = load_dataset("cais/mmlu", "auxiliary_train", split="train")
        for row in ds:
            r = row.get("train", row) if isinstance(row, dict) else row
            choices = r["choices"]
            if len(choices) != 4:
                continue
            ai = int(r["answer"])
            correct = choices[ai]
            distractors = [c for j, c in enumerate(choices) if j != ai]
            if r.get("question") and correct and distractors:
                tuples.append((r["question"], correct, distractors))
    except Exception as e:
        print(f"  [!] MMLU auxiliary_train load failed: {e}")

    random.shuffle(tuples)
    if limit and limit < len(tuples):
        tuples = tuples[:limit]
    return tuples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="brain.uchi")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--mode", choices=["additive", "dominant"], default="dominant",
                    help="dominant: qa leads, reg = dynamics-only (no collapse-causing geo)")
    ap.add_argument("--reg_weight", type=float, default=0.05,
                    help="weight on the dynamics regularizer in dominant mode")
    ap.add_argument("--weight", type=float, default=0.5, help="qa weight in additive mode")
    ap.add_argument("--val_frac", type=float, default=0.1,
                    help="held-out fraction to monitor generalization vs memorization")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    import torch
    from uchi.neuro_symbolic import get_ssm

    print(f"[*] loading brain {args.brain} (tokenizer) …")
    router = _load_router(args.brain)
    tok = router.tokenizer

    def _t(text: str) -> list:
        try:
            return list(tok.tokenize(text, is_inference=True))
        except Exception:
            return text.split()

    print("[*] loading QA tuples (ARC train + MMLU aux) …")
    tuples = _load_tuples(args.limit)
    print(f"[*] {len(tuples)} (question, correct, distractors) tuples")
    if len(tuples) < 20:
        print("[!] not enough data — aborting"); return

    # train / val split — val monitors generalization vs memorization
    n_val = max(20, int(len(tuples) * args.val_frac))
    val_tuples = tuples[:n_val]
    train_tuples = tuples[n_val:]
    print(f"[*] mode={args.mode}  lr={args.lr}  train={len(train_tuples)}  val={len(val_tuples)}")

    import torch.nn.functional as F

    def _val_eval() -> tuple:
        """Held-out qa-loss + cosine-gap + acc — divergence from train = memorization."""
        ssm.eval()
        ql, gap, hit, n = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for q, c, d in val_tuples:
                q_t, c_t = _t(q), _t(c)
                d_t = [_t(x) for x in d]
                if not q_t or not c_t:
                    continue
                ql += float(ssm.qa_contrastive_loss(q_t, c_t, d_t).item())
                qv = F.normalize(ssm.get_state(q_t), p=2, dim=-1)
                scored = []
                for txt, is_c in [(c, True)] + [(x, False) for x in d]:
                    ov = F.normalize(ssm.get_state(_t(txt)), p=2, dim=-1)
                    scored.append((float((qv * ov).sum()), is_c))
                scored.sort(reverse=True)
                hit += int(scored[0][1]); gap += scored[0][0] - scored[1][0]; n += 1
        ssm.train()
        n = max(n, 1)
        return ql / n, gap / n, hit / n

    ssm = get_ssm()
    ssm.train()
    if os.path.exists(_CKPT) and not os.path.exists(_BACKUP):
        shutil.copy(_CKPT, _BACKUP)
        print(f"[*] backed up {_CKPT} → {_BACKUP}")

    opt = torch.optim.Adam(ssm.parameters(), lr=args.lr)
    vq, vgap, vacc = _val_eval()
    print(f"[*] PRE-TRAIN val: qa={vq:.4f}  cos_gap={vgap:.4f}  acc={vacc:.3f}")

    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        random.shuffle(train_tuples)
        opt.zero_grad()
        run_qa, run_reg, seen = 0.0, 0.0, 0
        for idx, (q, correct, distractors) in enumerate(train_tuples):
            q_t, c_t = _t(q), _t(correct)
            d_t = [_t(d) for d in distractors]
            if len(q_t) < 1 or len(c_t) < 1:
                continue

            qa = ssm.qa_contrastive_loss(q_t, c_t, d_t)
            if args.mode == "dominant":
                # dynamics-only reg (reward=None ⇒ no value/policy/geo) keeps
                # predict_next sane WITHOUT the collapse-causing geometric InfoNCE.
                reg = ssm.train_dynamics(q_t + c_t)
                loss = qa + args.reg_weight * reg
            else:
                reg = ssm.compute_loss(q_t + c_t, reward=1.0)
                loss = reg + args.weight * qa
            (loss / args.batch).backward()

            run_qa += float(qa.item()); run_reg += float(reg.item()); seen += 1
            step += 1
            if step % args.batch == 0:
                torch.nn.utils.clip_grad_norm_(ssm.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            if seen % 500 == 0:
                print(f"  epoch {epoch+1}  [{idx+1}/{len(train_tuples)}]  "
                      f"train_qa={run_qa/seen:.4f}  reg={run_reg/seen:.4f}  "
                      f"({time.time()-t0:.0f}s)")

        torch.nn.utils.clip_grad_norm_(ssm.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        vq, vgap, vacc = _val_eval()
        print(f"[*] epoch {epoch+1} done  train_qa={run_qa/max(seen,1):.4f}  "
              f"|  VAL qa={vq:.4f}  cos_gap={vgap:.4f}  acc={vacc:.3f}")
        print(f"    (memorization check: train_qa≪val_qa ⇒ overfitting)")

    ssm.eval()
    torch.save(ssm.state_dict(), _CKPT)
    print(f"\n[*] saved fine-tuned SSM → {_CKPT}  ({time.time()-t0:.0f}s total)")
    print(f"[*] revert with:  cp {_BACKUP} {_CKPT}")
    print("[*] now re-run:  .venv/bin/python experiments/mcq_discrimination.py --sample 200")


if __name__ == "__main__":
    main()
