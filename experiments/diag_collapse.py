"""
diag_collapse.py — isolate capacity/collapse vs optimization for QA discrimination.

Two probes, ~1 minute, offline:

  1. ANISOTROPY: mean pairwise cosine between embeddings of unrelated texts.
     ~0 = healthy spread. →1 = representational collapse (everything is the same
     point, so InfoNCE physically cannot separate correct from distractor).

  2. OVERFIT TEST: take 16 questions, train qa_contrastive ONLY (qa-dominant,
     high lr) for N steps on those SAME 16. If qa-loss → 0 and the cosine gap
     opens, capacity is FINE and the failure was optimization/weighting →
     contrastive-dominant run will work. If qa-loss stays at ln(4)=1.386 even
     when memorizing 16 examples, it is capacity/collapse → escalate.
"""
from __future__ import annotations
import gzip, os, pickle, random, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
_LETTERS = {"A", "B", "C", "D"}


def _load_router(p):
    from uchi.omni_router import OmniRouter
    if os.path.exists(p):
        try:
            with gzip.open(p, "rb") as f: return pickle.load(f)
        except Exception:
            with open(p, "rb") as f: return pickle.load(f)
    return OmniRouter(use_bpe=False)


def main():
    import math, torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from uchi.neuro_symbolic import get_ssm
    random.seed(0)

    router = _load_router("brain.uchi")
    tok = router.tokenizer
    def T(s):
        try: return list(tok.tokenize(s, is_inference=True))
        except Exception: return s.split()

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    rows = []
    for r in ds:
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        ans = r["answerKey"].strip().upper()
        if set(labels).issubset(_LETTERS) and ans in labels:
            correct = texts[labels.index(ans)]
            distr = [t for l, t in zip(labels, texts) if l != ans]
            rows.append((r["question"], correct, distr))
        if len(rows) >= 16: break

    ssm = get_ssm(); ssm.eval()

    # ── Probe 1: anisotropy ──────────────────────────────────────────────
    with torch.no_grad():
        vecs = []
        for q, c, d in rows:
            for txt in [q, c] + d:
                vecs.append(F.normalize(ssm.get_state(T(txt)), p=2, dim=-1))
        V = torch.cat(vecs, dim=0)              # (M, d)
        M = V.shape[0]
        sims = (V @ V.T)
        off = sims[~torch.eye(M, dtype=torch.bool)]
        print(f"\nProbe 1 — ANISOTROPY over {M} text embeddings")
        print(f"  mean pairwise cosine : {off.mean().item():.4f}  "
              f"(→1.0 = collapsed)")
        print(f"  std  pairwise cosine : {off.std().item():.4f}")
        print(f"  min / max            : {off.min().item():.4f} / {off.max().item():.4f}")

    # ── Probe 2: can we OVERFIT 16 examples with qa-dominant high-lr? ─────
    print("\nProbe 2 — OVERFIT 16 examples (qa-dominant, lr=1e-3, 150 steps)")
    print(f"  random 4-way loss = ln(4) = {math.log(4):.4f}")
    ssm.train()
    opt = torch.optim.Adam(ssm.parameters(), lr=1e-3)
    for step in range(150):
        opt.zero_grad()
        tot = 0.0
        for q, c, d in rows:
            loss = ssm.qa_contrastive_loss(T(q), T(c), [T(x) for x in d])
            loss.backward(); tot += float(loss.item())
        torch.nn.utils.clip_grad_norm_(ssm.parameters(), 1.0)
        opt.step()
        if step % 30 == 0 or step == 149:
            print(f"  step {step:>3}  mean qa-loss = {tot/len(rows):.4f}")

    # gap on the trained 16
    ssm.eval()
    with torch.no_grad():
        gaps, correct_hits = [], 0
        for q, c, d in rows:
            qv = F.normalize(ssm.get_state(T(q)), p=2, dim=-1)
            cands = [(c, True)] + [(x, False) for x in d]
            scored = []
            for txt, is_c in cands:
                ov = F.normalize(ssm.get_state(T(txt)), p=2, dim=-1)
                scored.append((float((qv*ov).sum()), is_c))
            scored.sort(reverse=True)
            correct_hits += int(scored[0][1])
            gaps.append(scored[0][0] - scored[1][0])
        print(f"  after overfit: train acc={correct_hits}/{len(rows)}  "
              f"mean cosine gap={sum(gaps)/len(gaps):.4f}")
    print("\nVERDICT:")
    print("  qa-loss → ~0 & acc=16/16  → capacity FINE, failure was weighting → contrastive-dominant run.")
    print("  qa-loss stuck ~1.386       → COLLAPSE/capacity → escalate encoder.")


if __name__ == "__main__":
    main()
