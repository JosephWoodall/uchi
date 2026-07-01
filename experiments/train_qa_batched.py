"""
train_qa_batched.py — Phase 1 fix: break the 0.98 collapse with in-batch
negatives + an explicit uniformity term (Wang & Isola 2020).

The per-example contrastive run stalled at ln(4): pure alignment pressure (pull
positives) with only 3 negatives and a small batch could not provide the
*uniformity* pressure needed to escape the collapsed (anisotropy 0.98) basin.

This trainer:
  - encodes a BATCH of B questions + their correct answers + all distractors
  - InfoNCE with IN-BATCH negatives: for question i, negatives = every other
    question's correct answer (B-1 of them) PLUS every distractor in the batch
    (B*k hard/semi-hard negatives). 30-100x more separation pressure.
  - adds L_uniformity = log E_{i≠j} exp(-2||x_i - x_j||^2) over all embeddings,
    which directly pushes the collapsed cone apart.

loss = L_contrastive + lambda_u * L_uniformity + reg_w * dynamics_reg

Held-out val (contrastive loss, cosine gap, accuracy) is reported each epoch so
memorization (train≪val) is visible. Backs up ssm_dynamics.pt → .pre_qa.pt.

Usage (fast signal first):
    .venv/bin/python experiments/train_qa_batched.py --limit 800 --epochs 4
    .venv/bin/python experiments/train_qa_batched.py --limit 4000 --epochs 4 --batch 48
"""
from __future__ import annotations

import argparse, gzip, math, os, pickle, random, shutil, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_LETTERS = {"A", "B", "C", "D"}
_CKPT, _BACKUP = "ssm_dynamics.pt", "ssm_dynamics.pre_qa.pt"
_TAU = 0.1


def _load_router(p):
    from uchi.omni_router import OmniRouter
    if os.path.exists(p):
        try:
            with gzip.open(p, "rb") as f: return pickle.load(f)
        except Exception:
            with open(p, "rb") as f: return pickle.load(f)
    return OmniRouter(use_bpe=False)


def _load_tuples(limit):
    from datasets import load_dataset
    out = []
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        for r in ds:
            labels, texts = r["choices"]["label"], r["choices"]["text"]
            ans = r["answerKey"].strip().upper()
            if set(labels).issubset(_LETTERS) and ans in labels:
                out.append((r["question"], texts[labels.index(ans)],
                            [t for l, t in zip(labels, texts) if l != ans]))
    except Exception as e:
        print(f"  [!] ARC load failed: {e}")
    try:
        ds = load_dataset("cais/mmlu", "auxiliary_train", split="train")
        for row in ds:
            r = row.get("train", row) if isinstance(row, dict) else row
            ch = r["choices"]
            if len(ch) != 4: continue
            ai = int(r["answer"])
            if r.get("question"):
                out.append((r["question"], ch[ai],
                            [c for j, c in enumerate(ch) if j != ai]))
    except Exception as e:
        print(f"  [!] MMLU aux load failed: {e}")
    random.shuffle(out)
    return out[:limit] if limit and limit < len(out) else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="brain.uchi")
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_u", type=float, default=0.5, help="uniformity weight")
    ap.add_argument("--reg_w", type=float, default=0.02, help="dynamics reg weight")
    ap.add_argument("--val_frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    import torch
    import torch.nn.functional as F
    from uchi.neuro_symbolic import get_ssm

    router = _load_router(args.brain)
    tok = router.tokenizer
    def T(s):
        try: return list(tok.tokenize(s, is_inference=True))
        except Exception: return s.split()

    print("[*] loading tuples …")
    tuples = _load_tuples(args.limit)
    if len(tuples) < 40:
        print("[!] not enough data"); return
    n_val = max(40, int(len(tuples) * args.val_frac))
    val, train = tuples[:n_val], tuples[n_val:]
    print(f"[*] train={len(train)}  val={len(val)}  batch={args.batch}  lr={args.lr}  λ_u={args.lambda_u}")

    ssm = get_ssm(); ssm.train()
    if os.path.exists(_CKPT) and not os.path.exists(_BACKUP):
        shutil.copy(_CKPT, _BACKUP); print(f"[*] backed up → {_BACKUP}")

    def enc(s):
        return F.normalize(ssm.get_state(T(s)), p=2, dim=-1)  # (1,d)

    def contrastive(batch):
        """In-batch + hard-distractor InfoNCE + uniformity over a batch."""
        qs = torch.cat([enc(q) for q, _, _ in batch], 0)          # (B,d)
        cs = torch.cat([enc(c) for _, c, _ in batch], 0)          # (B,d)
        dlist = [enc(d) for _, _, ds in batch for d in ds]        # B*k
        ds_mat = torch.cat(dlist, 0) if dlist else cs[:0]         # (B*k,d)
        B = qs.shape[0]
        # logits: [ q·correct(all B) | q·distractors(all) ]  target = own index
        logits = torch.cat([qs @ cs.T, qs @ ds_mat.T], 1) / _TAU  # (B, B+B*k)
        target = torch.arange(B, device=qs.device)
        l_con = F.cross_entropy(logits, target)
        # uniformity (Wang-Isola) over all embeddings
        X = torch.cat([qs, cs], 0)
        sq = torch.pdist(X).pow(2)
        l_uni = sq.mul(-2.0).exp().mean().add(1e-9).log() if sq.numel() else X.sum()*0
        return l_con, l_uni

    @torch.no_grad()
    def val_eval():
        ssm.eval()
        cl, gap, hit, n = 0.0, 0.0, 0, 0
        for i in range(0, len(val), args.batch):
            b = val[i:i + args.batch]
            if len(b) < 2: continue
            lc, _ = contrastive(b)
            cl += float(lc.item()) * len(b); n += len(b)
            for q, c, ds in b:
                qv = enc(q)
                scored = sorted(
                    [(float((qv * enc(t)).sum()), ok) for t, ok in
                     [(c, True)] + [(d, False) for d in ds]], reverse=True)
                hit += int(scored[0][1]); gap += scored[0][0] - scored[1][0]
        ssm.train()
        n = max(n, 1)
        return cl / n, gap / n, hit / n

    opt = torch.optim.AdamW(ssm.parameters(), lr=args.lr)
    v = val_eval(); print(f"[*] PRE: val_con={v[0]:.4f}  gap={v[1]:.4f}  acc={v[2]:.3f}")

    t0 = time.time()
    for ep in range(args.epochs):
        random.shuffle(train)
        run_c, run_u, nb = 0.0, 0.0, 0
        for i in range(0, len(train), args.batch):
            b = train[i:i + args.batch]
            if len(b) < 2: continue
            l_con, l_uni = contrastive(b)
            reg = ssm.train_dynamics(T(b[0][0]) + T(b[0][1]))  # cheap dynamics anchor
            loss = l_con + args.lambda_u * l_uni + args.reg_w * reg
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ssm.parameters(), 1.0); opt.step()
            run_c += float(l_con.item()); run_u += float(l_uni.item()); nb += 1
        v = val_eval()
        print(f"[*] epoch {ep+1}  train_con={run_c/max(nb,1):.4f}  uni={run_u/max(nb,1):.4f}"
              f"  |  VAL con={v[0]:.4f}  gap={v[1]:.4f}  acc={v[2]:.3f}  ({time.time()-t0:.0f}s)")

    ssm.eval(); torch.save(ssm.state_dict(), _CKPT)
    print(f"[*] saved → {_CKPT}.  revert: cp {_BACKUP} {_CKPT}")
    print("[*] re-run spike: .venv/bin/python experiments/mcq_discrimination.py --sample 200")


if __name__ == "__main__":
    main()
