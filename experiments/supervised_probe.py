"""
supervised_probe.py — can SUPERVISED learning on the validated semantic
substrate extract CORRECTNESS (not just topical similarity)?

Uses the frozen skip-gram embeddings (experiments/skipgram_emb.pt). A text is the
mean of its word vectors. A small learnable projection f maps question-mean and
answer-mean into a space trained so cos(f(q), f(correct)) > cos(f(q), f(distractor))
via in-batch-negative InfoNCE on ARC-train + MMLU-auxiliary.

Then evaluates on held-out ARC-Challenge / ARC-Easy / MMLU test.

  Beats random on ARC-Challenge/MMLU → supervised+semantic works → integrate the
    substrate into the SSM encoder.
  Still random → these benchmarks need retrieval+reasoning over brain facts, not a
    better discriminator. That redirects the architecture, decisively.

Usage:
    .venv/bin/python experiments/supervised_probe.py --epochs 8
"""
from __future__ import annotations
import argparse, os, re, sys, time, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_WORD = re.compile(r"[a-z]{2,}")
_LETTERS = {"A", "B", "C", "D"}
_EMB = os.path.join(os.path.dirname(__file__), "skipgram_emb.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--proj", type=int, default=256)
    ap.add_argument("--limit", type=int, default=40000, help="train tuples cap")
    ap.add_argument("--sample", type=int, default=500)
    args = ap.parse_args()
    random.seed(0)

    import torch, torch.nn as nn, torch.nn.functional as F
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = torch.load(_EMB, map_location=dev)
    w2i, E, dim = d["w2i"], d["E"].to(dev), d["dim"]
    E = F.normalize(E, p=2, dim=-1)
    print(f"[*] loaded {len(w2i):,} embeddings dim={dim}  device={dev}")

    def mean_vec(text):
        ids = [w2i[w] for w in _WORD.findall(text.lower()) if w in w2i]
        if not ids:
            return torch.zeros(dim, device=dev)
        return E[torch.tensor(ids, device=dev)].mean(0)

    # ── data: (q_vec, correct_vec, [distractor_vecs]) precomputed as tensors ──
    def load_tuples(limit):
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
            print("  [!] arc:", e)
        try:
            ds = load_dataset("cais/mmlu", "auxiliary_train", split="train")
            for j, row in enumerate(ds):
                if j >= limit:
                    break
                r = row.get("train", row) if isinstance(row, dict) else row
                ch = r["choices"]
                if len(ch) == 4 and r.get("question"):
                    ai = int(r["answer"])
                    out.append((r["question"], ch[ai],
                                [c for k, c in enumerate(ch) if k != ai]))
        except Exception as e:
            print("  [!] mmlu aux:", e)
        random.shuffle(out)
        return out

    print("[*] loading + vectorising train tuples …")
    raw = load_tuples(args.limit)
    Q = torch.stack([mean_vec(q) for q, _, _ in raw])            # (N, dim)
    C = torch.stack([mean_vec(c) for _, c, _ in raw])            # (N, dim)
    # keep exactly 3 distractors per row for a clean tensor
    D = torch.stack([torch.stack([mean_vec(x) for x in (ds + ds)[:3]])
                     for _, _, ds in raw])                       # (N, 3, dim)
    N = Q.shape[0]
    print(f"[*] {N:,} train tuples vectorised")

    # ── model: projection MLP ────────────────────────────────────────────────
    proj = nn.Sequential(
        nn.Linear(dim, args.proj), nn.SiLU(), nn.LayerNorm(args.proj),
        nn.Linear(args.proj, args.proj),
    ).to(dev)
    opt = torch.optim.AdamW(proj.parameters(), lr=args.lr)
    tau = 0.07

    def f(x):
        return F.normalize(proj(x), p=2, dim=-1)

    # ── eval helper ──────────────────────────────────────────────────────────
    def evaluate(name, rows):
        proj.eval()
        hit, n = 0, 0
        with torch.no_grad():
            for q, labels, texts, ans in rows:
                qv = f(mean_vec(q).unsqueeze(0))
                ov = f(torch.stack([mean_vec(t) for t in texts]))
                pred = labels[int((ov @ qv.T).squeeze(1).argmax())]
                hit += int(pred == ans); n += 1
        proj.train()
        print(f"  {name:<16} {hit/max(n,1)*100:5.1f}%  ({hit}/{n})")

    def arc_rows(cfg):
        ds = load_dataset("allenai/ai2_arc", cfg, split="test"); out = []
        for r in ds:
            labels, texts = r["choices"]["label"], r["choices"]["text"]
            ans = r["answerKey"].strip().upper()
            if set(labels).issubset(_LETTERS) and ans in labels:
                out.append((r["question"], labels, texts, ans))
        random.shuffle(out); return out[:args.sample]

    def mmlu_rows():
        ds = load_dataset("cais/mmlu", "all", split="test")
        idx2 = ["A", "B", "C", "D"]; out = []
        for r in ds:
            ch = r["choices"]
            if len(ch) == 4:
                out.append((r["question"], idx2, ch, idx2[int(r["answer"])]))
        random.shuffle(out); return out[:args.sample]

    arc_c, arc_e, mmlu = arc_rows("ARC-Challenge"), arc_rows("ARC-Easy"), mmlu_rows()
    print("\n[*] PRE-TRAIN (random=25%):")
    evaluate("ARC-Challenge", arc_c); evaluate("ARC-Easy", arc_e); evaluate("MMLU", mmlu)

    # ── train: in-batch-negative InfoNCE ─────────────────────────────────────
    print(f"\n[*] training projection ({args.epochs} ep, batch {args.batch}) …")
    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(N, device=dev)
        tot = 0.0; nb = 0
        for k in range(0, N, args.batch):
            idx = perm[k:k + args.batch]
            if idx.numel() < 2:
                continue
            q = f(Q[idx]); c = f(C[idx])                     # (b,p)
            dd = f(D[idx].reshape(-1, dim)).reshape(idx.numel(), 3, -1)  # (b,3,p)
            # logits: [q·all_correct(in-batch) | q·own 3 distractors] target=self
            inb = q @ c.T                                    # (b,b)
            hard = torch.einsum("bp,bkp->bk", q, dd)         # (b,3)
            logits = torch.cat([inb, hard], 1) / tau
            target = torch.arange(idx.numel(), device=dev)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % 2 == 1 or ep == args.epochs - 1:
            print(f"  epoch {ep+1}  loss={tot/max(nb,1):.4f}  ({time.time()-t0:.0f}s)")

    print("\n[*] POST-TRAIN (random=25%):")
    evaluate("ARC-Challenge", arc_c); evaluate("ARC-Easy", arc_e); evaluate("MMLU", mmlu)
    print("\n  supervised+semantic beats random on ARC-C/MMLU → integrate into SSM.")
    print("  still random → these need retrieval+reasoning, not a better discriminator.")


if __name__ == "__main__":
    main()
