"""
train_decoder.py — train the Generate-and-Ground decoder and save a checkpoint
that `uchi.decoder.NeuralDecoder.load` can consume.

Uses the SAME model definition as `uchi/decoder.py` (_build_model) so the saved
state_dict loads cleanly at inference. Trains from scratch on SQuAD
(question+evidence → answer), embeddings warm-started from the brain's skip-gram.

    .venv/bin/python experiments/train_decoder.py --limit 40000 --epochs 8 \
        --out uchi/data/decoder.pt
"""
from __future__ import annotations
import argparse, os, re, sys, time, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from uchi.decoder import _build_model, tokenize, PAD, SOS, EOS, UNK, SEP  # noqa: E402
_EMB = os.path.join(os.path.dirname(__file__), "skipgram_emb.pt")


def answer_sentence(context, answer, start):
    sents = re.split(r"(?<=[.!?])\s+", context)
    pos, acc = 0, None
    for s in sents:
        if start <= pos + len(s) + 1 and start + len(answer) >= pos:
            acc = s
        pos += len(s) + 1
    return acc or (sents[0] if sents else context)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--dim", type=int, default=300)
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--max_src", type=int, default=60)
    ap.add_argument("--max_tgt", type=int, default=16)
    ap.add_argument("--out", default="uchi/data/decoder.pt")
    args = ap.parse_args()
    random.seed(0)

    import torch, torch.nn as nn, torch.nn.functional as F
    from datasets import load_dataset
    from collections import Counter
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] device={dev}")

    ds = load_dataset("rajpurkar/squad", split="train")
    rows = []
    for r in ds:
        a = r["answers"]["text"]
        if not a:
            continue
        ev = answer_sentence(r["context"], a[0], r["answers"]["answer_start"][0])
        q, at = tokenize(r["question"]), tokenize(a[0])
        if 1 <= len(at) <= args.max_tgt - 1 and q and ev:
            rows.append(((q + [SEP] + tokenize(ev))[:args.max_src], at))
        if len(rows) >= args.limit:
            break
    random.shuffle(rows)
    val, train = rows[:500], rows[500:]
    print(f"[*] train={len(train):,} val={len(val)}")

    cnt = Counter(w for s, t in train for w in s + t)
    vocab = [PAD, SOS, EOS, UNK, SEP] + [w for w, c in cnt.most_common(30000) if c >= 2]
    stoi = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
    cfg = {"dim": args.dim, "hid": args.hid, "max_src": args.max_src, "max_tgt": args.max_tgt}
    print(f"[*] vocab={V:,}")

    model = _build_model(cfg, V, stoi[PAD], torch, nn, F).to(dev)
    # warm-start embeddings from skip-gram
    try:
        sg = torch.load(_EMB, map_location="cpu"); sw2i, sE = sg["w2i"], sg["E"]
        hit = 0
        with torch.no_grad():
            for w, i in stoi.items():
                if w in sw2i:
                    model.emb.weight[i] = sE[sw2i[w]]; hit += 1
        print(f"[*] warm-started {hit:,}/{V:,} embeddings")
    except Exception as e:
        print(f"  [!] warm-start skipped: {e}")
    print(f"[*] params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def batches(rows):
        random.shuffle(rows)
        for i in range(0, len(rows), args.batch):
            b = rows[i:i + args.batch]
            src = [[stoi.get(w, stoi[UNK]) for w in s] for s, _ in b]
            tgt = [[stoi[SOS]] + [stoi.get(w, stoi[UNK]) for w in t] + [stoi[EOS]] for _, t in b]
            sl, tl = max(map(len, src)), max(map(len, tgt))
            S = torch.full((len(b), sl), stoi[PAD]); T = torch.full((len(b), tl), stoi[PAD])
            for j, x in enumerate(src): S[j, :len(x)] = torch.tensor(x)
            for j, x in enumerate(tgt): T[j, :len(x)] = torch.tensor(x)
            yield S.to(dev), T.to(dev)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tot = nb = 0
        for S, T in batches(train):
            lo = model(S, T)
            loss = F.cross_entropy(lo.reshape(-1, V), T[:, 1:].reshape(-1), ignore_index=stoi[PAD])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        print(f"[*] epoch {ep+1}  loss={tot/nb:.3f}  ({time.time()-t0:.0f}s)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"config": cfg, "vocab": vocab, "stoi": stoi,
                "state_dict": model.state_dict()}, args.out)
    print(f"[*] saved decoder checkpoint → {args.out}")


if __name__ == "__main__":
    main()
