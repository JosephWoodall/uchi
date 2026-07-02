"""
train_chat_decoder.py — train Uchi's conversational decoder on DailyDialog.

Same seq2seq architecture as the answer decoder (uchi.decoder), but trained on
(conversation context → response) pairs instead of (question+evidence → answer).
Social chat is SAFE to free-generate: it asserts no facts, so it needs no oracle
gate — the anti-confabulation machinery is only for factual answers.

src format matches NeuralDecoder.generate(context, evidence=[])  →  context + <sep>
so the same inference wrapper loads both checkpoints.

    .venv/bin/python experiments/train_chat_decoder.py --epochs 10 \
        --out uchi/data/chat_decoder.pt
"""
from __future__ import annotations
import argparse, os, sys, time, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from uchi.decoder import _build_model, tokenize, PAD, SOS, EOS, UNK, SEP  # noqa: E402
_EMB = os.path.join(os.path.dirname(__file__), "skipgram_emb.pt")


def load_pairs(limit):
    from datasets import load_dataset
    import ast
    pairs = []
    try:
        ds = load_dataset("Estwld/empathetic_dialogues_llm", split="train")
    except Exception as e:
        print(f"[!] dialogue dataset unavailable: {e}")
        return pairs
    for r in ds:
        conv = r.get("conversations")
        if isinstance(conv, str):
            try:
                conv = ast.literal_eval(conv)
            except Exception:
                conv = []
        turns = [t.get("content", "").strip() for t in (conv or []) if isinstance(t, dict)]
        turns = [t for t in turns if t]
        for i in range(len(turns) - 1):
            pairs.append((turns[i], turns[i + 1]))
        if len(pairs) >= limit:
            break
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--dim", type=int, default=300)
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--max_src", type=int, default=40)
    ap.add_argument("--max_tgt", type=int, default=24)
    ap.add_argument("--out", default="uchi/data/chat_decoder.pt")
    args = ap.parse_args()
    random.seed(0)

    import torch, torch.nn as nn, torch.nn.functional as F
    from collections import Counter
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] device={dev}")

    raw = load_pairs(args.limit)
    rows = []
    for ctx, resp in raw:
        src = (tokenize(ctx) + [SEP])[:args.max_src]
        tgt = tokenize(resp)[:args.max_tgt - 1]
        if src and tgt:
            rows.append((src, tgt))
    random.shuffle(rows)
    val, train = rows[:500], rows[500:]
    print(f"[*] train={len(train):,} val={len(val)}")
    if len(train) < 100:
        print("[!] not enough dialogue data"); return

    cnt = Counter(w for s, t in train for w in s + t)
    vocab = [PAD, SOS, EOS, UNK, SEP] + [w for w, c in cnt.most_common(25000) if c >= 2]
    stoi = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
    cfg = {"dim": args.dim, "hid": args.hid, "max_src": args.max_src, "max_tgt": args.max_tgt}
    print(f"[*] vocab={V:,}")

    model = _build_model(cfg, V, stoi[PAD], torch, nn, F).to(dev)
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
    print(f"[*] saved chat decoder → {args.out}")


if __name__ == "__main__":
    main()
