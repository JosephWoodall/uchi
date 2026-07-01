"""
train_answerability.py — train the answerability classifier on SQuAD 2.0 and save
a checkpoint for uchi.answerability.AnswerabilityChecker.

(question + evidence) → P(answerable). Evidence = the answer-bearing sentence for
answerable questions; for unanswerable questions, the sentence most lexically
similar to the question (the trap the model must learn to reject). Trained from
scratch, embeddings warm-started from the brain's skip-gram vectors.

    .venv/bin/python experiments/train_answerability.py --limit 60000 --epochs 4 \
        --out uchi/data/answerability.pt
"""
from __future__ import annotations
import argparse, os, re, sys, time, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from uchi.answerability import build_classifier, tokenize, PAD, UNK, SEP  # noqa: E402
_EMB = os.path.join(os.path.dirname(__file__), "skipgram_emb.pt")
_WORD = re.compile(r"[a-z]{2,}")


def best_sentence(context, question):
    """Sentence most lexically overlapping the question (the plausible trap)."""
    qw = set(_WORD.findall(question.lower()))
    best, bs = context, -1
    for s in re.split(r"(?<=[.!?])\s+", context):
        sw = set(_WORD.findall(s.lower()))
        ov = len(qw & sw)
        if ov > bs:
            bs, best = ov, s
    return best


def ans_sentence(context, answer, start):
    sents = re.split(r"(?<=[.!?])\s+", context)
    pos, acc = 0, None
    for s in sents:
        if start <= pos + len(s) + 1 and start + len(answer) >= pos:
            acc = s
        pos += len(s) + 1
    return acc or (sents[0] if sents else context)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--dim", type=int, default=300)
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=80)
    ap.add_argument("--out", default="uchi/data/answerability.pt")
    args = ap.parse_args()
    random.seed(0)

    import torch, torch.nn as nn, torch.nn.functional as F
    from datasets import load_dataset
    from collections import Counter
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] device={dev}")

    ds = load_dataset("rajpurkar/squad_v2", split="train")
    rows = []
    for r in ds:
        q = r["question"]
        if r["answers"]["text"]:
            ev = ans_sentence(r["context"], r["answers"]["text"][0], r["answers"]["answer_start"][0])
            label = 1
        else:
            ev = best_sentence(r["context"], q)
            label = 0
        toks = (tokenize(q) + [SEP] + tokenize(ev))[:args.max_len]
        if toks:
            rows.append((toks, label))
        if len(rows) >= args.limit:
            break
    random.shuffle(rows)
    val, train = rows[:2000], rows[2000:]
    pos = sum(l for _, l in train)
    print(f"[*] train={len(train):,} val={len(val)}  ({pos/len(train)*100:.0f}% answerable)")

    cnt = Counter(w for s, _ in train for w in s)
    vocab = [PAD, UNK, SEP] + [w for w, c in cnt.most_common(30000) if c >= 2]
    stoi = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
    cfg = {"dim": args.dim, "hid": args.hid, "max_len": args.max_len}
    print(f"[*] vocab={V:,}")

    model = build_classifier(cfg, V, stoi[PAD], torch, nn, F).to(dev)
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
            L = max(len(s) for s, _ in b)
            X = torch.full((len(b), L), stoi[PAD])
            for j, (s, _) in enumerate(b):
                X[j, :len(s)] = torch.tensor([stoi.get(w, stoi[UNK]) for w in s])
            y = torch.tensor([float(l) for _, l in b])
            yield X.to(dev), y.to(dev)

    @torch.no_grad()
    def evaluate():
        model.eval(); correct = n = 0
        for X, y in batches(val):
            p = (torch.sigmoid(model(X)) > 0.5).float()
            correct += (p == y).sum().item(); n += len(y)
        model.train(); return correct / max(n, 1)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tot = nb = 0
        for X, y in batches(train):
            loss = F.binary_cross_entropy_with_logits(model(X), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        print(f"[*] epoch {ep+1}  loss={tot/nb:.3f}  val_acc={evaluate()*100:.1f}%  ({time.time()-t0:.0f}s)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"config": cfg, "vocab": vocab, "stoi": stoi,
                "state_dict": model.state_dict()}, args.out)
    print(f"[*] saved → {args.out}")


if __name__ == "__main__":
    main()
