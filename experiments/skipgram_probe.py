"""
skipgram_probe.py — de-risk the semantic substrate BEFORE any SSM surgery.

Hypothesis: the char-CNN encoder failed at QA because character embeddings can't
carry word meaning. A word-level skip-gram (word2vec) space trained on the
brain's corpus SHOULD place related concepts near each other — the pre-LLM
solution to "different words, same meaning."

This trains skip-gram embeddings (torch, GPU) on wikipedia + MMLU-aux + ARC-train,
then runs the PUREST possible semantic probe on held-out ARC test:

    represent a text as the mean of its word embeddings
    predict = argmax_option  cos(mean(question), mean(option))

No SSM, no trie. If this clears 25%, the substrate carries semantics and the full
integration is worth building. If it's at random, the corpus is too small → iterate
on data before spending hours on architecture.

Usage:
    .venv/bin/python experiments/skipgram_probe.py --wiki 8000 --dim 300 --epochs 4
"""
from __future__ import annotations
import argparse, math, re, time, random, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_WORD = re.compile(r"[a-z]{2,}")
_LETTERS = {"A", "B", "C", "D"}


def _words(text: str):
    return _WORD.findall(text.lower())


def _build_corpus(wiki_n: int, art_words: int = 300, mmlu_cap: int = 60000):
    from datasets import load_dataset
    sents = []
    # Wikipedia — richest natural-language co-occurrence signal (truncate articles)
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train",
                          streaming=True)
        for i, r in enumerate(ds):
            if i >= wiki_n:
                break
            w = _words(r["text"])[:art_words]
            if len(w) > 20:
                sents.append(w)
    except Exception as e:
        print(f"  [!] wikipedia stream failed: {e}")
    # MMLU auxiliary_train — domain-relevant academic language (capped)
    try:
        ds = load_dataset("cais/mmlu", "auxiliary_train", split="train")
        for j, row in enumerate(ds):
            if j >= mmlu_cap:
                break
            r = row.get("train", row) if isinstance(row, dict) else row
            ch = r.get("choices") or []
            txt = (r.get("question") or "") + " " + " ".join(ch)
            w = _words(txt)
            if len(w) > 4:
                sents.append(w)
    except Exception as e:
        print(f"  [!] mmlu aux failed: {e}")
    # ARC train
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        for r in ds:
            txt = r["question"] + " " + " ".join(r["choices"]["text"])
            w = _words(txt)
            if len(w) > 4:
                sents.append(w)
    except Exception as e:
        print(f"  [!] arc train failed: {e}")
    return sents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", type=int, default=8000)
    ap.add_argument("--dim", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--neg", type=int, default=5)
    ap.add_argument("--min_count", type=int, default=5)
    ap.add_argument("--sample", type=int, default=400, help="ARC test probe size")
    args = ap.parse_args()
    random.seed(0)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] device={dev}")

    print("[*] building corpus …")
    sents = _build_corpus(args.wiki)
    tot = sum(len(s) for s in sents)
    print(f"[*] {len(sents)} sentences, {tot:,} word tokens")

    # vocab
    from collections import Counter
    cnt = Counter(w for s in sents for w in s)
    vocab = [w for w, c in cnt.items() if c >= args.min_count]
    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    print(f"[*] vocab={V:,} (min_count={args.min_count})")

    # subsampling probability (word2vec) + negative sampling table (unigram^0.75)
    total = sum(cnt[w] for w in vocab)
    keep = {w: (math.sqrt(cnt[w] / (1e-3 * total)) + 1) * (1e-3 * total) / cnt[w]
            for w in vocab}
    freq = torch.tensor([cnt[w] ** 0.75 for w in vocab], dtype=torch.float)
    neg_dist = (freq / freq.sum()).to(dev)

    # Flat subsampled token-id stream (memory-safe: O(corpus), not O(pairs)).
    # Pairs are sampled ON THE FLY during training — avoids materialising
    # hundreds of millions of tuples (the earlier OOM).
    stream = []
    for s in sents:
        for w in s:
            if w in w2i and random.random() < min(1.0, keep[w]):
                stream.append(w2i[w])
    stream = torch.tensor(stream, dtype=torch.long)
    L = stream.numel()
    print(f"[*] flat stream = {L:,} subsampled tokens")
    if L < 1000:
        print("[!] corpus too small"); return

    emb_in = nn.Embedding(V, args.dim).to(dev)
    emb_out = nn.Embedding(V, args.dim).to(dev)
    nn.init.uniform_(emb_in.weight, -0.5 / args.dim, 0.5 / args.dim)
    nn.init.zeros_(emb_out.weight)
    opt = torch.optim.Adam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=2e-3)

    stream_dev = stream.to(dev)
    B = 8192
    steps = max(1, L // B)
    t0 = time.time()
    for ep in range(args.epochs):
        tot_loss = 0.0
        for _ in range(steps):
            # sample B centers away from the edges, one random context each
            c_idx = torch.randint(args.window, L - args.window, (B,), device=dev)
            off = torch.randint(1, args.window + 1, (B,), device=dev)
            off = off * (torch.randint(0, 2, (B,), device=dev) * 2 - 1)  # ±
            o_idx = c_idx + off
            c = stream_dev[c_idx]; o = stream_dev[o_idx]
            negs = torch.multinomial(neg_dist, B * args.neg, replacement=True).view(B, args.neg)
            vc = emb_in(c); vo = emb_out(o); vn = emb_out(negs)
            pos = F.logsigmoid((vc * vo).sum(-1))
            neg = F.logsigmoid(-(vn * vc.unsqueeze(1)).sum(-1)).sum(-1)
            loss = -(pos + neg).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item()
        print(f"  epoch {ep+1}  loss={tot_loss/steps:.4f}  ({time.time()-t0:.0f}s)")

    E = F.normalize(emb_in.weight.detach(), p=2, dim=-1)  # (V, d) on device

    def vec(text):
        ids = [w2i[w] for w in _words(text) if w in w2i]
        if not ids:
            return None
        return F.normalize(E[torch.tensor(ids, device=dev)].mean(0), p=2, dim=-1)

    # nearest-neighbor sanity
    def nn_of(word, k=8):
        if word not in w2i:
            return f"{word}: OOV"
        q = E[w2i[word]]
        sims = E @ q
        top = sims.topk(k + 1).indices.tolist()[1:]
        return f"{word}: " + ", ".join(vocab[i] for i in top)

    print("\n[*] nearest neighbors (semantic sanity):")
    for w in ["mitochondria", "energy", "water", "gravity", "cell", "planet"]:
        print("   ", nn_of(w))

    # save embeddings for reuse (SSM integration, no retrain)
    out = os.path.join(os.path.dirname(__file__), "skipgram_emb.pt")
    torch.save({"vocab": vocab, "w2i": w2i, "E": E.cpu(), "dim": args.dim}, out)
    print(f"\n[*] saved embeddings → {out}")

    from datasets import load_dataset

    def _probe(name, rows):
        hit, n, gap = 0, 0, 0.0
        for q, labels, texts, ans in rows:
            qv = vec(q)
            if qv is None:
                continue
            scored = []
            for L, t in zip(labels, texts):
                ov = vec(t)
                s = float((qv @ ov).item()) if ov is not None else -9.9
                scored.append((s, L))
            scored.sort(reverse=True)
            hit += int(scored[0][1] == ans); gap += scored[0][0] - scored[1][0]; n += 1
        n = max(n, 1)
        print(f"  {name:<16} {hit/n*100:5.1f}%  ({hit}/{n})   gap={gap/n:.4f}")

    def _arc_rows(cfg):
        ds = load_dataset("allenai/ai2_arc", cfg, split="test")
        out = []
        for r in ds:
            labels, texts = r["choices"]["label"], r["choices"]["text"]
            ans = r["answerKey"].strip().upper()
            if set(labels).issubset(_LETTERS) and ans in labels:
                out.append((r["question"], labels, texts, ans))
        random.shuffle(out); return out[:args.sample]

    def _mmlu_rows():
        ds = load_dataset("cais/mmlu", "all", split="test")
        idx2 = ["A", "B", "C", "D"]; out = []
        for r in ds:
            ch = r["choices"]
            if len(ch) == 4:
                out.append((r["question"], idx2, ch, idx2[int(r["answer"])]))
        random.shuffle(out); return out[:args.sample]

    print("\n" + "─" * 56)
    print("  SKIP-GRAM mean-emb cosine probe (random = 25%)")
    print("─" * 56)
    _probe("ARC-Challenge", _arc_rows("ARC-Challenge"))
    _probe("ARC-Easy", _arc_rows("ARC-Easy"))
    _probe("MMLU", _mmlu_rows())
    print("─" * 56)
    print("  Note: ARC-Challenge is adversarial to similarity BY DESIGN.")
    print("  If ARC-Easy / MMLU clear 25%, the substrate helps knowledge tasks;")
    print("  the real test is SUPERVISED SSM discrimination on these features.")


if __name__ == "__main__":
    main()
