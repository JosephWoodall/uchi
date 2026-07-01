"""
neural_decoder.py — Phase 1 of Generate-and-Ground: a small, from-scratch,
retrieval-conditioned neural answer generator (NOT a pretrained LLM).

  (question + evidence sentence)  --BiGRU encoder + attention decoder-->  answer

Trained from scratch (embeddings warm-started from the brain's skip-gram vectors)
on (question, evidence, answer) triples derived from SQuAD. At inference the
generated answer is FACT-CHECKED against the evidence — unsupported → abstain.
This is where the validated oracle becomes non-vacuous: the decoder produces NOVEL
token sequences, and the oracle catches the fabricated ones.

Honest expectation: a from-scratch seq2seq on modest data is rough, not fluent.
The point is to prove the generative loop + oracle gating end to end.

Usage:
    .venv/bin/python experiments/neural_decoder.py --limit 40000 --epochs 8
"""
from __future__ import annotations
import argparse, os, re, sys, time, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
_TOK = re.compile(r"[a-z0-9]+|[^\sa-z0-9]")
_EMB = os.path.join(os.path.dirname(__file__), "skipgram_emb.pt")
PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"


def tok(s):
    return _TOK.findall(s.lower())


def answer_sentence(context, answer, start):
    """The context sentence containing the answer span."""
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
    args = ap.parse_args()
    random.seed(0)

    import torch, torch.nn as nn, torch.nn.functional as F
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] device={dev}")

    # ── data: (question + evidence) -> answer ────────────────────────────────
    print("[*] loading SQuAD …")
    ds = load_dataset("rajpurkar/squad", split="train")
    rows = []
    for r in ds:
        ans = r["answers"]["text"]
        if not ans:
            continue
        a = ans[0]; st = r["answers"]["answer_start"][0]
        ev = answer_sentence(r["context"], a, st)
        q, at = tok(r["question"]), tok(a)
        if 1 <= len(at) <= args.max_tgt - 1 and q and ev:
            src = q + ["<sep>"] + tok(ev)
            rows.append((src[:args.max_src], at))
        if len(rows) >= args.limit:
            break
    random.shuffle(rows)
    val = rows[:500]; train = rows[500:]
    print(f"[*] train={len(train):,} val={len(val)}")

    # ── vocab (warm-start embeddings from skip-gram) ─────────────────────────
    from collections import Counter
    cnt = Counter(w for s, t in train for w in s + t)
    vocab = [PAD, SOS, EOS, UNK, "<sep>"] + [w for w, c in cnt.most_common(30000) if c >= 2]
    stoi = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
    print(f"[*] vocab={V:,}")

    emb0 = torch.empty(V, args.dim).uniform_(-0.05, 0.05)
    try:
        sg = torch.load(_EMB, map_location="cpu")
        sw2i, sE = sg["w2i"], sg["E"]
        hit = 0
        for w, i in stoi.items():
            if w in sw2i:
                emb0[i] = sE[sw2i[w]]; hit += 1
        print(f"[*] warm-started {hit:,}/{V:,} embeddings from skip-gram")
    except Exception as e:
        print(f"  [!] skip-gram warm-start skipped: {e}")

    def enc(seq, add_eos=False):
        ids = [stoi.get(w, stoi[UNK]) for w in seq]
        return ids + ([stoi[EOS]] if add_eos else [])

    def batchify(rows):
        random.shuffle(rows)
        for i in range(0, len(rows), args.batch):
            b = rows[i:i + args.batch]
            src = [enc(s) for s, _ in b]
            tgt = [[stoi[SOS]] + enc(t) + [stoi[EOS]] for _, t in b]
            sl = max(len(x) for x in src); tl = max(len(x) for x in tgt)
            S = torch.full((len(b), sl), stoi[PAD], dtype=torch.long)
            T = torch.full((len(b), tl), stoi[PAD], dtype=torch.long)
            for j, x in enumerate(src): S[j, :len(x)] = torch.tensor(x)
            for j, x in enumerate(tgt): T[j, :len(x)] = torch.tensor(x)
            yield S.to(dev), T.to(dev)

    # ── model: BiGRU encoder + Bahdanau-attention GRU decoder ────────────────
    class Seq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, args.dim, padding_idx=stoi[PAD])
            self.emb.weight.data.copy_(emb0)
            self.enc = nn.GRU(args.dim, args.hid, batch_first=True, bidirectional=True)
            self.bridge = nn.Linear(2 * args.hid, args.hid)
            self.dec = nn.GRU(args.dim + 2 * args.hid, args.hid, batch_first=True)
            self.attn = nn.Linear(args.hid + 2 * args.hid, 1)
            self.out = nn.Linear(args.hid, V)

        def encode(self, S):
            mask = (S != stoi[PAD])
            e = self.emb(S)
            H, h = self.enc(e)                                  # H:(B,L,2h)
            h = torch.tanh(self.bridge(torch.cat([h[0], h[1]], -1))).unsqueeze(0)
            return H, h, mask

        def step(self, y, h, H, mask):
            e = self.emb(y).unsqueeze(1)                        # (B,1,dim)
            hd = h[-1].unsqueeze(1).expand(-1, H.size(1), -1)   # (B,L,hid)
            sc = self.attn(torch.cat([hd, H], -1)).squeeze(-1)  # (B,L)
            sc = sc.masked_fill(~mask, -1e9)
            a = F.softmax(sc, -1).unsqueeze(1)                  # (B,1,L)
            ctx = torch.bmm(a, H)                               # (B,1,2h)
            o, h = self.dec(torch.cat([e, ctx], -1), h)
            return self.out(o.squeeze(1)), h

        def forward(self, S, T):
            H, h, mask = self.encode(S)
            logits = []
            for t in range(T.size(1) - 1):
                lo, h = self.step(T[:, t], h, H, mask)
                logits.append(lo)
            return torch.stack(logits, 1)                       # (B,T-1,V)

    model = Seq2Seq().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[*] model params={nparams/1e6:.1f}M")

    def greedy(src_tokens, max_len=args.max_tgt):
        model.eval()
        with torch.no_grad():
            S = torch.tensor([enc(src_tokens[:args.max_src])], device=dev)
            H, h, mask = model.encode(S)
            y = torch.tensor([stoi[SOS]], device=dev); out = []
            for _ in range(max_len):
                lo, h = model.step(y, h, H, mask)
                y = lo.argmax(-1)
                w = vocab[y.item()]
                if w == EOS:
                    break
                out.append(w)
        model.train()
        return out

    # ── train ────────────────────────────────────────────────────────────────
    t0 = time.time()
    for ep in range(args.epochs):
        tot = nb = 0
        for S, T in batchify(train):
            logits = model(S, T)
            loss = F.cross_entropy(logits.reshape(-1, V), T[:, 1:].reshape(-1),
                                   ignore_index=stoi[PAD])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        # quick val exact-match
        em = 0
        for s, t in val[:200]:
            if " ".join(greedy(s)) == " ".join(t):
                em += 1
        print(f"[*] epoch {ep+1}  loss={tot/nb:.3f}  val_EM={em/200*100:.1f}%  ({time.time()-t0:.0f}s)")

    # ── demo: generate + fact-check (oracle) gating ──────────────────────────
    _STOP = set("the a an of to in and or is are was were be what which who how why".split())
    def factcheck(ans_tokens, evidence_tokens):
        terms = [w for w in ans_tokens if w not in _STOP and len(w) > 2]
        if not terms:
            return 0.0
        ev = set(evidence_tokens)
        return sum(w in ev for w in terms) / len(terms)

    print("\n[*] DEMO — generate then oracle-gate (held-out questions):")
    for s, t in val[:15]:
        gen = greedy(s)
        sep = s.index("<sep>") if "<sep>" in s else len(s)
        q, ev = s[:sep], s[sep + 1:]
        sup = factcheck(gen, ev)
        shown = " ".join(gen) if sup >= 0.5 else "[abstain: unsupported]"
        print(f"  Q: {' '.join(q)}")
        print(f"     gold: {' '.join(t)}")
        print(f"     gen : {' '.join(gen)}   support={sup:.2f}  -> {shown}\n")


if __name__ == "__main__":
    main()
