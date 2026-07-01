"""
transe_reason.py — test the ACTUAL hybrid claim: does a SCORED grounded relation
discriminate the correct answer from distractors? (Reachability didn't — the noisy
graph connects everything to everything.)

Two upgrades over relation_extract.py:
  1. CLEAN extraction — concepts must be content nouns/proper-nouns (no pronouns,
     determiners, stopwords, hub words); relations are content-verb lemmas or a
     typed "is_a" copula. Multi-word concepts via noun-chunk roots.
  2. TransE embeddings (Bordes 2013): learn entity + relation vectors so
     head + relation ≈ tail. Relatedness(h,t) = -min_r ||r - (t - h)||  — is there
     a PLAUSIBLE relation linking them, even if that exact edge was never seen?

Discrimination on held-out ARC:
     score(option) = max_{qc in Q, oc in option} relatedness(qc, oc)
     predict = argmax option.  Beats 25% → latent-grounded reasoning discriminates.

Usage:
    .venv/bin/python experiments/transe_reason.py --wiki 4000 --dim 100 --epochs 30
"""
from __future__ import annotations
import argparse, os, re, sys, time, random
from collections import defaultdict, Counter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_LETTERS = {"A", "B", "C", "D"}


def _load_spacy():
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    return nlp


def clean_extract(nlp, sentences, max_sents):
    """Content-word SVO + typed is_a copula triples."""
    triples = []
    def ok(tok):
        return (tok.pos_ in ("NOUN", "PROPN") and not tok.is_stop
                and len(tok.lemma_) >= 3 and tok.lemma_.isalpha())
    for doc in nlp.pipe(sentences[:max_sents], batch_size=64):
        for tok in doc:
            if tok.pos_ == "VERB":
                subs = [w for w in tok.children if w.dep_ in ("nsubj", "nsubjpass") and ok(w)]
                if not subs:
                    continue
                rel = tok.lemma_.lower()
                objs = []
                for w in tok.children:
                    if w.dep_ in ("dobj", "attr", "oprd") and ok(w):
                        objs.append((rel, w))
                    elif w.dep_ == "prep":
                        for p in w.children:
                            if p.dep_ == "pobj" and ok(p):
                                objs.append((f"{rel}_{w.lemma_.lower()}", p))
                for s in subs:
                    for r, o in objs:
                        if s.lemma_ != o.lemma_:
                            triples.append((s.lemma_.lower(), r, o.lemma_.lower()))
            elif tok.lemma_ == "be":
                # X is a Y  (typed hypernym relation)
                subs = [w for w in tok.children if w.dep_ == "nsubj" and ok(w)]
                attrs = [w for w in tok.children if w.dep_ == "attr" and ok(w)]
                for s in subs:
                    for a in attrs:
                        if s.lemma_ != a.lemma_:
                            triples.append((s.lemma_.lower(), "is_a", a.lemma_.lower()))
    return triples


def _words(t):
    return re.findall(r"[a-z]{3,}", t.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", type=int, default=4000)
    ap.add_argument("--max_sents", type=int, default=50000)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--min_edge", type=int, default=2, help="drop triples seen < this")
    ap.add_argument("--hub_drop", type=int, default=150, help="drop N highest-degree hub concepts")
    ap.add_argument("--sample", type=int, default=400)
    args = ap.parse_args()
    random.seed(0)

    import torch, torch.nn.functional as F
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    nlp = _load_spacy()
    print("[*] building corpus …")
    sents = []
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        for i, r in enumerate(ds):
            if i >= args.wiki:
                break
            for s in re.split(r"(?<=[.!?])\s+", r["text"][:3000]):
                if 5 <= len(s.split()) <= 40:
                    sents.append(s)
    except Exception as e:
        print("  [!] wiki:", e)
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        for r in ds:
            sents.append(r["question"] + " " + " ".join(r["choices"]["text"]))
    except Exception as e:
        print("  [!] arc:", e)
    random.shuffle(sents)
    print(f"[*] {len(sents):,} sentences")

    t0 = time.time()
    raw = clean_extract(nlp, sents, args.max_sents)
    tc = Counter(raw)
    triples = [t for t, c in tc.items() if c >= args.min_edge]
    print(f"[*] {len(raw):,} raw → {len(triples):,} triples (min_edge={args.min_edge})  ({time.time()-t0:.0f}s)")

    # drop hub concepts (highest degree) — they connect everything
    deg = Counter()
    for h, r, t in triples:
        deg[h] += 1; deg[t] += 1
    hubs = {c for c, _ in deg.most_common(args.hub_drop)}
    triples = [(h, r, t) for h, r, t in triples if h not in hubs and t not in hubs]

    ents = sorted({h for h, _, _ in triples} | {t for _, _, t in triples})
    rels = sorted({r for _, r, _ in triples})
    e2i = {e: i for i, e in enumerate(ents)}
    r2i = {r: i for i, r in enumerate(rels)}
    print(f"[*] graph: {len(ents):,} concepts, {len(rels):,} relation types, {len(triples):,} edges")
    print("\n[*] sample CLEAN triples:")
    for h, r, t in random.sample(triples, min(18, len(triples))):
        print(f"    ({h}) --{r}--> ({t})")

    if len(triples) < 500:
        print("[!] too few triples — extraction still the wall"); return

    # ── TransE ───────────────────────────────────────────────────────────────
    H = torch.tensor([e2i[h] for h, _, _ in triples], device=dev)
    R = torch.tensor([r2i[r] for _, r, _ in triples], device=dev)
    T = torch.tensor([e2i[t] for _, _, t in triples], device=dev)
    nE, nR = len(ents), len(rels)
    Emb = torch.nn.Embedding(nE, args.dim).to(dev)
    Rel = torch.nn.Embedding(nR, args.dim).to(dev)
    torch.nn.init.uniform_(Emb.weight, -6 / args.dim ** 0.5, 6 / args.dim ** 0.5)
    torch.nn.init.uniform_(Rel.weight, -6 / args.dim ** 0.5, 6 / args.dim ** 0.5)
    opt = torch.optim.Adam(list(Emb.parameters()) + list(Rel.parameters()), lr=1e-2)
    margin, N, B = 1.0, len(triples), 4096
    for ep in range(args.epochs):
        with torch.no_grad():
            Emb.weight.data = F.normalize(Emb.weight.data, p=2, dim=-1)
        perm = torch.randperm(N, device=dev); tot = 0.0; nb = 0
        for k in range(0, N, B):
            idx = perm[k:k + B]
            h, r, t = Emb(H[idx]), Rel(R[idx]), Emb(T[idx])
            tn = Emb(torch.randint(nE, (idx.numel(),), device=dev))  # corrupt tail
            pos = (h + r - t).norm(dim=-1)
            neg = (h + r - tn).norm(dim=-1)
            loss = F.relu(margin + pos - neg).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % 10 == 9 or ep == args.epochs - 1:
            print(f"  transE epoch {ep+1}  loss={tot/nb:.4f}")

    with torch.no_grad():
        EW = F.normalize(Emb.weight, p=2, dim=-1)      # (nE, d)
        RW = Rel.weight                                 # (nR, d)

    # ── grounded multi-hop path scoring ──────────────────────────────────────
    # Undirected edge tensors (both directions); each edge scored by TransE
    # plausibility P = -||h + r - t||.  Best K-hop path plausibility from the
    # question concepts to each node via scatter-max relaxation (grounded on REAL
    # edges, scored in latent space). This IS multi-step reasoning: compose hops.
    src, dst, rl = [], [], []
    for h, r, t in triples:
        a, b, ri = e2i[h], e2i[t], r2i[r]
        src += [a, b]; dst += [b, a]; rl += [ri, ri]
    A = torch.tensor(src, device=dev)
    Bd = torch.tensor(dst, device=dev)
    Rr = torch.tensor(rl, device=dev)
    with torch.no_grad():
        P = -(EW[A] + RW[Rr] - EW[Bd]).norm(dim=1)      # (E,) edge plausibility

    def path_score_vec(qc_ids, hops):
        score = torch.full((len(ents),), -1e9, device=dev)
        score[torch.tensor(qc_ids, device=dev)] = 0.0
        for _ in range(hops):
            cand = score[A] + P
            new = torch.full((len(ents),), -1e9, device=dev)
            new = new.scatter_reduce(0, Bd, cand, reduce="amax", include_self=True)
            score = torch.maximum(score, new)
        return score

    # ── benchmark loaders ────────────────────────────────────────────────────
    def arc_rows(cfg):
        ds = load_dataset("allenai/ai2_arc", cfg, split="test"); out = []
        for r in ds:
            labels, texts = r["choices"]["label"], r["choices"]["text"]
            ans = r["answerKey"].strip().upper()
            if set(labels).issubset(_LETTERS) and ans in labels:
                out.append((r["question"], labels, texts, ans))
        random.shuffle(out); return out[:args.sample]

    def mmlu_rows():
        ds = load_dataset("cais/mmlu", "all", split="test"); idx2 = ["A","B","C","D"]; out = []
        for r in ds:
            ch = r["choices"]
            if len(ch) == 4:
                out.append((r["question"], idx2, ch, idx2[int(r["answer"])]))
        random.shuffle(out); return out[:args.sample]

    def evaluate(name, rows, hops):
        hit = n = scored = 0
        for q, labels, texts, ans in rows:
            n += 1
            qc = [e2i[w] for w in _words(q) if w in e2i]
            if not qc:
                continue
            sv = path_score_vec(qc, hops)
            best_L, best_s, any_o = None, -1e18, False
            for L, txt in zip(labels, texts):
                oc = [e2i[w] for w in _words(txt) if w in e2i]
                if not oc:
                    continue
                any_o = True
                s = sv[torch.tensor(oc, device=dev)].max().item()
                if s > best_s:
                    best_s, best_L = s, L
            if any_o:
                scored += 1; hit += int(best_L == ans)
        print(f"  {name:<14} hop={hops}  {hit/max(scored,1)*100:5.1f}%  "
              f"(scorable {scored}/{n})")
        return hit / max(scored, 1)

    print("\n" + "─" * 62)
    print("  GROUNDED MULTI-HOP path reasoning (TransE-scored)  random=25%")
    print("─" * 62)
    arc_c, arc_e, mmlu = arc_rows("ARC-Challenge"), arc_rows("ARC-Easy"), mmlu_rows()
    for hops in (1, 2, 3):
        evaluate("ARC-Challenge", arc_c, hops)
        evaluate("ARC-Easy", arc_e, hops)
        evaluate("MMLU", mmlu, hops)
        print()
    print("─" * 62)
    print("  Read: does adding hops (composition) lift accuracy above 1-hop?")
    print("  Lift on ARC-Easy/MMLU → grounded composition helps knowledge tasks.")
    print("  Flat on ARC-Challenge → its reasoning isn't relational-path shaped.")


if __name__ == "__main__":
    main()
