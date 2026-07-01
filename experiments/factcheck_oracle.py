"""
factcheck_oracle.py — oracle v2 for Generate-and-Ground (the trie oracle failed).

Different mechanism: verify a claim by RETRIEVING relevant knowledge and checking
whether the claim's content is SUPPORTED by it — not the trie's n-gram probability.

Setup (honest test of the mechanism):
  - Knowledge base = Wikipedia sentences (proxy for Uchi's ingested brain), embedded
    with the validated skip-gram vectors (experiments/skipgram_emb.pt).
  - Test facts drawn FROM the KB (so the brain genuinely HAS them):
      claim = (subject-context words, OBJECT word)   [TRUE  — object is the real one]
      false = same context, OBJECT swapped for a plausible one from another fact.
  - Oracle score(context, object):
      retrieve top-k KB sentences by cos(context) ; support = is `object` present
      in / semantically consistent with the retrieved evidence?
  - Metric: pairwise % where support(true) > support(false)  (chance 50%),
    plus grounding recall (true supported) and false-accept rate.

If this separates true from false where the trie couldn't, the trustworthy
factual-answering scope of the architecture is validated.

Usage:
    .venv/bin/python experiments/factcheck_oracle.py --kb 8000 --facts 600 --topk 10
"""
from __future__ import annotations
import argparse, os, re, sys, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
_WORD = re.compile(r"[a-z]{3,}")
_EMB = os.path.join(os.path.dirname(__file__), "skipgram_emb.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", type=int, default=8000, help="wiki sentences in the knowledge base")
    ap.add_argument("--facts", type=int, default=600)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    random.seed(0)

    import torch, torch.nn.functional as F
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = torch.load(_EMB, map_location=dev)
    w2i, E = d["w2i"], F.normalize(d["E"].to(dev), p=2, dim=-1)
    dim = d["dim"]
    print(f"[*] embeddings {len(w2i):,}  dim={dim}  device={dev}")

    STOP = set("the a an of to in and or is are was were be been being for on at by with "
               "as it its his her their they them from that this these those which who "
               "what when where how why not but if then than also into over under more "
               "most some any all each other one two".split())

    def mean_vec(words):
        ids = [w2i[w] for w in words if w in w2i]
        if not ids:
            return None
        return F.normalize(E[torch.tensor(ids, device=dev)].mean(0), p=2, dim=-1)

    # ── knowledge base ────────────────────────────────────────────────────────
    print("[*] building knowledge base …")
    kb_words, kb_vecs = [], []
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    for i, r in enumerate(ds):
        if len(kb_words) >= args.kb:
            break
        for s in re.split(r"(?<=[.!?])\s+", r["text"][:2500]):
            w = _WORD.findall(s.lower())
            if 6 <= len(w) <= 40:
                v = mean_vec(w)
                if v is not None:
                    kb_words.append(w); kb_vecs.append(v)
                    if len(kb_words) >= args.kb:
                        break
    KB = torch.stack(kb_vecs)                      # (N, dim)
    kb_sets = [set(w) for w in kb_words]
    print(f"[*] KB = {len(kb_words):,} sentences")

    # ── test facts drawn from the KB ──────────────────────────────────────────
    content_pool = list({w for ws in kb_words for w in ws if w not in STOP and len(w) >= 5})
    facts = []
    idxs = list(range(len(kb_words)))
    random.shuffle(idxs)
    for i in idxs:
        w = kb_words[i]
        contents = [x for x in w if x not in STOP and len(x) >= 5]
        if len(contents) < 3:
            continue
        obj = contents[-1]                          # object = a salient late content word
        ctx = [x for x in w if x != obj][:12]       # context = the rest
        if mean_vec(ctx) is None:
            continue
        facts.append((ctx, obj, i))
        if len(facts) >= args.facts:
            break
    print(f"[*] {len(facts)} test facts")

    def support(ctx, obj):
        """Retrieve by context; is `obj` supported by the top-k evidence?"""
        qv = mean_vec(ctx)
        if qv is None:
            return 0.0
        sims = KB @ qv
        top = torch.topk(sims, min(args.topk, KB.shape[0])).indices.tolist()
        # hard support: object literally present in retrieved evidence
        for t in top:
            if obj in kb_sets[t]:
                return 1.0
        # graded fallback: max cosine of object to retrieved sentence vectors
        ov = mean_vec([obj])
        if ov is None:
            return 0.0
        return 0.5 * float(max(float(KB[t] @ ov) for t in top))

    # precompute pool vectors for ADVERSARIAL plausible-false selection
    pool = [w for w in content_pool if w in w2i]
    pool_vecs = F.normalize(E[torch.tensor([w2i[w] for w in pool], device=dev)], p=2, dim=-1)

    def plausible_false(ctx, obj):
        """A topically-related-but-wrong object (high cos to context) — a plausible
        hallucination, not a random word."""
        qv = mean_vec(ctx)
        if qv is None:
            return random.choice(pool)
        sims = (pool_vecs @ qv)
        top = torch.topk(sims, min(60, len(pool))).indices.tolist()
        cset = set(ctx)
        cands = [pool[t] for t in top if pool[t] != obj and pool[t] not in cset]
        return random.choice(cands[:40]) if cands else random.choice(pool)

    pair_hit = pair_n = 0
    true_supported = false_accept = 0
    for ctx, obj, src in facts:
        fobj = plausible_false(ctx, obj)   # ADVERSARIAL: topically related, wrong
        st = support(ctx, obj)
        sf = support(ctx, fobj)
        pair_n += 1
        pair_hit += int(st > sf)
        true_supported += int(st >= 1.0)
        false_accept += int(sf >= 1.0)

    n = max(pair_n, 1)
    print("\n" + "─" * 58)
    print(f"  FACT-CHECK ORACLE (retrieval + support)  KB={len(kb_words)}")
    print(f"  pairwise (true > false)  : {pair_hit/n*100:5.1f}%   (chance 50%)")
    print(f"  grounding recall (true supported)     : {true_supported/n*100:5.1f}%")
    print(f"  false-accept (false wrongly supported): {false_accept/n*100:5.1f}%")
    print("─" * 58)
    if pair_hit / n > 0.75:
        print("  VERDICT: fact-check oracle SEPARATES true from false → factual grounding viable.")
    elif pair_hit / n > 0.6:
        print("  VERDICT: partial signal → oracle works but needs sharpening.")
    else:
        print("  VERDICT: weak → retrieval alone insufficient; need KG/entailment.")


if __name__ == "__main__":
    main()
