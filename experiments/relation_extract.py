"""
relation_extract.py — de-risk the RISKIEST link: can we extract a usable relation
graph from the brain's text WITHOUT an LLM?

Uses spaCy dependency parsing (no LLM) to pull (head, relation, tail) triples:
  - SVO:      nsubj --VERB--> dobj/attr/pobj     e.g. (mitochondria, produce, atp)
  - copula:   X is/are a Y                        e.g. (mitochondria, is, organelle)

Then measures whether the graph is good enough to reason over:
  1. Extraction quality — sample triples + stats (triples, unique concepts/relations).
  2. COVERAGE — for held-out ARC questions, are the question-concepts AND the
     correct-answer concept present in the graph, and connected within k hops?
     If coverage is near zero, extraction is the wall (ingest more / better parse).
     If decent, a grounded path-reasoner is viable.

Usage:
    .venv/bin/python experiments/relation_extract.py --wiki 3000 --sample 300
"""
from __future__ import annotations
import argparse, os, re, sys, time, random
from collections import defaultdict, Counter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_LETTERS = {"A", "B", "C", "D"}


def _load_spacy():
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    # keep parser + tagger; add a light lemmatizer via .lemma_ needs lemmatizer,
    # so re-enable a rule-based one cheaply:
    if "lemmatizer" not in nlp.pipe_names:
        try:
            nlp.add_pipe("lemmatizer", config={"mode": "rule"}).initialize()
        except Exception:
            pass
    return nlp


def _concept(tok):
    """Normalise a token to a concept string (lemma of noun-ish head)."""
    t = (tok.lemma_ or tok.text).lower().strip()
    return t if re.match(r"^[a-z][a-z\-]{1,}$", t) else None


def extract_triples(nlp, sentences, max_sents):
    triples = []
    n = 0
    for doc in nlp.pipe(sentences[:max_sents], batch_size=64):
        n += 1
        for tok in doc:
            if tok.pos_ != "VERB" and tok.lemma_ != "be":
                continue
            subs = [c for w in tok.children if w.dep_ in ("nsubj", "nsubjpass")
                    for c in [_concept(w)] if c]
            if not subs:
                continue
            # objects: direct, attribute (copula), prep-object
            objs = []
            rel = tok.lemma_.lower()
            for w in tok.children:
                if w.dep_ in ("dobj", "attr", "acomp", "oprd"):
                    c = _concept(w)
                    if c:
                        objs.append((rel, c))
                elif w.dep_ == "prep":
                    for p in w.children:
                        if p.dep_ == "pobj":
                            c = _concept(p)
                            if c:
                                objs.append((f"{rel}_{w.text.lower()}", c))
            for s in subs:
                for r, o in objs:
                    if s != o:
                        triples.append((s, r, o))
    return triples, n


def _words(t):
    return set(re.findall(r"[a-z]{3,}", t.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", type=int, default=3000)
    ap.add_argument("--max_sents", type=int, default=40000)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--hops", type=int, default=2)
    args = ap.parse_args()
    random.seed(0)

    from datasets import load_dataset
    print("[*] loading spaCy …")
    nlp = _load_spacy()

    # ── corpus: declarative science text (wikipedia + ARC/MMLU answer statements) ──
    print("[*] building sentence corpus …")
    sents = []
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        for i, r in enumerate(ds):
            if i >= args.wiki:
                break
            for s in re.split(r"(?<=[.!?])\s+", r["text"][:2000]):
                if 5 <= len(s.split()) <= 40:
                    sents.append(s)
    except Exception as e:
        print("  [!] wiki:", e)
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        for r in ds:
            # turn "Q? correct." into a statement-ish fragment for extraction
            sents.append(r["question"] + " " + " ".join(r["choices"]["text"]))
    except Exception as e:
        print("  [!] arc:", e)
    random.shuffle(sents)
    print(f"[*] {len(sents):,} candidate sentences")

    t0 = time.time()
    triples, n_sent = extract_triples(nlp, sents, args.max_sents)
    print(f"[*] parsed {n_sent:,} sentences → {len(triples):,} triples  ({time.time()-t0:.0f}s)")

    # graph
    adj = defaultdict(set)          # concept -> set(concept)
    rels = Counter()
    concepts = set()
    for h, r, t in triples:
        adj[h].add(t); adj[t].add(h)
        rels[r] += 1; concepts.add(h); concepts.add(t)
    print(f"[*] graph: {len(concepts):,} concepts, {sum(len(v) for v in adj.values())//2:,} edges")

    print("\n[*] sample triples (quality eyeball):")
    for h, r, t in random.sample(triples, min(20, len(triples))):
        print(f"    ({h}) --{r}--> ({t})")
    print("\n[*] most common relations:", [r for r, _ in rels.most_common(12)])

    # ── coverage / connectivity on held-out ARC ──────────────────────────────
    def reachable(starts, goals, k):
        frontier = set(starts) & concepts
        if not frontier or not (set(goals) & concepts):
            return False
        seen = set(frontier)
        for _ in range(k):
            nxt = set()
            for c in frontier:
                nxt |= adj[c]
            if nxt & set(goals):
                return True
            nxt -= seen; seen |= nxt; frontier = nxt
            if not frontier:
                break
        return False

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rows = []
    for r in ds:
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        ans = r["answerKey"].strip().upper()
        if set(labels).issubset(_LETTERS) and ans in labels:
            rows.append((r["question"], texts[labels.index(ans)],
                         [t for l, t in zip(labels, texts) if l != ans]))
    random.shuffle(rows); rows = rows[:args.sample]

    q_present = ans_present = both = connected = correct_only = 0
    path_hits = 0; n = 0
    for q, correct, distractors in rows:
        n += 1
        qc = _words(q) & concepts
        ac = _words(correct) & concepts
        if qc: q_present += 1
        if ac: ans_present += 1
        if qc and ac:
            both += 1
            if reachable(qc, ac, args.hops):
                connected += 1
        # discrimination signal: correct connected but distractors not?
        d_conn = any(reachable(qc, _words(d) & concepts, args.hops)
                     for d in distractors if _words(d) & concepts)
        c_conn = bool(qc and ac and reachable(qc, ac, args.hops))
        if c_conn and not d_conn:
            path_hits += 1

    print("\n" + "─" * 60)
    print(f"  RELATION-GRAPH COVERAGE — ARC-Challenge test, {n} q  (k={args.hops} hops)")
    print("─" * 60)
    print(f"  question concept in graph : {q_present/n*100:5.1f}%")
    print(f"  answer concept in graph   : {ans_present/n*100:5.1f}%")
    print(f"  both present              : {both/n*100:5.1f}%")
    print(f"  Q→correct connected ≤{args.hops}    : {connected/n*100:5.1f}%")
    print(f"  correct-connected & NOT distractor-connected : {path_hits/n*100:5.1f}%  (naive disc. signal)")
    print("─" * 60)
    if connected / n > 0.25:
        print("  VERDICT: graph reaches answers → grounded path-reasoner viable.")
    else:
        print("  VERDICT: coverage low → extraction/ingestion is the wall to fix first.")


if __name__ == "__main__":
    main()
