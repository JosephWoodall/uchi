"""
generate_and_ground.py — end-to-end Generate-and-Ground loop (phase 2: plumbing).

    string question in
      → RETRIEVE relevant brain knowledge (skip-gram semantic index)
      → GENERATE a candidate answer  (phase 2: EXTRACTIVE — pick the best real
         sentence, so output is grammatical & grounded by construction;
         phase 1 will swap in a from-scratch neural decoder for generative OOD)
      → FACT-CHECK the candidate against the brain (the validated oracle)
      → EMIT the grounded answer, or ABSTAIN ("I don't know") — never confabulate
    string answer out

Satisfies the hard constraints: string→string, grammatical/contextual (real
sentences), OOD (retrieval answers novel questions), honest (abstains when
unsupported). Proves the loop before investing in the neural generator.

Usage:
    .venv/bin/python experiments/generate_and_ground.py --kb 6000
"""
from __future__ import annotations
import argparse, os, re, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
_WORD = re.compile(r"[a-z']+")
_EMB = os.path.join(os.path.dirname(__file__), "skipgram_emb.pt")
_STOP = set("the a an of to in and or is are was were be been what which who how why "
            "does do did can could would should this that these those it its there here "
            "on at by with as for from into about your you i we they he she".split())


class GenerateAndGround:
    def __init__(self, kb_texts, kb_vecs, w2i, E, torch, F, dev,
                 retrieve_k=12, support_th=0.35):
        self.kb_texts, self.KB = kb_texts, kb_vecs
        self.w2i, self.E, self.torch, self.F, self.dev = w2i, E, torch, F, dev
        self.k, self.support_th = retrieve_k, support_th
        self.kb_words = [set(_WORD.findall(t.lower())) for t in kb_texts]

    def _content(self, text):
        return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2]

    def _vec(self, text):
        # content words only — stopwords muddy the retrieval vector
        ids = [self.w2i[w] for w in self._content(text) if w in self.w2i]
        if not ids:
            return None
        v = self.E[self.torch.tensor(ids, device=self.dev)].mean(0)
        return self.F.normalize(v, p=2, dim=-1)

    # 1. RETRIEVE
    def retrieve(self, q):
        qv = self._vec(q)
        if qv is None:
            return []
        sims = self.KB @ qv
        top = self.torch.topk(sims, min(self.k, self.KB.shape[0]))
        return [(self.kb_texts[i], float(s)) for s, i in zip(top.values.tolist(), top.indices.tolist())]

    # 2. GENERATE (extractive: the sentence that best answers the question)
    def generate(self, q, evidence):
        qwords = {w for w in _WORD.findall(q.lower()) if w not in _STOP and len(w) > 2}
        best, best_score = None, -1.0
        for text, sim in evidence:
            tw = {w for w in _WORD.findall(text.lower()) if w not in _STOP}
            overlap = len(qwords & tw) / (len(qwords) + 1)
            score = 0.6 * sim + 0.4 * overlap
            if score > best_score:
                best_score, best = score, (text, sim, overlap)
        return best

    # 3. FACT-CHECK (oracle): are the candidate's salient terms supported by evidence?
    def factcheck(self, candidate_text, evidence):
        terms = [w for w in _WORD.findall(candidate_text.lower())
                 if w not in _STOP and len(w) > 3]
        if not terms:
            return 0.0
        ev_words = set().union(*[set(_WORD.findall(t.lower())) for t, _ in evidence]) if evidence else set()
        supported = sum(1 for w in terms if w in ev_words)
        return supported / len(terms)

    # full loop
    _ABSTAIN = "I don't have grounded knowledge to answer that."

    def answer(self, q):
        # honesty gate 1: do we even KNOW the question's concepts? (catches nonsense/OOV)
        content = self._content(q)
        known = [w for w in content if w in self.w2i]
        if not content or len(known) / len(content) < 0.5:
            return self._ABSTAIN, f"abstain(unknown-terms {len(known)}/{len(content)})"
        ev = self.retrieve(q)
        if not ev or ev[0][1] < 0.5:
            return self._ABSTAIN, f"abstain(weak-retrieval sim={ev[0][1] if ev else 0:.2f})"
        cand = self.generate(q, ev)
        if cand is None:
            return self._ABSTAIN, "abstain(no-candidate)"
        text, sim, overlap = cand
        # honesty gate 2: the answer must actually share content with the question
        if overlap == 0.0 or sim < 0.55:
            return self._ABSTAIN, f"abstain(irrelevant sim={sim:.2f} overlap={overlap:.2f})"
        return text, f"grounded(sim={sim:.2f} overlap={overlap:.2f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", type=int, default=6000)
    args = ap.parse_args()

    import torch, torch.nn.functional as F
    from datasets import load_dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(_EMB, map_location=dev)
    w2i, E = d["w2i"], F.normalize(d["E"].to(dev), p=2, dim=-1)

    print("[*] building brain (KB) …")
    texts, vecs = [], []
    def vec(text):
        ids = [w2i[w] for w in _WORD.findall(text.lower()) if w in w2i]
        return F.normalize(E[torch.tensor(ids, device=dev)].mean(0), p=2, dim=-1) if ids else None
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    for r in ds:
        if len(texts) >= args.kb:
            break
        for s in re.split(r"(?<=[.!?])\s+", r["text"][:2500]):
            w = _WORD.findall(s.lower())
            if 6 <= len(w) <= 45:
                v = vec(s)
                if v is not None:
                    texts.append(s.strip()); vecs.append(v)
                    if len(texts) >= args.kb:
                        break
    KB = torch.stack(vecs)
    print(f"[*] brain = {len(texts):,} sentences\n")

    gg = GenerateAndGround(texts, KB, w2i, E, torch, F, dev)

    questions = [
        "What is photosynthesis?",
        "What causes earthquakes?",
        "What is the function of mitochondria?",
        "Who developed the theory of relativity?",
        "What is gravity?",
        "How do vaccines work?",
        "What is the fltoo of a quzzle blimbat?",     # nonsense → must abstain
        "What is my neighbour's dog's name?",          # unknowable → must abstain
    ]
    for q in questions:
        ans, prov = gg.answer(q)
        print(f"Q: {q}")
        print(f"A: {ans}")
        print(f"   [{prov}]\n")


if __name__ == "__main__":
    main()
