# Project Roadmap

## Status: Module 1 complete, Module 2 in progress

---

## Module 1 — Done

- [x] Rewrite predictor to O(k) prefix trie (replaced O(n²) flat-nodes)
- [x] CTW-style credibility blend with KT smoothing
- [x] Confidence-proportional degradation (`lr_down = lr × (1 + c/C_MAX)`)
- [x] Forest ensemble (heterogeneous k, dropout, stagger, inter-tree credibility)
- [x] IEEE benchmark suite with 6 standard + 4 drift + scaling + ablation tables
- [x] Electricity real-world concept-drift benchmark (45K steps)
- [x] Fix log-loss computation (was returning 1348 bits/symbol)
- [x] Fix significance test (per-step Wilcoxon, not 5-seed same-sequence)
- [x] README rewrite to match actual architecture

---

## Module 2 — In Progress

- [x] `module2.py` skeleton: `GoalDirectedGenerator` class
- [x] Autoregressive completion (`complete()`)
- [x] Beam search generation (`beam_search()`)
- [x] Retrieval-based answering (`retrieve()`)
- [x] End-to-end demo: train on formatted Q&A sequence, answer novel queries (`demo_module2.py`)
- [x] Two-stage retrieval: Bhattacharyya (exact match) + Jaccard fallback (novel tokens → domain-correct)
- [ ] Benchmark Module 2 on a factual retrieval task (e.g., SimpleQuestions subset)
- [ ] Context-window extension: sliding-window attention over long prompts
- [ ] Evaluate beam search vs. greedy vs. retrieval on short-answer tasks

---

## Paper (IEEE submission)

- [x] All benchmark tables generated (`ieee_tables/`)
- [ ] Write paper body (target: IEEE TPAMI or similar)
  - [ ] Abstract: concept-drift framing + key numbers (97% drift, DNA best, Forest 83.7% Electricity)
  - [ ] Introduction: the forgetting problem in online prediction
  - [ ] Related work: CTW, PPM-D, LSTM, ADWIN-based approaches
  - [ ] Architecture section: trie, CTW blend, credibility update, confidence-proportional degradation
  - [ ] Experiments section: wire in the LaTeX tables from `ieee_tables/`
  - [ ] Ablation section: table 04 with narrative
  - [ ] Module 2 section: once benchmark exists
  - [ ] Conclusion
- [ ] Camera-ready figures (currently PDFs in `ieee_tables/`)

---

## Near-term priorities (next session)

1. ~~Module 2 end-to-end demo~~ — DONE (`demo_module2.py`)
2. ~~Module 2 two-stage retrieval~~ — DONE (Jaccard fallback for novel tokens)
3. Paper abstract + introduction — the narrative frame determines what experiments to add
4. Module 2 benchmark on a factual retrieval task (SimpleQuestions subset)
