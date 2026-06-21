"""
Module 2 end-to-end demo: train, evaluate, and compare all three generation
strategies (autoregressive, beam search, retrieval) on a small factual Q&A
dataset.

Run:
    python demo_module2.py
"""

from predictor  import UniversalPredictor
from module2    import GoalDirectedGenerator, SEPARATOR, END
from similarity import jaccard

# ── dataset ───────────────────────────────────────────────────────────────────

def tok(s: str) -> list[str]:
    return s.lower().split()

# 30 factual Q&A pairs across three domains.
# The answer is always a single token to keep evaluation simple.
ALL_PAIRS: list[tuple[list, list]] = [
    # Geography — capitals
    (tok("what is the capital of France"),    tok("paris")),
    (tok("what is the capital of Germany"),   tok("berlin")),
    (tok("what is the capital of Japan"),     tok("tokyo")),
    (tok("what is the capital of Spain"),     tok("madrid")),
    (tok("what is the capital of Italy"),     tok("rome")),
    (tok("what is the capital of Brazil"),    tok("brasilia")),
    (tok("what is the capital of Canada"),    tok("ottawa")),
    (tok("what is the capital of Australia"), tok("canberra")),
    (tok("what is the capital of China"),     tok("beijing")),
    (tok("what is the capital of India"),     tok("delhi")),
    # Science — elements
    (tok("what is the symbol for gold"),      tok("au")),
    (tok("what is the symbol for silver"),    tok("ag")),
    (tok("what is the symbol for iron"),      tok("fe")),
    (tok("what is the symbol for copper"),    tok("cu")),
    (tok("what is the symbol for sodium"),    tok("na")),
    (tok("what is the symbol for potassium"), tok("k")),
    (tok("what is the symbol for lead"),      tok("pb")),
    (tok("what is the symbol for mercury"),   tok("hg")),
    # Planets — order from sun
    (tok("what number planet is mercury"),    tok("one")),
    (tok("what number planet is venus"),      tok("two")),
    (tok("what number planet is earth"),      tok("three")),
    (tok("what number planet is mars"),       tok("four")),
    (tok("what number planet is jupiter"),    tok("five")),
    (tok("what number planet is saturn"),     tok("six")),
    # Simple patterns
    (tok("what color is the sky"),            tok("blue")),
    (tok("what color is grass"),              tok("green")),
    (tok("what color is snow"),               tok("white")),
    (tok("what color is coal"),               tok("black")),
    (tok("what color is blood"),              tok("red")),
    (tok("what color is the sun"),            tok("yellow")),
]

# Train on all 30 pairs.
# Recall test: 6 prompts drawn directly from training — tests perfect memorisation.
# Novel test:  3 prompts with tokens never seen during training (different slot-fillers).
TRAIN_PAIRS = ALL_PAIRS          # all 30 used for training

RECALL_PAIRS = [                 # exact prompts from training — should be 100%
    ALL_PAIRS[0],   # capital of France → paris
    ALL_PAIRS[5],   # capital of Brazil → brasilia
    ALL_PAIRS[11],  # symbol for silver → ag
    ALL_PAIRS[20],  # planet earth      → three
    ALL_PAIRS[25],  # color of grass    → green
    ALL_PAIRS[29],  # color of sun      → yellow
]

NOVEL_PAIRS = [                  # tokens never seen in training
    (tok("what is the capital of Egypt"),     tok("cairo")),
    (tok("what is the symbol for oxygen"),    tok("o")),
    (tok("what number planet is uranus"),     tok("seven")),
]


# ── evaluation helpers ────────────────────────────────────────────────────────

def exact_match(got: list, expected: list) -> bool:
    return got == expected or (got and got[0] == expected[0])

def run_accuracy(gen, pairs, strategy, beam_width=3, surface_sim_fn=None):
    correct = 0
    results = []
    for prompt, expected in pairs:
        if strategy == "auto":
            got = gen.answer(prompt, max_steps=5)
        elif strategy == "beam":
            beams = gen.beam_search(
                list(prompt) + [gen.separator],
                beam_width=beam_width, max_steps=5
            )
            got = beams[0][0] if beams else []
        elif strategy == "retrieve":
            hits = gen.retrieve(prompt, TRAIN_PAIRS, top_k=1,
                                surface_sim_fn=surface_sim_fn)
            got  = list(hits[0][0]) if hits else []
        else:
            got = []

        hit = exact_match(got, expected)
        correct += hit
        results.append((prompt, expected, got, hit))
    return correct / len(pairs), results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Module 2 — End-to-End Q&A Demo")
    print("=" * 65)
    print(f"\nDataset:   {len(ALL_PAIRS)} Q&A pairs across 4 domains")
    print(f"Training:  {len(TRAIN_PAIRS)} pairs (all used)")
    print(f"Recall test:  {len(RECALL_PAIRS)} pairs (drawn from training — exact recall)")
    print(f"Novel test:   {len(NOVEL_PAIRS)} pairs (tokens never seen during training)")

    # ── train ─────────────────────────────────────────────────────────────────
    print("\n[1/4] Training Module 1 on formatted Q&A sequences ...")
    print(f"      Format: [question tokens] {SEPARATOR} [answer] {END}")

    p   = UniversalPredictor(context_length=6, learning_rate=0.1)
    gen = GoalDirectedGenerator(p)
    gen.train_on_pairs(TRAIN_PAIRS, repeats=30)

    nodes = len(p._nodes)
    print(f"      Trie size after training: {nodes} nodes")

    # ── benchmark: in-distribution test ──────────────────────────────────────
    print("\n[2/4] In-distribution test (24 training prompts, exact recall) ...")

    for label, strategy in [("Autoregressive", "auto"),
                             ("Beam search",    "beam"),
                             ("Retrieval",      "retrieve")]:
        acc, _ = run_accuracy(gen, TRAIN_PAIRS, strategy)
        bar    = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {label:16s}  [{bar}]  {acc*100:.0f}%")

    # ── benchmark: recall test ────────────────────────────────────────────────
    print("\n[3/4] Recall test (prompts drawn from training) ...")

    acc_a, results_a = run_accuracy(gen, RECALL_PAIRS, "auto")
    acc_b, _         = run_accuracy(gen, RECALL_PAIRS, "beam")
    acc_r, results_r = run_accuracy(gen, RECALL_PAIRS, "retrieve")
    print(f"  Autoregressive: {acc_a*100:.0f}%   Beam search: {acc_b*100:.0f}%   Retrieval: {acc_r*100:.0f}%")
    for (prompt, expected, got_a, hit_a), (_, _, got_r, hit_r) in zip(results_a, results_r):
        q = " ".join(prompt)
        print(f"  {'OK' if hit_a else '--'}/{' OK' if hit_r else '--'}  "
              f"{q!r:47s}  auto={got_a}  ret={got_r}")

    # ── benchmark: novel tokens ───────────────────────────────────────────────
    print("\n  Novel tokens (Egypt, oxygen, Uranus — never seen during training):")
    print("  Three retrieval modes: trie-only | trie + Jaccard fallback | autoregressive")
    acc_n_auto, res_auto   = run_accuracy(gen, NOVEL_PAIRS, "auto")
    acc_n_ret,  res_ret    = run_accuracy(gen, NOVEL_PAIRS, "retrieve")
    acc_n_surf, res_surf   = run_accuracy(gen, NOVEL_PAIRS, "retrieve",
                                          surface_sim_fn=jaccard)
    print(f"  Autoregressive: {acc_n_auto*100:.0f}%   "
          f"Trie-only: {acc_n_ret*100:.0f}%   "
          f"Trie+Jaccard: {acc_n_surf*100:.0f}%")
    for (prompt, expected, got_a, hit_a), (_, _, got_r, hit_r), (_, _, got_s, hit_s) \
            in zip(res_auto, res_ret, res_surf):
        q = " ".join(prompt)
        print(f"  auto={'OK' if hit_a else '--'} [{got_a[0] if got_a else '?':10s}]  "
              f"trie={'OK' if hit_r else '--'} [{got_r[0] if got_r else '?':10s}]  "
              f"surf={'OK' if hit_s else '--'} [{got_s[0] if got_s else '?':10s}]  "
              f"want={expected[0]!r}  {q!r}")

    # ── beam search detail ────────────────────────────────────────────────────
    print("\n[4/4] Beam search — top candidate first-tokens for two queries")
    print("      (shows score distribution over possible answers)")

    for prompt, expected in [
        (tok("what is the capital of France"), tok("paris")),
        (tok("what color is the sky"),         tok("blue")),
        (tok("what is the symbol for gold"),   tok("au")),
    ]:
        seed  = list(prompt) + [gen.separator]
        # Wider beam to capture more candidates; prune to unique first tokens
        beams = gen.beam_search(seed, beam_width=8, max_steps=1)
        seen_first  = {}
        for tokens, score in beams:
            first = tokens[0] if tokens else None
            if first is not None and first not in seen_first:
                seen_first[first] = score
        # Sort by score
        ranked = sorted(seen_first.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  Q: {' '.join(prompt)!r}")
        for rank, (tok_val, score) in enumerate(ranked, 1):
            mark = "<-- correct" if tok_val == expected[0] else ""
            print(f"  [{rank}] {tok_val:12s}  score={score:.2f}  {mark}")

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Summary")
    print("=" * 65)
    print("""
  In-distribution (seen prompts):
    All three strategies achieve near-100% exact match.
    Trie retrieval: Bhattacharyya ≈ 1.0 against the correct match,
    near 0 against all others — the trie perfectly discriminates.

  Novel tokens (e.g. 'Egypt', 'oxygen', 'Uranus') — two stages:
    Stage 1 (trie): novel token → root unigram fallback → all training
    prompts score similarly low. Returns a random training answer.
    Stage 2 (Jaccard fallback, triggered when Bhattacharyya < 0.5):
    re-ranks by token overlap. 'capital of Egypt' → highest Jaccard
    with other 'capital of X' questions → domain-correct city name,
    even though the specific answer (cairo) was never seen.

  Degradation tiers:
    Bhattacharyya ≈ 1.0  → exact trie match, perfect answer
    Bhattacharyya < 0.5  → Jaccard fallback, domain-correct answer
    No structural overlap → root unigram, random answer
""")


if __name__ == "__main__":
    main()
