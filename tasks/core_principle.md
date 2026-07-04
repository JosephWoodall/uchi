# This Repo's North Star

**One sentence:** Uchi is a phenomenal recall machine — a credibility-weighted
context trie — fused with a grounded reasoner: a 256D vector-symbolic manifold
that synthesizes answers to out-of-distribution questions it has never seen, by
reasoning *forward from evidence retrieved from its own brain*, never inventing
facts that aren't grounded there.

## The Core Intuition

Two capabilities, one system, no LLM:

1. **Recall (the trie).** CTW-style multi-order blending over a prefix trie,
   credibility-updated by MWU. When the answer has been seen, exact-match
   retrieval returns it with calibrated confidence. This is already excellent
   and must be preserved untouched.

2. **Grounded generalization (the manifold).** When the trie backs off — i.e.
   the question is out of distribution — the system must not collapse to a prior
   (random) or free-generate (confabulate). Instead it: (a) encodes the query
   into the 256D manifold, (b) retrieves the nearest *real* brain facts, (c)
   binds query⊗evidence via HRR, and (d) lets the SSM policy/value heads
   synthesize an answer **constrained to that retrieved evidence**. If no
   evidence grounds an answer, Uchi abstains rather than invents.

The leash is the point: generalization is only trustworthy because every
synthesized token is anchored to something actually in the brain. Reasoning
forward from grounded evidence ≠ hallucination.

## What Changed From v0.3.0 (and why)

The prior North Star claimed the trie "generates by extrapolation." The code
proved it does pure exact-n-gram retrieval with backoff — no semantic
extrapolation exists (`use_similarity_fallback=False` everywhere;
`semantic_index.py` never wired in). That is why MMLU sat at 22.5% ≈ random on
OOD questions. We reject the old assumption. The SSM is promoted from
confidence-scaffold to **grounded co-generator**, governed by trie credibility:
deep trie match → trie answers; trie backoff → retrieval-grounded SSM synthesis.

## State-of-the-Art Grounding

- **CTW** — Willems et al. 1995. The recall trie (unchanged).
- **MWU** — Arora, Hazan, Kale 2012. Credibility = depth-selection regret bound.
- **kNN-LM** — Khandelwal et al. 2020. Interpolate parametric model with a
  datastore, weighted by retrieval confidence. Uchi inverts the usual default:
  trie (datastore) dominates in-distribution; SSM (parametric) takes over on
  backoff. The mixing weight λ is a function of trie credibility / match depth.
- **HRR / VSA** — Plate 1995. Circular-convolution binding of query⊗evidence;
  enables grounded analogical synthesis (King−Man+Woman≈Queen) with zero
  representational cost.
- **Hard-negative contrastive learning** — the manifold learns to separate a
  correct answer from its distractors (InfoNCE with option-level negatives).
- **MCTS / UCB1** — Kocsis & Szepesvári 2006. Deliberation over candidates.

## Why This Beats Alternatives

| Alternative | Why Rejected |
|-------------|--------------|
| Transformer LLM | 100B+ params, opaque, confabulates, GPU clusters |
| Pure trie (v0.3.0) | Exact-match only → random on OOD; cannot synthesize |
| Free parametric SSM generator | Confabulates; not grounded in the brain |
| KNN retrieval alone | No synthesis; answer must exist verbatim |

Uchi is the only design that recalls verbatim when it can AND synthesizes
grounded answers when it can't — online, deterministic-where-possible, no GPU,
and constitutionally unable to assert facts it cannot retrieve.

## Drift Check

Every change must answer: **Does this improve grounded generalization without
letting the system assert anything it cannot trace back to brain content?** A
change that lets the SSM emit ungrounded tokens (confabulation) violates the
North Star. A change that improves recall, retrieval quality, manifold
discrimination, or the grounding/abstention gate aligns with it.
