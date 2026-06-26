# This Repo's North Star

**One sentence:** Uchi is a credibility-weighted context trie that learns to predict any sequence without neural weights, using MCTS to generate coherent multi-step futures and a thin GRU value head to grade its own confidence.

## The Core Intuition

Context collision is the brittleness of deterministic sequence modeling: when two different past patterns map to the same context window, the predictor has no principled way to pick. Uchi solves this with CTW-style multi-order blending — it doesn't pick a single order; it averages across all valid context lengths, weighted by per-node credibility scores that are updated via multiplicative weights update (MWU). The credibility scores are the memory of what worked.

The second insight: pure trie generation produces rigid, repetitive text because it follows the single highest-probability edge. MCTS over the trie explores multiple futures simultaneously, using UCB1 to balance exploitation (high-probability sequences) with exploration (low-probability, potentially better paths). This is the same principle that took AlphaGo from superhuman to god-level — applied to text generation.

## State-of-the-Art Grounding

- **CTW (Context Tree Weighting)** — Willems et al. 1995, IEEE Trans. Inf. Theory. The trie blending strategy is mathematically optimal for stationary sources. Uchi adapts it for online, non-stationary streams.
- **MWU (Multiplicative Weights Update)** — Arora, Hazan, Kale 2012. The credibility update is a regret-minimizing online learning algorithm with provable no-regret guarantees over adversarial sequences.
- **MCTS with UCB1** — Kocsis & Szepesvári 2006. Applied here to sequence generation, yielding better-than-greedy token selection with O(log n) rollout overhead.
- **GRPO (Group Relative Policy Optimization)** — DeepSeekMath 2024. Used for value head training without a separate critic, making the SSM self-improving from user sentiment signals.

## Why This Architecture Beats Alternatives

| Alternative | Why Rejected |
|-------------|--------------|
| Transformer LLM | 100B+ parameters, opaque, not truly deterministic, requires GPU clusters |
| N-gram model (fixed order) | Context collision at every boundary; no interpolation; brittle |
| Pure RNN/LSTM | Requires training corpus upfront; can't online-learn per conversation turn |
| KNN retrieval | No generative capability; answers must exist verbatim in corpus |

Uchi's trie is the only approach that: (1) learns online from every token seen, (2) generates by extrapolation rather than retrieval, (3) provides exact probability bounds via CTW blending, and (4) requires zero GPU at inference.

## Current Architecture (v0.3.0)

```
User Input
    ↓
OmniRouter.chat()
    ├─ ProceduralMemory (intent: CODE / MATH / SEARCH / CONVERSATIONAL)
    ├─ OmniTokenizer (WordNet synset normalization)
    ├─ AssociativeMemory.query() → CPUVectorMemory cosine search via SSM states
    ├─ SequenceGenerator.generate() → MCTS over CTW trie
    ├─ SSM value head → hallucination gating
    └─ GRPO update → SSM self-improves from reward signal

Offline Bootstrap (cold start, once):
    bootstrap_code.py    → 1000 Python stdlib function patterns
    bootstrap_wikidata.py → 25 Wikipedia topic triples via SPARQL
```

## Drift Check

Every code change must answer: **Does this make the trie's predictions more accurate or the routing more precise?** If a change adds neural weights as the primary generator (rather than a confidence scaffold), it violates the North Star.
