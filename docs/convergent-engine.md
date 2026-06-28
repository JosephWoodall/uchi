# Convergent Engine (MCTS)

> **Python users:** the convergent engine runs automatically when you call `u.ask("...")` on any `Uchi` instance. No direct configuration is required for standard use. See [Python API →](python-api.md)

---



The `ConvergentEngine` adds a deliberative "System 2" search mechanism on top of the deterministic sequence predictor. By simulating multiple generation paths via Monte Carlo Tree Search (MCTS), filtering them through binary Oracles, and ranking them via SSM Vector Geometry, Uchi transforms probabilistic hallucinations into verified intelligence.

## Core Mechanisms

### 1. Adaptive MCTS Budgeting
Instead of guessing the next token instantly, the engine simulates full delivery-length responses up to a maximum budget (default 50 rollouts). It continuously measures Token-Level Jaccard Diversity to dynamically halt the simulation early when consensus is reached.

### 2. Tiered Oracles
Oracles provide absolute binary rejection gates to kill invalid candidates:
- **TieredCodeOracle**: Adjusts strictness based on the training state (from basic syntax parsing to full REPL compilation).
- **CoherenceOracle**: Filters conversational noise by failing repetitive loops, ultra-short stubs, and low SSM value-head scores.

### 3. Geometric Vector Ranking
All surviving candidates are embedded via the SSM into an $\mathbb{R}^{64}$ space. The candidate geometrically closest (highest cosine similarity) to the original Query Vector is chosen. Tools and skills exist in this exact same vector space and will dynamically intercept generation if they beat the text candidates by a predefined relative margin.

### 4. Background Contrastive Update
The vector geometry is trained post-hoc through an asynchronous `contrastive_update`, pushing Query/Response pairs closer together and punishing invalid tool correlations.

### Realistic Use Cases

1. **Complex Algorithm Generation:** A user asks for a Python script to parallelize web scraping. The engine runs 50 MCTS rollouts, the Oracle destroys 49 that have syntax errors or fail to compile, and the 1 surviving perfect script is ranked and returned.
2. **Ambiguous Conversational Routing:** A user asks "How do I reverse a string?". The engine realizes the `WebSearch` vector is completely irrelevant, but the `CodeEngine` tool is geometrically very close to the query intent. It routes the prompt cleanly to the dedicated code logic.
3. **Self-Correction on Factual Errors:** The Trie generates multiple conversational responses to a historical query. The Coherence Oracle filters out recursive stammers, and the SSM Geometric Ranking picks the response most historically correlated with the question's concept vector.

### The Ultimate Benefit
The Convergent Engine fundamentally breaks the autoregressive curse of single-shot text generation. By marrying fast deterministic sampling with strict algorithmic verification and topological geometry, Uchi achieves verified intelligence and complex multi-tool routing natively—without relying on external Large Language Models.
