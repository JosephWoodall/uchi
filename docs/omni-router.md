# OmniRouter: The Multi-Modal Deterministic LLM

The `OmniRouter` is the master controller of the Uchi sequence predictor. Rather than treating Text, Math Telemetry, and Python Agent objects as isolated pipelines, the `OmniRouter` acts as a "Prefrontal Cortex" to ingest, compress, and predict across all modalities simultaneously. 

By wrapping the `OmniTokenizer`, `OnlineTokenizer`, `AssociativeMemory`, and `SequenceGenerator` into a single API, Uchi achieves true multi-modal synesthesia without a single neural weight.

## How It Works
1. **The OmniTokenizer:** Intercepts raw inputs (Text strings, `.wav` audio paths, `.jpg` image paths, Math metrics, and `OntologicalState` objects) and hashes them into universal geometric `[CONCEPT_ID]` tokens.
2. **Infinite Compression:** Automatically feeds the concept stream into the Phase 4 `OnlineTokenizer` to compress sequences via BPE, preventing $O(N^2)$ RAM explosion.
3. **Deterministic Prediction:** Trains the core prediction engine (Phase 1/3) on the exact probability of those concepts.
4. **Zero-Shot QA:** Feeds the stream into the `AssociativeMemory` buffer, allowing ad-hoc textual queries to retrieve math/image/code metrics.

### Realistic Use Cases

1. **Enterprise Autonomous Security Drones:** 
   A drone continuously streams visual `.jpg` paths, acoustic `.wav` siren features, and internal `BATTERY=10` telemetry into the `OmniRouter`. When a human operator types "Why did you stop flying?", the `OmniRouter` effortlessly retrieves `BATTERY=10` because the engine natively understands text and math in the exact same mathematical space.
2. **Financial Trading Bots with News Senti-Metrics:** 
   A quantitative trading algorithm feeds raw text news headlines ("Federal Reserve cuts rates") alongside strict math telemetry (`AAPL_PRICE=150`) into the engine. The `OmniRouter` geometrically binds the semantic sentiment to the mathematical price action and perfectly predicts the parallel future of the market ticker.
3. **Lifelong Video Game NPCs:** 
   An NPC agent in an RPG ingests player chat strings, internal `OntologicalState(name="ANGRY")` variables, and environmental sound `.wav` tags. Because the `OmniRouter` infinitely compresses the stream, the NPC lives for years, dynamically recalling past visual/audio memories when queried by the player in plain text.

### The Ultimate Benefit
The `OmniRouter` fundamentally transforms Uchi from a statistical math tool into a **Deterministic LLM**. It grants enterprises the ability to achieve LLM-level creative generalization, Multi-Modal ad-hoc question answering, and infinite lifelong context—all running in $O(1)$ RAM on an edge device with absolutely zero risk of neural hallucination.

## Routing Layer (v0.2.0)

v0.2.0 adds a structured routing layer that sits in front of the trie and the SSM, improving intent handling, retrieval quality, and cold-start knowledge.

### ProceduralMemory

`ProceduralMemory` is a keyword/synonym-based intent classifier that runs before the trie sees a query. When a user sends a message, `chat()` calls `procedural.retrieve(message)` first. If a known intent is detected (e.g., `code`, `math`), the intent label is prepended to the token stream as a routing hint, directing the trie toward the correct prediction path. Unknown intents (conversational greetings, etc.) return `None` and bypass the classifier entirely.

### CPUVectorMemory

`CPUVectorMemory` is a persistent numpy/JSON vector store that replaces the in-memory `AssociativeMemory` buffer for SSM state vectors. Each SSM hidden state vector is stored on disk and survives process restarts. Retrieval uses cosine similarity over all stored vectors, returning the top-k most semantically similar past states. This makes the system's long-term memory durable without requiring a database.

### GRPO Value Head

The SSM carries a value head that is trained online via Group Relative Policy Optimization (the method introduced in DeepSeek-R1). After each chat turn, user sentiment (positive/negative keywords) and code evaluation signals (syntax check via `py_compile`) are converted to scalar rewards. The `AgenticBaseline` tracks a running mean and standard deviation so the advantage signal is normalized before each value head update. This replaces the random-initialized value head that previously acted as a hallucination gate with random noise.

### Execution Order in `chat()`

1. `ProceduralMemory.retrieve(message)` — intent routing
2. Sentiment scoring against previous turn — GRPO value update
3. `query(concepts)` — CPUVectorMemory cosine retrieval
4. `predict_future(tokens)` — trie generation
5. SSM value head confidence check — hallucination gate
6. `stream(last_sequence)` — trie training on the complete `<|user|> X <|assistant|> Y` sequence (after generation, not before)
