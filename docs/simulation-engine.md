# Simulation Engine & Advanced Architecture (v0.2.0)

Version 0.2.0 introduced five major architectural extensions that take Uchi from a standard sequence predictor to a lifelong simulation engine.

---

## 1. Mathematical Simulation Stream

The engine learns arithmetic and algebraic rules completely online, extracting structure and intelligence purely through sequence prediction without any pre-training.

```python
from uchi.forest import PredictorForest
from datasets import load_math_corpus

# 1. Initialize an empty forest
forest = PredictorForest(context_length=6, n_trees=5)

# 2. Load a continuous stream of random math equations
math_stream = load_math_corpus(n_chars=20000, seed=42)

# 3. Stream data and let the model grow dynamically
for char in math_stream:
    pred, conf = forest.predict()
    forest.observe(char)
    forest.feedback(char)
```

### Realistic Use Cases
1. **Zero-Config Log Parsing**: Stream raw server logs containing formatted physics (`CPU=45,REQ=100`). Uchi organically learns the mathematical relationships without you writing a single regex parser.
2. **Auto-Completing Dirty Data**: Automatically repair dropped values in financial/crypto transaction streams by predicting the missing mathematical result purely based on sequence geometry.
3. **IoT Sensor Calibration**: Deploy on tiny embedded edge chips where memory is too low for neural networks. If a sensor starts drifting from its physical mathematical ratio, Uchi detects the anomaly instantly.

### The Ultimate Benefit
Phase 1 proves that Uchi acts as a **universal, self-learning physics engine**. If there is *any* underlying logical or mathematical structure in your raw text stream, Uchi will find it and exploit it for perfectly accurate predictions in milliseconds, using zero pre-training.

---

## 2. Optimal Vector Retrieval

When the sequence generator encounters a novel context, it seamlessly falls back to an optimal geometric dense vector query, querying historical distributions rather than guessing randomly.

```python
from uchi.generative import SequenceGenerator
from datasets import load_gutenberg_text

generator = SequenceGenerator(context_length=5, use_vector_retrieval=True)
text = load_gutenberg_text(n_chars=5000)

for char in text:
    generator.observe(char)
    generator.feedback(char)

# If it hits a context it hasn't seen natively, it falls back to the vector index.
generated = generator.generate(max_tokens=100)
```

### Realistic Use Cases
1. **Bulletproof Robustness to Typos**: If an incoming data stream contains a typo (e.g., `vieew_dashboard`), the vector retrieval maps it to the geometrically identical `view_dashboard` history, preventing the engine from losing context.
2. **Eliminating Generative Hallucinations**: When procedurally generating content, hitting a novel sequence usually causes models to hallucinate. Phase 2 ensures the generator always falls back to the nearest logical historical distribution instead of guessing randomly.
3. **Zero-Shot Transfer Learning**: Train the engine on Server A's logs. When deployed on Server B (which has slightly different naming conventions), the vector fallback seamlessly maps the new logs to Server A's learned geometry.

### The Ultimate Benefit
Phase 2 completely eliminates the fatal "amnesia" flaw of classic exact-match algorithms. It guarantees **Graceful Degradation**, ensuring your model stays highly calibrated and predictive even when the real-world data gets messy or entirely novel.

---

## 3. Ontological Process Predictor

Instead of predicting raw strings or numbers, the system can predict discrete, strictly-typed states and actions to model complex workflows, agent loops, or business logic.

```python
from uchi.process import ProcessPredictor, OntologicalState, OntologicalAction

predictor = ProcessPredictor(context_length=4)
s_sort = OntologicalState(name="NeedSort", properties=("unordered_list",))
a_sort = OntologicalAction(name="ExecuteSort", target="QuickSort")

predictor.observe_state(s_sort)
pred_action, conf = predictor.predict_next_action()
predictor.feedback_action(a_sort)
```

### Realistic Use Cases
1. **Autonomous Agent Orchestration**: Instead of paying for a slow LLM to decide what tool an agent should use next, feed the agent's state history into Phase 3. Uchi predicts the exact next function the agent should invoke in 2 milliseconds for free.
2. **E-Commerce Funnel Prediction**: Track users moving through an app using typed states. Uchi learns the progression of user journeys and natively predicts exactly when a user is likely to churn or abandon their cart.
3. **Automated Threat Hunting**: Map hacker "kill chains" to an ontology. The moment a user's network traffic geometrically aligns with a threat actor's sequence, Uchi predicts the server they will attack next, allowing firewalls to preemptively sever access.

### The Ultimate Benefit
Phase 3 transitions Uchi from a text predictor into a **Self-Driving State Machine**. It allows enterprise developers to tightly couple Uchi's hyper-fast predictive intelligence directly to their existing software architecture, objects, and databases.

---

## 4. Plural Simulation Engine

Simulate thousands of lives (or workflows) in parallel, accumulating their separate experiences into a unified meta-prediction intuition.

```python
from uchi.simulation_engine import LifelongSimulationEngine

engine = LifelongSimulationEngine(n_instances=5, context_length=4)
life_1 = ["Eat", "Work", "Sleep", "Eat", "Work", "Sleep"]
life_2 = ["Play", "Eat", "Sleep", "Play", "Eat", "Sleep"]

engine.stream_parallel([life_1, life_2])
plural_prediction, meta_confidence = engine.vote_plural()
```

### Realistic Use Cases
1. **Wisdom of the Crowd Trading**: Simulate 1,000 different quantitative trading agents in parallel and use the pluralistic vote to extract the optimal meta-prediction for stock movements.
2. **A/B Testing Customer Journeys**: Simulate thousands of different UI workflows simultaneously to predict which application flow will yield the highest conversion rate without needing to run live A/B tests.
3. **Multiplayer Game Server Prediction**: Simulate the independent lives of 10,000 players in an MMO simultaneously to predict macro-level server load spikes or economic inflation before it happens.

### The Ultimate Benefit
Phase 4 leverages the aggregated intuition of the entire "crowd" rather than an isolated agent. It allows you to transition from linear sequence prediction to **massively parallel simulation**, extracting high-confidence macro signals from chaotic micro-behaviors.

---

## 5. Infinite Context Engine (Phase 5)

Prefix tries historically suffer from two fatal physical limits: the rigid `k` horizon limit and the $O(V^k)$ RAM explosion. Version 0.2.0 entirely eliminates both limitations, unlocking infinite context ingestion.

```python
from uchi.online_tokenizer import OnlineTokenizer
from uchi.node_compressor import NodeCompressor

# 1. Beating the Horizon Limit
tokenizer = OnlineTokenizer(max_merges=64)
compressed_stream = tokenizer.tokenize(["h", "e", "l", "l", "o"]) 

# 2. Bounding the RAM Explosion
compressor = NodeCompressor()
stats = compressor.compress_pass(predictor._root, cred_max=6.05)
```

### Realistic Use Cases
1. **Unbounded System Log Analysis**: Ingest weeks of heavy server telemetry without the engine "forgetting" the context of an anomaly that occurred 5 days ago.
2. **Lifelong Chatbot Memory**: Continuously merge conversational token pairs over a user's entire lifetime, allowing an agent to remember semantic context from months prior without increasing trie depth.
3. **Infinite Streaming IoT**: Deploy the engine onto tiny embedded hardware chips and let it stream forever. The node compressor ensures the RAM will never blow out and crash the hardware.

### The Ultimate Benefit
Phase 5 unshackles the engine from the physical boundaries of hardware memory and fixed-time horizons. It guarantees that Uchi can run in an **Infinite Lifelong Learning** loop, dynamically compressing history and pruning memory without human intervention.
