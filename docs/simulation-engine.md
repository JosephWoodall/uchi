# Simulation Engine & Advanced Architecture (v0.2.0)

Version 0.2.0 introduced four major architectural extensions that take Uchi from a standard sequence predictor to a lifelong simulation engine.

---

## 1. Mathematical Simulation Stream

The engine learns arithmetic and algebraic rules completely online, extracting structure and intelligence purely through sequence prediction without any pre-training.

```python
from uchi.forest import PredictorForest
from datasets import load_math_corpus # Assumes you have a dynamic generator

# 1. Initialize an empty forest
forest = PredictorForest(context_length=6, n_trees=5)

# 2. Load a continuous stream of random math equations
math_stream = load_math_corpus(n_chars=20000, seed=42)

# 3. Stream data and let the model grow dynamically
for char in math_stream:
    # Predict the next character based on current context
    pred, conf = forest.predict()
    
    # Observe the actual character and provide feedback
    forest.observe(char)
    forest.feedback(char)
```

---

## 2. Optimal Vector Retrieval

When the sequence generator encounters a novel context, it seamlessly falls back to an optimal geometric dense vector query, querying historical distributions rather than guessing randomly.

```python
from uchi.generative import SequenceGenerator
from datasets import load_gutenberg_text

# 1. Enable vector retrieval when instantiating the generator
generator = SequenceGenerator(
    context_length=5, 
    use_vector_retrieval=True  # Enables Phase 2 Vector Fallback
)

text = load_gutenberg_text(n_chars=5000)

# 2. Train online
for char in text:
    generator.observe(char)
    generator.feedback(char)

# 3. Generate novel text
# If 'generator' hits a context it hasn't seen natively, 
# it falls back to the dense NumpyVectorIndex.
generated = generator.generate(max_tokens=100)
```

---

## 3. Ontological Process Predictor

Instead of predicting raw strings or numbers, the system can predict discrete, strictly-typed states and actions to model complex workflows, agent loops, or business logic.

```python
from uchi.process import ProcessPredictor, OntologicalState, OntologicalAction

predictor = ProcessPredictor(context_length=4)

# Define typed states and actions
s_sort   = OntologicalState(name="NeedSort", properties=("unordered_list",))
a_sort   = OntologicalAction(name="ExecuteSort", target="QuickSort")
s_search = OntologicalState(name="NeedSearch", properties=("sorted_list",))

# Stream observations of the workflow
predictor.observe_state(s_sort)

# Predict the next action the agent/system should take
pred_action, conf = predictor.predict_next_action()
print(f"Predicted Action: {pred_action}")

# Provide the ground truth action that was actually taken
predictor.feedback_action(a_sort)
```

---

## 4. Plural Simulation Engine

Simulate thousands of lives (or workflows) in parallel, accumulating their separate experiences into a unified meta-prediction intuition.

```python
from uchi.simulation_engine import LifelongSimulationEngine

# 1. Initialize multiple independent predictor forests ("lives")
engine = LifelongSimulationEngine(n_instances=5, context_length=4)

# 2. Define sequences representing different life experiences
life_1 = ["Eat", "Work", "Sleep", "Eat", "Work", "Sleep"]
life_2 = ["Play", "Eat", "Sleep", "Play", "Eat", "Sleep"]

# 3. Stream in parallel; instances learn independently
engine.stream_parallel([life_1, life_2])

# 4. Request a pluralistic vote across all independent lives
plural_prediction, meta_confidence = engine.vote_plural()
print(f"Wisdom of the crowd prediction: {plural_prediction} (Conf: {meta_confidence:.2f})")
```
