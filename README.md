![Uchi Logo](docs/logo.png)

[![PyPI version](https://img.shields.io/pypi/v/uchi_python.svg)](https://pypi.org/project/uchi_python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/uchi_python.svg)](https://pypi.org/project/uchi_python/)
[![Tests](https://github.com/JosephWoodall/uchi/actions/workflows/ci.yml/badge.svg)](https://github.com/JosephWoodall/uchi/actions/workflows/ci.yml)

## Uchi v0.3.0: The Empirical Synthesis Engine

LLMs are trapped in an imitation paradigm—they mimic patterns without understanding reality. Uchi breaks this cycle. By fusing a high-speed generative engine (FLUX) with a live mathematical sandbox (REPLOracle), Uchi doesn't just predict the next word; it writes code, tests its own assumptions, and **discovers** the truth from first principles.

### The 5 Axioms of v0.3.0
1. **Empirical Grounding:** Text fails; reality doesn't. Uchi uses Test-Driven Development (TDD) to prove its logic in a live Python sandbox before speaking.
2. **The Swarm Synthesizer:** Complex problems are autonomously shattered into atomic concepts, solved in parallel by independent agents, and stitched back together.
3. **Multi-Agent Debate:** Subjective claims are violently cross-examined by a "Devil's Advocate" instance to prune hallucinations.
4. **Human-Readable Interface:** Complex mathematical discovery is seamlessly translated back into warm, conversational English.
5. **The Compounding Effect:** Every verified script is permanently saved as an autonomous tool. Uchi never has to solve the same problem twice.

### The Architecture: Under the Hood

To understand Uchi, you must understand the fundamental flaw in modern AI: **LLMs are trapped in an imitation paradigm.** They are autocomplete engines that predict the most statistically likely next word without understanding reality. Uchi solves this by separating the "creative reasoning" from the "factual grounding," and forcing the AI to prove its claims against reality before it is allowed to speak.

Here is the exhaustive breakdown of how the architecture achieves this:

#### 1. The FLUX Proposer (The Creative Brain)
* **What it is:** A custom-built, ultra-lightweight (116M parameter) neural network trained from scratch. 
* **The Philosophy:** *Small, fast, and creative.* Modern LLMs are massive because they try to memorize the entire internet. FLUX doesn't memorize trivia; it is trained strictly on *how to reason*, *how to code*, and *how to generalize*. It provides the raw, creative horsepower to tackle novel problems.

#### 2. The Semantic Index / `brain.uchi` (The Factual Anchor)
* **What it is:** A local, persistent database that ingests your PDFs, CSVs, and codebases into an algorithmic search space. 
* **The Philosophy:** *Separate reasoning from facts.* Because FLUX is tiny and has amnesia regarding world trivia, the `brain.uchi` file provides the cold, hard truths. When you ask a question, the facts are pulled from the index, and FLUX applies its reasoning directly to *your* data. No hallucinations based on outdated training data.

#### 3. The REPL Sandbox (Empirical Discovery)
* **What it is:** A live, isolated Python execution environment. If the answer to your question isn't explicitly written in your documents, FLUX writes a Python script, injects Test-Driven Development (TDD) assertions to check its own logic, and executes the code.
* **The Philosophy:** *Don't guess, calculate.* If you ask a standard LLM, "What is the average of these 50 invoices?", it will guess and often fail. Uchi will write a script, run the math, prove it empirically, and return the absolute truth. Reality doesn't lie.

#### 4. The Swarm Synthesizer (Map-Reduce Reasoning)
* **What it is:** An orchestration layer that intercepts complex questions, shatters them into atomic sub-tasks, and spins up a parallel swarm of FLUX agents to solve them simultaneously before stitching the final answer together.
* **The Philosophy:** *Divide and conquer.* LLMs degrade rapidly when trying to hold complex, multi-step logic in a single context window. By breaking the problem down into independent threads, Uchi can scale its intelligence dynamically based on how difficult the prompt is.

#### 5. Multi-Agent Debate (The Devil's Advocate)
* **What it is:** An adversarial loop that violently cross-examines generated answers. The **FactCheck Oracle** algorithmically rejects any claims not grounded in the source text. The **Devil's Advocate** agent attacks the logical soundness of the argument and forces FLUX to reflect and rewrite if a flaw is found.
* **The Philosophy:** *Truth is forged in conflict.* LLMs are sycophants; they want to please the user, even if it means lying. By forcing the model to defend its logic against a hostile critic, we physically prevent hallucinations from reaching the user. 

#### 6. Procedural Memory (The Compounding Flywheel)
* **What it is:** Whenever Uchi successfully uses the REPL Sandbox to solve a novel problem, it permanently caches that verified Python script as an autonomous tool.
* **The Philosophy:** *Never solve the same problem twice.* Standard LLMs start from zero every single time you open a chat. Uchi actually *compounds* its knowledge over time. The longer you use it, the larger its library of custom-built tools grows, making it exponentially faster and more capable.

When you type `uchi tui` and ask a question, you aren't just talking to a chatbot. You are kicking off a microscopic software engineering team. The Swarm breaks your question down, the Index pulls the facts, FLUX writes the code, the REPL executes it, the Devil's Advocate audits the logic, and finally, Uchi translates the mathematically proven result back into warm, conversational English.

> **On benchmarks, honestly:** FLUX is a small (~116M) from-scratch model. MMLU,
> SWE-bench, and ARC-Challenge are tracked as a **dashboard** to watch the proposer
> improve — at this scale they stay near baseline, and that is expected. The point
> of the pairing is *trustworthiness* (grounded answers or honest abstention), not
> a leaderboard score. See [`docs/training.md`](docs/training.md) for how FLUX is
> trained and what the final `flux_best.pt` artifact is.

## How It Connects

Training produces **one artifact**, and every interface loads it through **one place** — `Uchi.__init__`. The SDK, TUI, and REST server all construct the same `Uchi` object, so training FLUX once makes it live everywhere automatically. If no checkpoint exists yet, the proposer degrades gracefully and the verifier falls back to grounded extraction / honest abstention — Uchi still runs.

```
scripts/train_all.sh ─► uchi/flux/checkpoints/flux_best.pt   ◄── the artifact
                                     │
        Uchi.__init__ picks first that exists:
        flux_best → qat_best → cot_best → sft_best   (else None)
                                     │
        FluxProposer.load(ckpt) → build_generate_fn(ckpt)
           loads the HybridTSSM model, returns generate_fn(prompt) -> str
                                     │
        self.proposer ──► GenerateAndGround(index, oracle, proposer)
                                     │              (retrieve → propose → verify → emit/abstain)
                               Uchi.ask(q)
             ┌───────────────────────┼───────────────────────┐
            SDK                      TUI                      REST
   from uchi import Uchi     `uchi tui` → UchiApp     `uchi serve` → api_server
   u = Uchi(); u.ask(q)      → router = Uchi()        → _router = Uchi(); POST /ask
```

See [`docs/training.md`](docs/training.md) for the training pipeline and the `flux_best.pt` artifact.

## Simplified Public API (SDK, TUI, & REST)

Uchi v0.3.0 standardizes all interactions across three human-readable interfaces. Whether you are scripting, using the terminal, or building a web app, the commands are identical.

### 1. Python SDK

The SDK is designed to be completely modular. Because all input and output is human-readable, multiple autonomous `Uchi` instances can be chained together. **The output of one instance becomes the factual grounding for the next.**

```python
from uchi import Uchi

# ── 1. Basic Ingestion & Question Answering ──
u = Uchi()
u.ingest("docs/").ingest("data.csv") # Recursively ingest entire directories
answer = u.ask("What is the primary conclusion of the Q3 data?")


# ── 2. The Compounding Effect (Agent Chaining) ──
# Instance 1: The Data Analyst
analyst = Uchi()
analyst.ingest("financials.csv")
report = analyst.ask("Write a comprehensive financial report calculating YoY growth.")

# Instance 2: The Executive Strategist
executive = Uchi()
executive.learn(report) # Ground the executive agent on the analyst's output
strategy = executive.ask("Based on this report, should we cut marketing spend?")
print(strategy)


# ── 3. Analytical Tools via Slash Commands ──
# You can bypass conversational text and run raw ML tasks through the exact same interface
u.ask("/classify", X=X_train, y=y_train)
u.ask("/forecast", X=time_series_data, steps=20)
```

### 2. Terminal UI (TUI)
The TUI isn't just a chatbot; it is a live telemetry dashboard into the Empirical Synthesis Engine. When you ask a question, you will see the Swarm decomposing the task, the REPL executing code, and the FactCheck Oracle pruning hallucinations in real-time.

```bash
# Launch the dashboard
uchi tui

# You can also preload a knowledge base directly from the command line:
uchi tui --preload ./my_project_folder
```

Inside the TUI, use the exact same commands as the SDK:
```bash
> What is the Eiffel Tower?
> /classify data.csv --label target_col
```

### 3. REST API
Uchi can be deployed as a headless reasoning microservice with a single command. 

```bash
# Boot the Uchi Swarm on port 8000
uchi serve --port 8000
```

Because the API is universal, you send the exact same queries via HTTP:
```bash
# Ask a reasoning question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Calculate the factorial of 5 using Python."}'

# Execute a specialized skill
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "/classify data.csv"}'
```

## Install

```bash
pip install uchi_python
```
See `docs/` for architecture details and the full API reference.
