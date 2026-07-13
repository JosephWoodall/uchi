![Uchi Logo](docs/logo.png)

[![PyPI version](https://img.shields.io/pypi/v/uchi_python.svg)](https://pypi.org/project/uchi_python/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20Noncommercial%201.0.0-blue.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0)
[![Python Versions](https://img.shields.io/pypi/pyversions/uchi_python.svg)](https://pypi.org/project/uchi_python/)
[![Tests](https://github.com/JosephWoodall/uchi/actions/workflows/ci.yml/badge.svg)](https://github.com/JosephWoodall/uchi/actions/workflows/ci.yml)

## Uchi v0.4.0: The Autonomous Empirical Synthesis Engine

LLMs are trapped in an imitation paradigm—they mimic patterns without understanding reality. Uchi breaks this cycle. By fusing a high-speed generative engine (FLUX) with a live mathematical sandbox (REPLOracle), Uchi doesn't just predict the next word; it writes code, tests its own assumptions, and **discovers** the truth from first principles.

**v0.4.0 gives Uchi hands.** Everything that made v0.3.0 trustworthy — grounded
answers or honest abstention, never confabulation — is unchanged. What's new is
autonomy: `MetaUchi`, the default orchestrator, wraps the same verified engine
with tool calling, a sandboxed Python scratchpad, web search, multi-step goal
tracking with pause/resume, and macro distillation from past successes. Handed
a goal, it can read/write its own files, run and test code, browse for facts
it doesn't have, and pick up exactly where it left off — closing the
capability gap with general assistants (ChatGPT/Claude) without giving up the
one thing that makes Uchi different: it never asserts what it can't trace back
to something real.

### The Axioms
1. **Empirical Grounding:** Text fails; reality doesn't. Uchi uses Test-Driven Development (TDD) to prove its logic in a live Python sandbox before speaking.
2. **The Swarm Synthesizer:** Complex problems are autonomously shattered into atomic concepts, solved in parallel by independent agents, and stitched back together — now self-healing: an identical delegation that already failed doesn't get blindly repeated.
3. **Multi-Agent Debate:** Subjective claims are violently cross-examined by a "Devil's Advocate" instance to prune hallucinations.
4. **Human-Readable Interface:** Complex mathematical discovery is seamlessly translated back into warm, conversational English.
5. **The Compounding Effect:** Every verified script is permanently saved as an autonomous tool. Uchi never has to solve the same problem twice.
6. **Autonomy on a Leash (new in 0.4.0):** Tool calls are logged and loop-guarded (an exact repeat of a failed call is blocked, not retried), goal state compacts instead of losing the original intent over a long task, and when the agent is genuinely stuck it yields to a human instead of guessing. None of that changes axiom 1 — autonomy never gets to assert something ungrounded.

### The Architecture: Under the Hood

To understand Uchi, you must understand the fundamental flaw in modern AI: **LLMs are trapped in an imitation paradigm.** They are autocomplete engines that predict the most statistically likely next word without understanding reality. Uchi solves this by separating the "creative reasoning" from the "factual grounding," and forcing the AI to prove its claims against reality before it is allowed to speak.

Here is the exhaustive breakdown of how the architecture achieves this:

#### 1. The FLUX Proposer (The Creative Brain)
* **What it is:** A custom-built, ultra-lightweight (64M parameter, pruned-vocab) neural network trained from scratch. 
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

#### 7. The Tool Calling Interface (new in 0.4.0) — Giving Uchi Hands
* **What it is:** A formalized grammar (`<|tool_call|> name(args) <|end_tool|>`, plus a concurrent `<|tool_call_async|>` variant) that halts generation, dispatches to a real Python function — filesystem read/write, the scratchpad, web search — and splices the result back in before Uchi resumes. Every call is logged, and an exact repeat of a previously-failed call is blocked outright rather than retried.
* **The Philosophy:** *An action is just another verified fact.* A tool call isn't a side channel bolted onto the chat — it's dispatched through the same registry that logs, loop-guards, and (via a sandboxed filesystem root) contains what an autonomous instance can touch on your machine.

#### 8. Goal State & Checkpointing (new in 0.4.0)
* **What it is:** A `GoalState` object tracks a multi-step task's original intent and what's been learned so far, compacting raw tool logs into short notes once they grow large — so a long-running goal never loses the plot. The whole state (goal, tool history, pending questions) serializes to disk, so a task can be paused and resumed later, even in a different process.
* **The Philosophy:** *Long tasks shouldn't need a babysitter, and they shouldn't need to restart from zero either.*

#### 9. Human-in-the-Loop Yielding (new in 0.4.0)
* **What it is:** When a tool call fails the same way twice in a row, or Uchi explicitly doesn't know how to proceed, it yields — pausing and asking the human a clarifying question — instead of guessing or looping. The next message is treated as the answer, and the task picks back up with that context folded in.
* **The Philosophy:** *Stuck is a valid state.* Brute-forcing through ambiguity is how autonomous agents break things; asking is cheaper than a bad guess.

When you type `uchi tui` and ask a question, you aren't just talking to a chatbot. You are kicking off a microscopic software engineering team. The Swarm breaks your question down, the Index pulls the facts, FLUX writes the code, the REPL executes it, the Devil's Advocate audits the logic, and finally, Uchi translates the mathematically proven result back into warm, conversational English. And now, if the answer requires taking an action instead of just reasoning about one, it can.

> **On benchmarks, honestly:** FLUX is a small (~64M-class) from-scratch model. MMLU,
> SWE-bench, and ARC-Challenge are tracked as a **dashboard** to watch the proposer
> improve — at this scale they stay near baseline, and that is expected. The point
> of the pairing is *trustworthiness* (grounded answers or honest abstention), not
> a leaderboard score. See [`docs/training.md`](docs/training.md) for how FLUX is
> trained and what the final `flux_best.pt` artifact is.

## The Non-Negotiables

1. **Compounding Effect** — `learn()` always accepts a string; `ask()` always
   returns one. The output of one `Uchi` instance is directly learnable by the
   next — knowledge compounds across instances with zero glue code.
2. **Simplified Public API** — the exact same commands work identically via
   the Python SDK, the TUI (`uchi tui`), and the REST API (`uchi serve`).
3. **General Reasoning & Reasoning Chains** — FLUX proposes multi-step
   reasoning (real, CoT-trained); Uchi verifies every link — by fact-checking
   against the brain, executing and testing code in a real sandbox, or
   cross-examining the logic — and abstains the moment a step can't be proven.
4. **Human-Readable I/O** — every input and output is a clear, interpretable
   string. No raw tokens, no opaque state.
5. **OOD Generalization** — FLUX supplies the generative capability to attempt
   questions it has never seen verbatim; Uchi's grounding gate keeps those
   attempts honest, emitting an answer only when it can be traced to something
   real and abstaining otherwise.
6. **The Autonomy Contract (new in 0.4.0)** — every tool call is logged and
   loop-guarded, goal state compacts instead of losing the original intent,
   and Uchi yields to a human rather than looping when it's genuinely stuck.
   None of this weakens non-negotiable 5 — an autonomous instance still never
   asserts a claim it can't trace back to something it actually retrieved,
   executed, or verified.

See [`tasks/0.3.0 Itemized Deliverables.md`](tasks/0.3.0%20Itemized%20Deliverables.md)
and [`tasks/0.4.0 Itemized Deliverables.md`](tasks/0.4.0%20Itemized%20Deliverables.md)
for the problem/intent behind each one.

## How It Connects

Training produces **one artifact**, and every interface loads it through **one place** — `Core.__init__`, wrapped by `MetaUchi` (what `from uchi import Uchi` actually returns as of 0.4.0). The SDK, TUI, and REST server all construct the same facade, so training FLUX once — or adding a new tool — makes it live everywhere automatically. If no checkpoint exists yet, the proposer degrades gracefully and the verifier falls back to grounded extraction / honest abstention — Uchi still runs, and the autonomy layer (tool calling, goal state) works identically regardless of whether FLUX is loaded.

```
scripts/train_all.sh ─► uchi/flux/checkpoints/flux_best.pt   ◄── the artifact
                                     │
        Core.__init__ picks first that exists:
        flux_best → qat_best → cot_best → sft_best   (else None)
                                     │
        FluxProposer.load(ckpt) → build_generate_fn(ckpt)
           loads the HybridTSSM model, returns generate_fn(prompt) -> str
                                     │
        self.proposer ──► GenerateAndGround(index, oracle, proposer)
                                     │        (retrieve → propose → verify → emit/abstain)
                                     ├──► self.swarm    SwarmSynthesizer   (IQ-gated,
                                     │                  loop-guarded delegation)
                                     └──► self.tools    ToolRegistry       (filesystem,
                                                         scratchpad, web search —
                                                         <|tool_call|> / <|tool_call_async|>
                                                         dispatch + splice, loop-guarded)
                                     │
                                  Core.ask(q)
                                     │
              MetaUchi(Core) ── from uchi import Uchi returns this; forwards
                                 ask()/learn()/ingest() unchanged, adds tool
                                 calling, goal state, checkpointing on top
                                     │
             ┌───────────────────────┼──────────────────────────────────┐
            SDK                      TUI                                 REST
   from uchi import Uchi     `uchi tui` → UchiApp              `uchi serve` → api_server
   u = Uchi(); u.ask(q)      → router = Uchi()                 POST /ask, /ask/stream (SSE),
   uchi.Core for the raw     → live Glass Brain tool-call       /v1/chat/completions, /chat
   unorchestrated node         trace panel
```

The `GenerateAndGround` box above is simplified — `oracle` is itself a layered,
additive-only veto cascade (word-overlap → entailment classifier + OOD gate →
numeric plausibility → relational transitivity), and `GenerateAndGround` also
takes `task_config_cache` (dynamic-N self-consistency voting) and an optional
`proprioception` component. See [`docs/architecture.md`](docs/architecture.md)
for the full breakdown of what each layer actually checks.

See [`docs/training.md`](docs/training.md) for the training pipeline and the `flux_best.pt` artifact.

## Simplified Public API (SDK, TUI, & REST)

Uchi standardizes all interactions across three human-readable interfaces. Whether you are scripting, using the terminal, or building a web app, the commands are identical.

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


# ── 4. Giving Uchi Hands (new in 0.4.0) ──
# Track a multi-step goal; tool calls it makes along the way are logged,
# loop-guarded, and folded into the goal's running context automatically.
u.start_goal("Reconcile this quarter's expense report")
u.ingest("expenses.csv")
result = u.ask("Write and run a script that flags any expense over $10,000.")

# Pause anytime — serializes goal state, tool history, and pending questions.
u.checkpoint("task_state.json")
# ... later, even in a different process ...
u2 = Uchi()
u2.resume("task_state.json")   # picks up exactly where it left off

# A successful multi-step task gets distilled into a reusable macro automatically.
u.distill_and_learn()
```

### 2. Terminal UI (TUI)
The TUI isn't just a chatbot; it is a live telemetry dashboard into the Empirical Synthesis Engine. When you ask a question, you will see the Swarm decomposing the task, the REPL executing code, and the FactCheck Oracle pruning hallucinations in real-time — plus, new in 0.4.0, a **Glass Brain** panel showing the live tool-call trace: which tool ran, with what arguments, and whether it succeeded, failed, or got blocked by the loop guard.

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

New in 0.4.0 — stream the internal reasoning trace live (Server-Sent Events),
or plug Uchi into any OpenAI-API-compatible frontend (Open-WebUI, etc.):
```bash
# Observable Monologue: stream "thought" events as they happen, then a final
# "speech" event with the answer -- masks perceived latency by showing the
# agent working, not a blank wait.
curl -N -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the boiling point of water?"}'

# Drop-in OpenAI chat completions shape -- point any compatible client at
# this instead of api.openai.com.
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is the boiling point of water?"}]}'
```

## Install

```bash
pip install uchi_python
```
See `docs/` for architecture details and the full API reference.

## Training FLUX Yourself

A trained FLUX checkpoint and the premade general-knowledge brain already
exist — you don't need to train anything to use Uchi. This section is for
anyone who wants to retrain FLUX, fine-tune it further, or reproduce the
results.

### Use the shipped weights (default, no training)

**`pip install uchi_python`** gets you the code plus the premade
general-knowledge brain (`uchi/data/embeddings.pt`, ~221MB) — it's fetched
automatically and cached (`~/.uchi/data/`) the first time you construct
`Uchi()`, so the wheel itself stays small (PyPI hard-caps uploads at 100MB per
file, well under the brain's size). After the first run it's local, no
further network needed. The FLUX proposer isn't bundled with `pip install`
(same size constraint) — without it, `Uchi()` still works, just with the
proposer degraded to grounded extraction / honest abstention only.

**`git clone` + Git LFS** gets you everything as real files, no download step
at construction time:

```bash
git clone https://github.com/JosephWoodall/uchi.git
cd uchi
git lfs install        # once per machine — see below if you don't have Git LFS
git lfs pull           # fetches uchi/flux/checkpoints/flux_best.pt (~450MB)
                        # and uchi/data/embeddings.pt (~221MB, the premade brain)
```

```python
from uchi import Uchi
u = Uchi()   # loads uchi/flux/checkpoints/flux_best.pt + the premade brain automatically
```

If you don't have Git LFS: `sudo apt install git-lfs` (Debian/Ubuntu),
`brew install git-lfs` (macOS), `sudo pacman -S git-lfs` (Arch), or see
[git-lfs.github.com](https://git-lfs.github.com). Without it, the LFS-tracked
files show up as small text pointers instead of the real weights — `Uchi()`
will still run, just with the proposer degraded to `None` (grounded
extraction / honest abstention only, no FLUX generation) until you `git lfs
pull`.

### Train from scratch

The full pipeline — pre-training → SFT → CoT distillation → ternary QAT — is
one script:

```bash
bash scripts/train_all.sh
```

This produces `uchi/flux/checkpoints/flux_best.pt` (the same artifact
`Uchi()` loads) via four phases, each skipped automatically if its output
already exists and is newer than its input (safe to re-run/resume). Needs a
CUDA GPU; `~12GB` VRAM covers the default config. Full details — exact phase
breakdown, hyperparameters, and what each phase teaches — are in
[`docs/training.md`](docs/training.md).

```
Pre-train (FineWeb-Edu)  →  SFT (SQuAD/Dolly/Code/Chat)  →  CoT (Math/General/Code)  →  Ternary QAT
     ckpt_best.pt              sft_best.pt                     cot_best.pt              flux_best.pt
```

Pre-training's default path (`train_all.sh`) pre-tokenizes FineWeb-Edu (falls
back to OpenWebText if unreachable) into a memmap for GPU-bound throughput.
Running `train_v2.py` directly without a pre-tokenized `--data-bin` hits its
own streaming fallback instead (OpenWebText + Wikipedia + Code), ~10× slower
but useful for a quick proof run without a separate tokenization step.

**0.4.0 FLUX scale-up (complete):** two structural changes to reclaim
embedding-table capacity for reasoning layers, both real and measured before
being folded into the full training run that produced the current default
checkpoint — not proposed, not in progress, live:
- **Vocab pruning** — the full cl100k_base vocab (100,300 tokens) was ~66% of
  the prior 116M-class model's parameters. `scripts/build_pruned_vocab.py`
  computed which tokens the actual training corpus uses and kept the top
  32,018 — measured at 97–99% real coverage depending on the sample. The
  current default checkpoint (`flux_best.pt`) is this pruned-vocab model:
  64,010,496 parameters total, confirmed directly against the loaded
  checkpoint's own weight shapes, not a target or an estimate. Still
  available as an explicit flag (`--pruned-vocab uchi/flux/checkpoints/pruned_vocab_32k.json`
  on `train_v2.py` / `sft_train.py`) for anyone retraining from scratch.
- **Fused SSM scan** — `UCHI_FUSE_SSM_SCAN=1` kernel-fuses the sequential scan
  (same algorithm, not a switch to a parallel scan — that was tried before and
  reverted for measured memory-bandwidth reasons) via `torch.compile` over the
  whole loop rather than per-step. Measured 3× forward speedup on an RTX 5070;
  a one-time ~2min compile cost per shape, so it's meant for a real training
  run's thousands of steps, not interactive single-token decode.
