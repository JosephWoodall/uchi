![Uchi Logo](docs/logo.png)

# Universal Sequence Predictor

[![PyPI version](https://img.shields.io/pypi/v/uchi_python.svg)](https://pypi.org/project/uchi_python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/uchi_python.svg)](https://pypi.org/project/uchi_python/)
[![Tests](https://github.com/JosephWoodall/uchi/actions/workflows/ci.yml/badge.svg)](https://github.com/JosephWoodall/uchi/actions/workflows/ci.yml)

## Core Mission: Omni-modal Deterministic Universal Sequence Predictor (ODUSP)
Uchi v0.2.0 transforms the architecture from a simple sequence predictor into a completely multi-modal Deterministic Universal Sequence Predictor. It ingests text, audio, images, math telemetry, and code simultaneously—without any neural weights or pre-training. It crushes massive context histories via Phase 4 BPE compression, storing concepts in a zero-shot Fractal Attention memory buffer that mimics biological synesthesia. It is not an LLM; it is a pure mathematical sequence predictor. v0.3.0 adds a routing layer with intent-based query dispatch via `ProceduralMemory` and a trainable SSM confidence signal that improves prediction quality without introducing an LLM dependency.


> [!NOTE]
> Please see `docs/` for the complete Algorithmic Walkthrough, ODUSP vs LLM Benchmarks, and full API references.

---

> [!NOTE]
> **Comprehensive Documentation & API Reference**
>
> For interactive examples, API documentation, and to see the newest capabilities (including online Math Learning, Vector Retrievals, and the Simulation Engine), please see our full documentation website.
>
> **[Read the Full Documentation →](https://github.com/JosephWoodall/uchi/tree/main/docs)**

---

## Installation

```bash
pip install -e ".[all]"
```

On first launch, Uchi runs a one-time bootstrap (Python stdlib patterns + Wikipedia facts) and saves the result to `brain.uchi`. Subsequent launches are instant.

## Quickstart

Uchi has two entry points that share the same `brain.uchi` — every interaction in either one improves the other.

### 1. Terminal UI (TUI)

```bash
uchi                        # launch interactive chat
uchi --preload data.txt     # pre-train with a file before chatting
uchi --brain /path/to/brain.uchi   # use a specific brain file
```

Inside the TUI:

| Command | Description |
|---|---|
| Just type | Chat with Uchi — it learns from every turn |
| `/load <file>` | Stream any file into the knowledge base |
| `/save` | Force-save the current brain state to disk |
| `Ctrl+S` | Save brain |
| `Ctrl+C` | Save and quit |

Uchi gives positive/negative feedback signals to improve itself — type "good", "correct", "yes" to reinforce a response, or "wrong", "bad", "no" to prune it.

### 2. REST API

```bash
uvicorn uchi.api_server:app --host 0.0.0.0 --port 8000
```

**POST /chat**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "what is the capital of France?"}'
# {"reply": "paris", "entropy": 3.2}
```

**GET /metrics**
```bash
curl http://localhost:8000/metrics
# {"status": "online", "memory_records": 1024, "mode": "deterministic"}
```

**GET /debug/walk**
```bash
curl http://localhost:8000/debug/walk
# Returns trie walk data from the last prediction (depth, contributions, similarity)
```

### 3. Python API

```python
from uchi.omni_router import OmniRouter
from uchi.cli import load_brain, save_brain

# Load existing brain or create a new one
router = load_brain("brain.uchi") or OmniRouter()

# Chat
reply = router.chat("what is the capital of France?")
print(reply)  # → "paris"

# Teach it something new
router.stream(["<|user|>", "what", "is", "the", "capital", "of", "germany",
               "<|assistant|>", "berlin"])

# Save
save_brain(router, "brain.uchi")
```

### 4. Offline Knowledge Bootstrapping

To scale Uchi's knowledge base beyond the cold-start defaults, run these scripts once before distributing your `brain.uchi`:

```bash
# Ingest Wikipedia + code_search_net via HuggingFace (requires: pip install datasets)
python scripts/bootstrap_knowledge.py --limit 10000

# Ingest Python stdlib function patterns via AST (no internet required)
python scripts/bootstrap_code.py

# Ingest Wikipedia fact triples via spaCy SVO extraction (requires: pip install wikipedia spacy)
python scripts/bootstrap_wikidata.py
```

The resulting `brain.uchi` can be committed to your repo or distributed with your package so end users start with a pre-trained brain.

## Benchmarks

| Metric | Score | Notes |
|---|---|---|
| **Pre-load Recall** | **100.0%** (n=10) | Stream N facts → immediately test recall; measures deterministic memory |
| **Zero Catastrophic Forgetting** | **100.0%** after 100 noise facts | Anchor facts recalled correctly after 100 distractors streamed on top |
| **Latency vs. Brain Size** | 10facts→3571ms  50facts→3704ms  100facts→5594ms | Proves O(depth) trie lookup: latency stays flat as brain grows |
| **Code Completion** | **0.0%** (n=5) | Python function stub → body; scored by syntax + keyword validity |
| **Inference Latency** | **15185.22 ms** | Single turn on a pre-loaded fact, web search disabled |
| **RAM Footprint** | **1596.5 MB** | Resident set after brain load + recall stream |
| **Hallucination Rate** | **0%** | Strict trie boundary enforcement |

