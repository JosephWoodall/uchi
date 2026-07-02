# OmniRouter

> **Python users:** `OmniRouter` is the internal engine. The recommended interface
> is `Uchi`, which wraps it. See [Python API →](python-api.md).
>
> ```python
> from uchi import Uchi
> u = Uchi()
> u.learn("text")     # streams into the trie + retrieval index
> u.ask("question")   # routes through OmniRouter.chat()
> u.router            # escape hatch: direct OmniRouter access
> ```

---

`OmniRouter` is the front controller. It classifies each message and dispatches it
to one of three lanes, then owns persistence and the compounding `learn()` path.

## `chat(message)` — the three-lane router

```
message ─► intent_router.classify_intent ─► skill  | social | factual
```

- **skill** → `SkillRegistry.dispatch` (analytical commands, code).
- **social** → `ConversationEngine.reply` — free-generated chit-chat; no oracle,
  because social replies assert no facts.
- **factual** → `answer()` (Generate-and-Ground). Multi-step factual questions
  (`;`, `then`, numbered) route to `ReasoningEngine.reason` first.

## `answer(question)` — Generate-and-Ground

Retrieve evidence from the semantic index → answerability gate → generate a
candidate (neural decoder, or extractive) → fact-check → emit or abstain. Never
confabulates. See [Generate-and-Ground →](generate-and-ground.md).

## `stream(tokens)` / `learn(text)`

Ingests knowledge into **both** the recall trie (`predictor`) and the retrieval
index, so learned text is immediately groundable.

## What it holds

| Attribute | Role |
|-----------|------|
| `predictor` | credibility-weighted trie — recall + grounding (append-only, never forgets) |
| `_semantic_index` | `SemanticIndex` retrieval over ingested knowledge |
| `skills` | `SkillRegistry` — analytical/code skills |
| `tokenizer`, `memory` | tokenisation + associative memory |

Model artifacts (answer decoder, answerability classifier, chat decoder) are
lazy-loaded from `uchi/data/*.pt` and are not pickled with the brain.

## Retired

The SSM value-head QA path, GRPO training loop, `AgenticBaseline`, the convergent
MCTS engine, and trie-based text generation were removed. Generation is now the
neural decoder gated by the fact-check oracle; the trie is kept for recall/grounding.
