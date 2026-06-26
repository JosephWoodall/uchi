---
name: Cognitive Debugger
description: Diagnoses Uchi engine failures by probing the tokeniser, trie, or MCTS tree for a given input string. Use this when Uchi produces empty outputs, loops, hallucinations, or token fragmentation.
---

# Cognitive Debugger

Exposes three surgical probes into the Uchi engine. Always start with
`--probe-tokenizer` and `--probe-trie` before escalating to `--probe-mcts`;
the first two are fast (< 1s) and rule out the most common failure classes.

## When to Use

| Symptom | Start with |
|---|---|
| "Empty prediction path" or blank output | `--probe-trie` |
| Repeated output ("see you later see you later") | `--probe-mcts` |
| Token fragmentation / OOV | `--probe-tokenizer` |
| Nonsense code / wrong completions | all three in sequence |
| Diagnose gradient collapse / loop statistics | `--telemetry` |

## Invocation

```bash
# Diagnose tokeniser fragmentation
python scripts/cognitive_debugger.py --probe-tokenizer "a complete gibberish word likepythoon"

# Diagnose empty trie path
python scripts/cognitive_debugger.py --probe-trie "write a python function"

# Dump MCTS tree with value/visit statistics
python scripts/cognitive_debugger.py --probe-mcts "write a python function" --nodes 80

# All three at once
python scripts/cognitive_debugger.py \
  --probe-tokenizer "likepythoon" \
  --probe-trie "write a python function" \
  --probe-mcts "write a python function"

# Show latest telemetry snapshot (written after every MCTS search or dream flush)
python scripts/cognitive_debugger.py --telemetry

# Show last 5 sessions from SQLite history
python scripts/cognitive_debugger.py --telemetry 5

# Custom brain path
python scripts/cognitive_debugger.py --brain /path/to/brain.uchi --probe-trie "hello"
```

## Reading the Output

### `--probe-tokenizer`
- **Trie-direct**: token was in the known vocabulary — no fragmentation.
- **BPE-fallback**: token was OOV; shattered into subwords by tiktoken/bigrams.
- **OOV(n/m)**: n of m subwords are still not in the trie after fallback.
  If coverage < 50%, MCTS will ignore trie statistics and rely on the neural prior alone.

### `--probe-trie`
- Each row shows one token in the seed sequence, whether the trie has a path for it, and how many children exist at that depth.
- **← LOST** marks the exact depth where the trie loses the sequence.
  Continuations shown below are the trie's best guesses from that last-known prefix.

### `--probe-mcts`
- **nodes/s**: throughput. Cold-start SSM gives ~20–60 nodes/s on CPU; trained SSM > 100 nodes/s.
- **pruned %**: fraction of nodes cut by the value head below `PRUNE_THRESHOLD=-0.5`. > 80% pruned suggests the value head is untrained or the input is fully OOV.
- **Top branches**: ranked by `visits × value`. High-visits + high-value = the engine is confident. High-visits + low-value = exploration without conviction.
- **Expansions by depth**: a sharp dropoff at depth 1–2 means the grammar mask or value head is killing all branches immediately.

### `--telemetry`
Reads `.uchi/telemetry/latest.json` (written after every MCTS search) and `.uchi/telemetry/history.db` (SQLite, append-only). Four sections:
- **tokenizer**: `bpe_fallback_count`, `exact_dictionary_count`, `bpe_splits` (per-word breakdowns for OOV words)
- **mcts**: `total_nodes_explored`, `gpu_batch_utilization` (0–1, fraction of EXPAND_K slots filled), `repetition_penalty_applied`, `oracle_interventions`, `top_branches`
- **latent_space**: `vector_l2_norm` (healthy range 1–15; near-zero = collapsed), `kv_cache_stats` (states received / evicted / heavy-hitters retained)
- **dreaming**: `ast_blame_index` (valid prefix length), `contrastive_cosine_sim` (positive vs hard-neg similarity; near 1.0 = latent space not yet separated)

## Diagnostic Recipes

**"Uchi outputs nothing for code prompts"**
```
probe-tokenizer → is coverage < 50%?  yes → need more BPE or ontology mappings
probe-trie      → does path break at depth 0-2?  yes → cold brain; run bootstrap_code.py
probe-mcts      → pruned > 80%?  yes → value head untrained; run offline_dream.py
```

**"Uchi loops: 'see you later see you later'"**
```
probe-mcts → look for top branches where the same token appears ≥ 3 times
             → REP_DECAY penalty should handle this; if loops persist, reduce
               PRUNE_THRESHOLD or increase REP_DECAY in tree_search_engine.py
```

**"Uchi hallucinates on OOV input"**
```
probe-tokenizer → locate fragmented tokens
probe-trie      → confirm path breaks at the fragmented token
                → fix: either teach the word via /teach, or ingest text containing it
```
