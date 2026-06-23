---
name: dream
description: Trigger an offline dreaming cycle to self-improve
args: (none)
mode: dream
---

Spawns `scripts/offline_dreaming.py` as a background process.

The dreaming cycle replays high-confidence trie patterns with synthetic
RL challenges, reinforcing correct predictions and pruning incorrect ones
without blocking the interactive session.

**Example**
```
/dream
```
