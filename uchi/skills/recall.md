---
name: recall
description: Query AssociativeMemory directly without trie generation
args: <query>
mode: memory
---

Performs a pure cosine-similarity lookup in the CPUVectorMemory store.
Faster than `/search` but only returns what is already in memory —
no web fallback, no trie generation.

Useful for checking whether a concept has been learned yet.

**Example**
```
/recall gravity
/recall quicksort
/recall joseph woodall
```
