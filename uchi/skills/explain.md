---
name: explain
description: Explain a concept using everything Uchi knows
args: <concept>
mode: chat
prefix: explain the concept of
---

Routes through the full chat pipeline with an explanatory prefix prepended.
Uses the AssociativeMemory and trie together to form the response.

**Example**
```
/explain recursion
/explain the Fourier transform
/explain gradient descent
```
