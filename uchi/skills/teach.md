---
name: teach
description: Directly teach Uchi a fact as a Q→A pair
args: <question> | <answer>
mode: teach
---

Streams a new `<|user|> Q <|assistant|> A` sequence directly into the
trie. The separator is `|`. No MCTS, no hallucination gate — the
exact pair is learned immediately.

This is the fastest way to correct a wrong answer or add new knowledge.

**Example**
```
/teach what is the speed of light | approximately 3 times 10 to the 8 meters per second
/teach who invented the telephone | Alexander Graham Bell invented the telephone in 1876
/teach what is a monad | a monad is a design pattern for chaining computations with context
```
