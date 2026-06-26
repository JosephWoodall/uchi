---
name: Trie Walker
description: Detaches the neural network and queries the pure Trie to reveal the exact historical continuations and raw probabilities for a given prefix.
---

# Trie Walker

Use this skill to determine if a failure is due to a lack of raw statistical knowledge (Trie failure) or a bad neural evaluation (Actor-Critic failure).

Given a prefix, it prints the exact raw probabilities from the historical training data before the neural network touches them.

**Implementation note:** This skill requires a python script (e.g., `scripts/trie_walker.py`) that loads `brain.uchi`, takes a sequence of prefix tokens, and prints `predictor.peek_distribution(prefix)`.
