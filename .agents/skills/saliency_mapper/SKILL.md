---
name: Saliency Mapper
description: Calculates the mathematical gradient of the Value Network with respect to the input prompt tokens to see which words drove the evaluation.
---

# Saliency Mapper

Use this skill when you need to explain *why* the SSM Value Head gave a high or low score to a specific path.

This calculates token contribution (red words drag the score down, green words push it up). It proves exactly which parts of the prompt steered the 256D hidden state.

**Implementation note:** This skill requires a python script (e.g., `scripts/saliency_mapper.py`) that computes backpropagation gradients from the Value Network to the token embeddings, and prints out a color-coded or scaled representation of the prompt tokens.
