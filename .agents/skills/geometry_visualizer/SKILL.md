---
name: Geometry Visualizer
description: Pulls vectors from the Experience Replay database and runs UMAP to plot a 2D/3D scatter plot image artifact to visualize Uchi's latent clustering.
---

# Geometry Visualizer

Use this skill when you need to literally *see* the 256-dimensional space.
If the engine is failing on a specific domain (e.g. Code vs Conversational), use this skill to verify if the clusters in the latent space have bled into each other.

**Implementation note:** This skill requires a python script (e.g., `scripts/geometry_visualizer.py`) that connects to `brain.uchi` or the SQLite replay DB, runs UMAP/t-SNE on the intent vectors, and generates a `.png` artifact for the user.
