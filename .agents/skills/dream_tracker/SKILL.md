---
name: Dream Tracker
description: Queries the SQLite Experience Replay queue to show the highest-loss hard negatives the offline dreaming daemon is currently training on.
---

# Dream Tracker

Use this skill when you need to diagnose catastrophic forgetting or over-optimization by the offline daemon.

If Uchi's performance suddenly degrades, use this to query the Replay database. It will show exactly which tuples the model is struggling to cluster, revealing if it is memorizing noise.

**Implementation note:** This skill requires a python script (e.g., `scripts/dream_tracker.py`) that connects to the `ExperienceReplayBuffer` and prints the top 10 highest-priority (highest loss) tuples.
