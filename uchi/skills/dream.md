---
name: dream
description: Trigger offline self-play to consolidate and reinforce learned patterns
args: [--daemon]
mode: chat
prefix: run offline dream consolidation
---

Invokes the ConvergentEngine offline dream daemon. The daemon runs self-play
sequences through the SSM, reinforcing high-value trie paths and pruning
low-confidence branches. This is how the brain consolidates knowledge between
active sessions.
