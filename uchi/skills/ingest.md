---
name: ingest
description: Ingest a file or directory into the brain
args: <path>
mode: ingest
---

Streams file contents into the trie with structural bounds
(`<|file:name|>` ... `<|/file|>`).

Supports `.txt`, `.md`, `.py`, `.json`, `.csv` files.
Directories are walked recursively (skips `__pycache__`, `.venv`, etc.).

**Example**
```
/ingest ./notes.md
/ingest ./src/
/ingest /home/user/research/paper.txt
```
