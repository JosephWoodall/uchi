---
name: search
description: Search memory and web for a topic
args: <query>
mode: web_search
---

Searches AssociativeMemory first, then falls back to web if the
`uchi.web_search` plugin is available.

All retrieved content is streamed into the trie so future queries
about the same topic benefit immediately.

**Example**
```
/search quantum entanglement
/search python list comprehension
```
