---
name: code
description: Generate code using parallel MCTS workers + REPL oracle
args: <description>
mode: code
---

Bypasses the conversational routing and goes directly to `CodeEngine`.

Runs `n_workers=4` parallel MCTS workers with varied temperatures.
The first candidate to pass `py_compile` wins. If none compile,
a `??HOLE:description??` skeleton is returned — type your implementation
in the input box to teach Uchi the correct pattern.

If `brain_code.uchi` is bootstrapped, the code-specialist brain is used
instead of the default brain.

**Example**
```
/code a function that checks if a number is prime
/code binary search over a sorted list
/code reverse a linked list in place
```
