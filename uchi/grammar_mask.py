"""
grammar_mask.py
===============
Grammar-constrained sampling for Python code generation via a deterministic
Pushdown Automaton (PDA).

Applied at MCTS expansion time: given the token sequence accumulated so far,
filters the trie's next-token distribution to zero out tokens that would
create a syntactically invalid Python state (unbalanced brackets, illegal
operator sequences, keyword placement violations).

Design constraints:
- O(n) state computation from token sequence — no global parse state needed.
- Purely subtractive: valid tokens are untouched; probabilities are not
  renormalized here (the caller may renormalize if needed).
- Conservative: only mask tokens we are *certain* are invalid. Uncertain cases
  pass through — false negatives are safer than false positives during search.

PDA state tracked:
  bracket_stack    LIFO stack of open brackets (ensures balanced pairing)
  block_stack      LIFO stack of block-opening keywords (def/class/if/for/while…)
  in_code_block    whether inside a ```python … ``` fence
  last_meaningful  previous non-whitespace token
  after_def_class  True when last block keyword was def/class (needs identifier)
"""

from __future__ import annotations

from typing import Dict, List

# ── bracket automaton ────────────────────────────────────────────────────────
_OPEN  = {"(", "[", "{"}
_CLOSE = {")", "]", "}"}
_MATCH: Dict[str, str] = {")": "(", "]": "[", "}": "{"}

# ── operator sets ────────────────────────────────────────────────────────────
_BINARY_OPS = {
    "=", "+=", "-=", "*=", "/=", "//=", "%=", "**=",
    "+", "-", "*", "/", "//", "%", "**", "&", "|", "^",
    ">>", "<<", "==", "!=", "<", ">", "<=", ">=",
}

# ── keyword context ──────────────────────────────────────────────────────────
_BLOCK_OPENERS = {
    "def", "class", "if", "elif", "else", "for", "while",
    "with", "try", "except", "finally",
}
_LOOP_KEYWORDS   = {"for", "while"}
_DEF_CLASS       = {"def", "class"}
_JUMP_STMTS      = {"return", "break", "continue", "yield"}

# These tokens are syntactically invalid directly after a def/class keyword
# (the next token must be a name/identifier).
_CANNOT_FOLLOW_DEF_CLASS: set[str] = (
    _CLOSE | _BINARY_OPS | _BLOCK_OPENERS | _JUMP_STMTS |
    {"pass", "import", "from", "raise", "assert", "del",
     "global", "nonlocal", ":", ",", ";", "@"}
)


def _compute_state(
    tokens: List[str],
) -> "tuple[list[str], list[str], bool, str | None, bool]":
    """
    Scan *tokens* and return the current PDA state.

    Returns:
        bracket_stack   open bracket stack (most recent last)
        block_stack     stack of active block-opening keywords
        in_code_block   whether inside a ```python … ``` fence
        last_meaningful last non-whitespace token seen (or None)
        after_def_class True when the most recent block opener was def/class
                        and no identifier-like token has followed yet
    """
    bracket_stack: list[str] = []
    block_stack:   list[str] = []
    in_code        = False
    last_tok: "str | None" = None
    after_def_class = False

    for t in tokens:
        if t == "```python":
            in_code = True
            bracket_stack  = []
            block_stack    = []
            last_tok       = None
            after_def_class = False
            continue
        if t == "```" and in_code:
            in_code = False
            bracket_stack  = []
            block_stack    = []
            last_tok       = None
            after_def_class = False
            continue
        if not in_code:
            continue

        s = t.strip()
        if not s:
            continue

        # Bracket PDA
        if s in _OPEN:
            bracket_stack.append(s)
            after_def_class = False
        elif s in _CLOSE:
            if bracket_stack and bracket_stack[-1] == _MATCH[s]:
                bracket_stack.pop()
            after_def_class = False
        elif s in _BLOCK_OPENERS:
            block_stack.append(s)
            after_def_class = s in _DEF_CLASS
        elif after_def_class and s not in _CANNOT_FOLLOW_DEF_CLASS:
            # An identifier-like token followed def/class — expectation fulfilled
            after_def_class = False
        elif s == ":":
            # Colon closes the header of the current block opener
            after_def_class = False
        elif s not in _BINARY_OPS:
            after_def_class = False

        last_tok = s

    return bracket_stack, block_stack, in_code, last_tok, after_def_class


def apply(tokens: List[str], distribution: Dict[str, float]) -> Dict[str, float]:
    """
    Filter *distribution* using the PDA state derived from *tokens*.

    Returns the distribution with provably-invalid tokens removed.
    If the current position is not inside a ```python``` code block, returns
    *distribution* unchanged.
    """
    if not distribution:
        return distribution

    bracket_stack, block_stack, in_code, last_tok, after_def_class = _compute_state(tokens)

    if not in_code:
        return distribution

    in_loop = any(k in _LOOP_KEYWORDS for k in block_stack)
    in_func = any(k == "def" for k in block_stack)

    filtered: Dict[str, float] = {}
    for tok, prob in distribution.items():
        s = tok.strip()
        if not s:
            filtered[tok] = prob
            continue

        # ── Rule 1: bracket balancing ────────────────────────────────────────
        if s in _CLOSE:
            if not bracket_stack or bracket_stack[-1] != _MATCH[s]:
                continue  # no matching open bracket — mask it

        # ── Rule 2: no close bracket after a binary operator ─────────────────
        if last_tok in _BINARY_OPS and s in _CLOSE:
            continue

        # ── Rule 3: no binary operator immediately after another ─────────────
        # Exception: unary minus ("-") is always valid.
        if last_tok in _BINARY_OPS and s in _BINARY_OPS and s != "-":
            continue

        # ── Rule 4: after def/class, first token must be an identifier ────────
        if after_def_class and s in _CANNOT_FOLLOW_DEF_CLASS:
            continue

        # ── Rule 5: break/continue only valid inside a loop ──────────────────
        if s in ("break", "continue") and not in_loop:
            continue

        # ── Rule 6: return/yield only valid inside a function ────────────────
        if s in ("return", "yield") and not in_func:
            continue

        filtered[tok] = prob

    # Safety fallback: if every token was masked, return the original
    # distribution unchanged to prevent MCTS deadlock.
    return filtered if filtered else distribution
