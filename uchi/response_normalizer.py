"""
response_normalizer.py
======================
Deterministic post-processing pass applied to every string returned by
``Uchi.ask()``.

The trie operates on an internal vocabulary of WordNet synset tokens
(``water.n.01``), BPE-compressed compound words (``united_states``), and
routing control tokens (``<|assistant|>``, ``[Uncertain]``).  None of these
should reach the caller.  This module strips them and repairs basic sentence
structure so that:

  - Humans can read the output naturally.
  - ``u2.learn(u1.ask(...))`` chains feed clean English into the trie rather
    than raw trie vocabulary — creating a self-reinforcing quality loop.

All transforms are regex-based and deterministic.  No LLM, no model weights,
no external dependencies beyond the Python standard library.
"""

from __future__ import annotations

import re

# ── compiled patterns ─────────────────────────────────────────────────────────

# WordNet synset markers:  water.n.01  run.v.03  happy.a.01  quickly.r.01
_SYNSET = re.compile(r"\b(\w[\w'-]*)\.(n|v|a|r|s)\.\d{2}\b")

# Internal routing / control tokens
_CONTROL = re.compile(
    r"<\|(?:user|assistant|inner_monologue|system)[^|]*\|>"
    r"|\[(?:Uncertain|uncertain|UNCERTAIN)\]"
    r"|<\|[^|>]{1,40}\|>",           # any remaining <|...|> marker
    re.IGNORECASE,
)

# Repeated punctuation / artefacts from tokeniser merge  (e.g. "....", ",,")
_PUNCT_REPEAT = re.compile(r"([.!?,;])\1{1,}")

# Space before punctuation artefact:  "word ." → "word."
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.!?,;:])")

# Sentence boundary:  one of [.!?] followed by whitespace then a lower-case letter
_SENTENCE_START = re.compile(r"([.!?]\s+)([a-z])")

# Underscore-joined OmniTokenizer compound tokens:  united_states → united states
# Only applies when surrounded by word characters (not in URLs or file paths).
_UNDERSCORE_COMPOUND = re.compile(r"(?<=[a-zA-Z])_(?=[a-zA-Z])")


# ── public API ────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Convert raw ``Uchi`` trie output to human-readable text.

    Safe to call on any string — returns the input unchanged if it is empty
    or already clean.

    Transforms applied in order
    ---------------------------
    1. Strip internal control tokens (``<|assistant|>``, ``[Uncertain]``, …)
    2. De-synset WordNet markers  (``water.n.01`` → ``water``)
    3. Expand underscore compounds  (``united_states`` → ``united states``)
    4. Collapse whitespace
    5. Capitalise the first letter of each sentence
    6. Repair punctuation artefacts
    7. Ensure the response ends with a sentence-terminating character

    Parameters
    ----------
    text : str
        Raw string from the trie / OmniRouter.

    Returns
    -------
    str
        Clean, human-readable string.
    """
    if not text or not text.strip():
        return text

    # 1. Strip control tokens first so they don't interfere with other patterns
    text = _CONTROL.sub("", text)

    # 2. De-synset: keep the base lemma, drop the POS+sense tag
    text = _SYNSET.sub(r"\1", text)

    # 3. Underscore compounds → spaces
    text = _UNDERSCORE_COMPOUND.sub(" ", text)

    # 4. Collapse runs of whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 5. Capitalise sentence starts
    text = _capitalise_sentences(text)

    # 6. Repair punctuation artefacts
    text = _PUNCT_REPEAT.sub(r"\1", text)        # "..." → "."
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)  # "word ." → "word."

    # 7. Ensure terminal punctuation
    if text and text[-1] not in ".!?":
        text += "."

    return text


# ── private helpers ───────────────────────────────────────────────────────────

def _capitalise_sentences(text: str) -> str:
    """Upper-case the first character of the string and after each [.!?]."""
    if not text:
        return text
    text = text[0].upper() + text[1:]
    text = _SENTENCE_START.sub(lambda m: m.group(1) + m.group(2).upper(), text)
    return text
