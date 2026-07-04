"""
intent_router.py — classify a message into one of three lanes.

    skill    — an analytical / code command (/classify, /forecast, "run anomaly on…")
    social   — chit-chat: greetings, thanks, small talk, opinions, feelings.
               Safe to free-generate: it asserts no facts, so it needs NO oracle.
    factual  — a question with a truth value → Generate-and-Ground (grounded / abstain).

The split is the whole trick to giving Uchi a personality without weakening its
honesty: only the *factual* lane goes through the anti-confabulation machinery.
Social replies have no ground truth to violate, so they can be generated freely.
"""
from __future__ import annotations

import re

_SKILL_INTENTS = {"classify", "regress", "anomaly", "forecast", "tsclassify", "code"}

_GREET = re.compile(
    r"^\s*(hi|hey+|hello|yo|sup|howdy|hiya|good\s*(morning|afternoon|evening|day)|"
    r"greetings)\b", re.I)
_FAREWELL = re.compile(r"\b(bye|goodbye|see\s*(you|ya)|later|good\s*night|farewell|take care)\b", re.I)
_THANKS = re.compile(r"\b(thanks|thank you|thx|appreciate it|cheers)\b", re.I)
_HOWAREYOU = re.compile(r"\b(how are you|how's it going|how are things|what'?s up|how do you do|"
                        r"how have you been)\b", re.I)
_FEELING = re.compile(r"^\s*(i('?m| am| feel)|that'?s|you'?re|that is|so|wow|nice|cool|great|awesome|"
                      r"lol|haha|ok(ay)?|sure|yeah|yep|nope|no problem)\b", re.I)
_SMALLTALK = re.compile(r"\b(tell me (a )?(joke|story)|who are you|what'?s your name|do you like|"
                        r"favorite|your opinion|what do you think|nice to meet)\b", re.I)

# Markers that signal a factual question (a truth-valued query).
_FACTUAL_Q = re.compile(
    r"\b(what|which|who|whom|whose|when|where|why|how)\b.*\b(is|are|was|were|do|does|did|can|"
    r"will|would|should|means?|caused?|happens?|works?)\b", re.I)
_DEFINE = re.compile(r"\b(define|definition of|explain|what is a|what are|meaning of|how (do|does|to))\b", re.I)


def is_social(message: str) -> bool:
    m = message.strip()
    if not m:
        return False
    # Unambiguous small talk wins first ("how are you?" looks like a wh-question).
    if _HOWAREYOU.search(m) or _THANKS.search(m) or _FAREWELL.search(m) or _SMALLTALK.search(m):
        return True
    # An embedded factual question overrides a greeting prefix
    # ("hey, who invented the phone?" → factual).
    if _FACTUAL_Q.search(m) or _DEFINE.search(m):
        return False
    if _GREET.match(m):
        return True
    # short, opinion/feeling-shaped, and not a factual question → social
    if len(m.split()) <= 8 and _FEELING.match(m) and "?" not in m:
        return True
    return False


def classify_intent(message: str, procedural=None) -> str:
    """Return 'skill', 'social', or 'factual'."""
    if procedural is not None:
        try:
            k = procedural.get_intent_key(message)
        except Exception:
            k = None
        if k in _SKILL_INTENTS:
            return "skill"
    if is_social(message):
        return "social"
    return "factual"
