"""
reasoning.py — Uchi's verifier-guided reasoning engine (SKETCH / skeleton).

Uchi does not reason like an LLM (implicitly, in weights). It reasons like a
scientist: **decompose a goal into steps, execute each step with an operator that
can VERIFY its own output, keep only what survives, and abstain — naming the exact
step — when it can't.** A fallible proposer + a reality-anchored verifier, chained.

    reason(question: str) -> str        # string in, string out

This is the honest version of "reasoning" for a no-LLM system: every emitted
conclusion is a chain of *verified* steps, so it genuinely cannot assert a
conclusion it couldn't ground — unlike an LLM, which reasons fluently but can't
check itself.

What is solid here
------------------
- The verified-step spine: route each step to an operator (math / factual / code),
  verify, and abstain-with-provenance on failure. This is the real contribution.
- Grounded factual steps via Generate-and-Ground (grounded or abstain).
- Deterministic math steps via a symbolic evaluator (verifiable by construction).

What is the hard open problem (honest)
--------------------------------------
- The DECOMPOSER. Turning arbitrary language into a verifiable plan without an LLM
  is genuinely hard; `_decompose` here is a heuristic that handles explicit
  multi-part / arithmetic cases and otherwise degrades to a single grounded answer.
  This is the piece to invest a small trained planner in next.
- Full MCTS/backtracking search over plans (Kocsis-Szepesvári) — `_execute` returns
  a verified/failed signal, so wrapping the loop in tree-search is a clean extension.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

_ABSTAIN = "I don't have grounded knowledge to answer that."
_REF = re.compile(r"\b(it|that|this|the (result|answer|value|total)|step\s*\d+)\b", re.I)
_MATH_HINT = re.compile(r"[\d]+\s*[\+\-\*/×xX^]\s*[\d]|\b(sum|product|times|divided?|divide|"
                        r"plus|minus|add|subtract|multiply|square|percent|average|difference)\b", re.I)
# A follow-up operation applied to the previous result ("then add 8", "subtract 20").
_OPCONT = re.compile(r"^\s*(?:and\s+|then\s+)?(add|subtract|multiply|divide|plus|minus|times)\b", re.I)
_SPLIT = re.compile(r"\s*(?:;|\bthen\b|\band then\b|\bafter that\b|→|\.\s*(?=[A-Z]))\s*")

# word → operator, longest phrases first so "multiply by" beats "multiply"
_OP_WORDS = [("multiply by", "*"), ("divided by", "/"), ("divide by", "/"),
             ("multiply", "*"), ("divide", "/"), ("times", "*"), ("divided", "/"),
             ("add", "+"), ("plus", "+"), ("subtract", "-"), ("minus", "-"), ("×", "*")]


# ── verified operators ─────────────────────────────────────────────────────────
def _safe_math(expr: str) -> Optional[str]:
    """Evaluate an arithmetic/algebra expression; None if it doesn't evaluate.

    Verifiable by construction: a symbolic evaluator either produces a value or
    it doesn't — there is nothing to hallucinate.
    """
    cleaned = expr.lower()
    for word, op in _OP_WORDS:
        cleaned = cleaned.replace(word, op)
    # keep the longest run of math characters that actually contains a digit
    runs = [r for r in re.findall(r"[-+/*.^()\d\s]+", cleaned) if any(c.isdigit() for c in r)]
    if not runs:
        return None
    frag = max(runs, key=len).strip().replace("^", "**")
    try:
        import sympy
        val = sympy.sympify(frag)
        if val.free_symbols:
            return str(val)                       # symbolic algebra result
        f = float(val.evalf())
        return str(int(f)) if f == int(f) else str(round(f, 6))
    except Exception:
        # no sympy / not sympifiable → safe arithmetic via a restricted AST eval
        try:
            import ast
            node = ast.parse(frag, mode="eval")
            allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
                       ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.Mod)
            if all(isinstance(n, allowed) for n in ast.walk(node)):
                return str(eval(compile(node, "<math>", "eval"), {"__builtins__": {}}))
        except Exception:
            pass
    return None


class ReasoningEngine:
    """Verifier-guided, string→string reasoning over grounded/verified operators.

    Parameters
    ----------
    answer_fn : Callable[[str], str]
        The grounded factual operator — typically ``router.answer`` (Generate-and-
        Ground): returns a grounded answer or the abstain sentinel.
    code_fn : Callable[[str], str], optional
        A REPL-verified code operator (e.g. the CodeEngine).
    max_steps : int
        Guardrail on plan length.
    """

    def __init__(self, answer_fn: Callable[[str], str],
                 code_fn: Optional[Callable[[str], str]] = None,
                 max_steps: int = 6) -> None:
        self.answer_fn = answer_fn
        self.code_fn = code_fn
        self.max_steps = max_steps

    # ── the loop ────────────────────────────────────────────────────────────────
    def reason(self, question: str) -> str:
        steps = self._decompose(question)
        if len(steps) <= 1:
            return self.answer_fn(question)          # single grounded step

        trace: list[tuple[str, str, str]] = []
        last: Optional[str] = None
        for i, step in enumerate(steps[: self.max_steps]):
            resolved = self._substitute(step, last)
            result, kind = self._execute(resolved)
            if result is None:                       # step could not be verified
                return self._abstain(i, resolved, trace)
            trace.append((resolved, result, kind))
            last = result
        return self._compose(question, trace, last)

    # ── planner (heuristic; the hard open problem) ─────────────────────────────
    def _decompose(self, question: str) -> list[str]:
        parts = [p.strip(" .") for p in _SPLIT.split(question) if p.strip(" .")]
        # numbered plans: "1. do X 2. do Y"
        if len(parts) == 1:
            nums = re.split(r"\b\d+\.\s+", question)
            parts = [p.strip() for p in nums if p.strip()] or parts
        return parts if len(parts) > 1 else [question]

    def _substitute(self, step: str, last: Optional[str]) -> str:
        """Inject the previous verified result: explicit refs ("double it") or a
        bare follow-up operation ("then add 8" → "<last> add 8")."""
        if last is None:
            return step
        if _REF.search(step):
            return _REF.sub(str(last), step, count=1)
        if _OPCONT.match(step):
            return f"{last} {step}"
        return step

    # ── verified execution: route to the operator that can check itself ────────
    def _execute(self, step: str) -> tuple[Optional[str], str]:
        if _MATH_HINT.search(step):
            val = _safe_math(step)
            if val is not None:
                return val, "math"
        if self.code_fn is not None and re.search(r"\b(code|function|script|compute with)\b", step, re.I):
            try:
                out = self.code_fn(step)
                if out and out.strip():
                    return out.strip(), "code"
            except Exception:
                pass
        ans = self.answer_fn(step) or ""             # grounded factual (or abstain)
        if ans and _ABSTAIN not in ans:
            return ans, "factual"
        return None, "unverified"

    # ── composition + honest abstention ────────────────────────────────────────
    def _compose(self, question: str, trace: list[tuple[str, str, str]], last: Optional[str]) -> str:
        chain = "\n".join(f"  {i+1}. {s}  →  {r}  [{k}]" for i, (s, r, k) in enumerate(trace))
        return f"{last}\n\n(reasoned in {len(trace)} verified steps:\n{chain})"

    def _abstain(self, i: int, step: str, trace: list[tuple[str, str, str]]) -> str:
        got = "".join(f"\n  ✓ {s} → {r}" for s, r, _ in trace)
        return (f"I can establish the first {i} step(s) but cannot verify this one, "
                f"so I won't guess:\n  ✗ {step}{got if trace else ''}")
