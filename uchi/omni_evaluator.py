"""
Omni Evaluator
==============
Programmatic evaluation suite for the Uchi OmniRouter.

Provides reproducible, quantitative metrics across two domains:

**Coding Metrics**
  - pass_at_1          : Fraction of challenges solved on the first attempt.
  - self_correction_rate: Fraction solved after feeding tracebacks back (up to N retries).

**Conversational Metrics**
  - active_teaching_trigger_rate : How reliably the router triggers its
    "I do not have enough context" fallback on truly unknown prompts.
  - knowledge_recall_rate        : After teaching the router a new fact,
    can it recall the expected keyword?
  - average_prompt_entropy       : Mean bits-per-token across a set of
    prompts (lower = more confident / stable predictions).

Usage::

    from uchi.omni_router import OmniRouter
    from uchi.omni_evaluator import OmniEvaluator

    router = OmniRouter()
    evaluator = OmniEvaluator(router)
    results = evaluator.run_full_evaluation()
    evaluator.save_metrics(results, "eval_metrics.json")
"""

from __future__ import annotations

import ast
import datetime
import json
import math
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from uchi.omni_router import OmniRouter


# ─── Constants ────────────────────────────────────────────────────────────────

_SANDBOX_TIMEOUT_SECONDS = 10
_UNCERTAIN_PREFIX = "[Uncertain]"

# All strings that indicate the system correctly admitted it does not know.
# Three signals across architectural generations:
#   v0.1 legacy gate  → "I do not have enough context"
#   v0.2 hallucination gate → "I do not know the answer to that"
#   v0.3 ConvergentEngine → "[Uncertain] ..."
_FALLBACK_SIGNALS: tuple[str, ...] = (
    "i do not have enough context",
    "i do not know the answer to that",
    "[uncertain]",
)

# ─── Built-in Benchmark Data ─────────────────────────────────────────────────

CODING_CHALLENGES: list[dict] = [
    {
        "prompt": "write a python function called add that takes two numbers a and b and returns their sum",
        "assert_check": textwrap.dedent("""\
            assert add(2, 3) == 5
            assert add(-1, 1) == 0
            assert add(0, 0) == 0
        """),
    },
    {
        "prompt": "write a python function called reverse_string that takes a string s and returns it reversed",
        "assert_check": textwrap.dedent("""\
            assert reverse_string("hello") == "olleh"
            assert reverse_string("") == ""
            assert reverse_string("a") == "a"
        """),
    },
    {
        "prompt": "write a python function called is_even that takes an integer n and returns True if n is even else False",
        "assert_check": textwrap.dedent("""\
            assert is_even(4) is True
            assert is_even(3) is False
            assert is_even(0) is True
        """),
    },
    {
        "prompt": "write a python function called factorial that takes a non negative integer n and returns n factorial",
        "assert_check": textwrap.dedent("""\
            assert factorial(0) == 1
            assert factorial(1) == 1
            assert factorial(5) == 120
        """),
    },
    {
        "prompt": "write a python function called max_of_three that takes three numbers a b c and returns the largest",
        "assert_check": textwrap.dedent("""\
            assert max_of_three(1, 2, 3) == 3
            assert max_of_three(10, 5, 7) == 10
            assert max_of_three(-1, -2, -3) == -1
        """),
    },
]

UNKNOWN_PROMPTS: list[str] = [
    "explain the Boltzmann distribution in quantum chromodynamics",
    "what are the trade agreements between Wakanda and Latveria",
    "describe the lifecycle of a Zygothian marsupial",
    "translate this sentence into Klingon",
    "what is the current stock price of XYZ Corp",
    "how many moons does the planet Xerion have",
    "tell me about the 2087 Mars colony elections",
    "what is the airspeed velocity of an unladen swallow on Jupiter",
]

QA_PAIRS: list[dict] = [
    {
        "teach": "<|user|> the capital of France is Paris <|assistant|> understood the capital of France is Paris",
        "query": "what is the capital of France",
        "expected_keyword": "paris",
    },
    {
        "teach": "<|user|> water boils at 100 degrees celsius <|assistant|> understood water boils at 100 degrees celsius",
        "query": "at what temperature does water boil",
        "expected_keyword": "100",
    },
    {
        "teach": "<|user|> the speed of light is 299792458 meters per second <|assistant|> understood the speed of light is 299792458 meters per second",
        "query": "what is the speed of light",
        "expected_keyword": "299792458",
    },
    {
        "teach": "<|user|> Python was created by Guido van Rossum <|assistant|> understood Python was created by Guido van Rossum",
        "query": "who created Python",
        "expected_keyword": "guido",
    },
]

ENTROPY_PROMPTS: list[str] = [
    "hello",
    "who are you",
    "what can you do",
    "tell me a joke",
    "write a python function to add two numbers",
    "how do you work",
    "good morning",
]

# Queries that go through ConvergentEngine (no analytical intent_key).
# Used by convergent_oracle_pass_rate to measure how often the engine
# produces an oracle-valid response rather than [Uncertain].
CONVERGENT_PROMPTS: list[str] = [
    "hello how are you",
    "tell me something interesting",
    "what do you think about that",
    "can you explain what you are",
    "give me some advice",
    "describe yourself",
    "what is your purpose",
    "how can you help me",
]

# Tool routing evaluation: labeled queries for measuring precision / recall.
# Each entry has "query", "expected_tool" (True = should dispatch a skill,
# False = should produce conversational text).
TOOL_ROUTING_QUERIES: list[dict] = [
    # --- tool=True (must beat ABSOLUTE_FLOOR + MARGIN to pass at cold start) ---
    {"query": "classify my dataset",           "expected_tool": True},
    {"query": "run regression analysis",       "expected_tool": True},
    {"query": "detect anomalies in my data",   "expected_tool": True},
    {"query": "forecast the time series",      "expected_tool": True},
    {"query": "run time series classification","expected_tool": True},
    # --- tool=False (false-positive check) ---
    {"query": "hello how are you",             "expected_tool": False},
    {"query": "what is the meaning of life",   "expected_tool": False},
    {"query": "tell me a story",               "expected_tool": False},
    {"query": "explain machine learning",      "expected_tool": False},
    {"query": "good morning uchi",             "expected_tool": False},
]

# In-domain conversational prompts for the canary uncertainty rate.
CANARY_PROMPTS: list[str] = [
    "hello",
    "who are you",
    "how are you today",
    "tell me something",
    "what can you do",
    "explain yourself",
    "good morning",
    "thank you",
    "what is your name",
    "how do you work",
]

# Teach/query pairs used to measure MCTS efficiency (rollout count tracking).
# After streaming the teach sequence, the engine should converge faster on
# the known query than on a cold start.
EFFICIENCY_TEACH_PAIRS: list[dict] = [
    {
        "teach": "<|user|> hello <|assistant|> hello there how can i help you today",
        "query": "hello",
    },
    {
        "teach": "<|user|> who are you <|assistant|> i am uchi a deterministic sequence predictor",
        "query": "who are you",
    },
    {
        "teach": "<|user|> what can you do <|assistant|> i can answer questions and help with tasks",
        "query": "what can you do",
    },
]


# ─── Oracle Hindsight Experience Replay (AST Blame) ──────────────────────────

def _find_blame_token(tokens: list[str], error: SyntaxError) -> int:
    """
    Map a SyntaxError back to the token index that caused it.

    Uses the error's character offset when available; falls back to a binary
    search over token prefixes to isolate the first prefix that already fails
    ast.parse().  Returns the index of the first "bad" token (0-indexed).
    """
    # Attempt character-offset mapping
    if error.lineno is not None and error.offset is not None:
        target_char = max(0, error.offset - 1)  # 0-indexed
        cursor = 0
        for i, tok in enumerate(tokens):
            end = cursor + len(tok)
            if cursor <= target_char <= end:
                return i
            cursor = end + 1  # account for the joining space
        return len(tokens)

    # Binary search fallback
    lo, hi = 0, len(tokens)
    while lo < hi:
        mid = (lo + hi) // 2
        try:
            ast.parse(" ".join(tokens[: mid + 1]))
            lo = mid + 1
        except SyntaxError:
            hi = mid
    return lo


def oracle_ast_blame(tokens: list[str]) -> list[float]:
    """
    Assign per-token rewards using AST-based hindsight blame.

    Prevents the Punished Prefix Trap: instead of assigning a flat 0.0 to an
    entire failed sequence, the prefix up to (but not including) the first
    syntactically invalid token receives a positive 1.0 reward, so the value
    head learns to score valid prefixes correctly even when the tail fails.

    Args:
        tokens: flat list of string tokens representing a code sequence.

    Returns:
        List of floats with the same length as *tokens*:
          - 1.0 for every token in the valid prefix
          - 0.0 from the first syntax-error-causing token onward
        If the entire sequence is valid, all rewards are 1.0.
        If no prefix is valid, all rewards are 0.0.
    """
    if not tokens:
        return []

    code = " ".join(tokens)
    try:
        ast.parse(code)
        return [1.0] * len(tokens)
    except SyntaxError as exc:
        blame = _find_blame_token(tokens, exc)
        return [1.0] * blame + [0.0] * (len(tokens) - blame)


# ─── Sandbox helper ──────────────────────────────────────────────────────────

def _extract_code_block(text: str) -> str | None:
    """Extract the first ```python ... ``` block from *text*.

    Returns the code as a string or ``None`` if no block is found.
    """
    if "```python" not in text:
        return None
    parts = text.split("```python", 1)
    if len(parts) < 2:
        return None
    remainder = parts[1]
    # Handle both ``` on its own line and inline
    if "```" in remainder:
        code = remainder.split("```", 1)[0]
    else:
        code = remainder  # unterminated block – try anyway
    return code.strip()


def _run_code_in_sandbox(code: str, assert_check: str, timeout: int = _SANDBOX_TIMEOUT_SECONDS) -> tuple[bool, str]:
    """Execute *code* + *assert_check* in a subprocess sandbox.

    Returns ``(passed, output)`` where *output* contains stdout/stderr
    or a timeout message.
    """
    full_code = f"{code}\n\n{assert_check}\n"
    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="uchi_eval_")
        with os.fdopen(fd, "w") as f:
            fd = None  # os.fdopen takes ownership
            f.write(full_code)
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"[Sandbox Timeout] Execution exceeded {timeout}s"
    except Exception as exc:
        return False, f"[Sandbox Error] {exc}"
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ─── OmniEvaluator ───────────────────────────────────────────────────────────

class OmniEvaluator:
    """Comprehensive evaluation harness for :class:`OmniRouter`.

    Parameters
    ----------
    router : OmniRouter
        A fully initialised (persona-bootstrapped) OmniRouter instance.
    verbose : bool
        If ``True`` (default), print progress and per-challenge diagnostics
        to *stdout*.
    """

    def __init__(self, router: OmniRouter, *, verbose: bool = True) -> None:
        self.router = router
        self.verbose = verbose

    # ── Helpers ───────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            sys.stdout.write(msg + "\n")
            sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════════════
    # Coding Metrics
    # ══════════════════════════════════════════════════════════════════════

    def pass_at_1(self, challenges: list[dict] | None = None) -> float:
        """Fraction of coding challenges passed on the first attempt.

        Each element of *challenges* must contain:

        - ``"prompt"`` – natural-language description of the function to write.
        - ``"assert_check"`` – Python assertions that exercise the generated
          function.

        The router's ``chat()`` method is called with the prompt.  If the
        reply contains a ``\`\`\`python`` block, it is extracted, concatenated
        with the assertions, and run in an isolated subprocess.
        """
        if challenges is None:
            challenges = CODING_CHALLENGES

        if not challenges:
            return 0.0

        passed = 0
        self._log("\n" + "=" * 60)
        self._log("  METRIC: pass@1")
        self._log("=" * 60)

        for i, ch in enumerate(challenges, 1):
            prompt = ch["prompt"]
            assert_check = ch["assert_check"]
            self._log(f"\n  [{i}/{len(challenges)}] {prompt[:72]}…")

            try:
                reply = self.router.chat(prompt)
            except Exception as exc:
                self._log(f"    ✗ Router error: {exc}")
                continue

            code = _extract_code_block(reply)
            if code is None:
                self._log("    ✗ No ```python block in reply")
                continue

            ok, output = _run_code_in_sandbox(code, assert_check)
            if ok:
                self._log("    ✓ PASSED")
                passed += 1
            else:
                self._log(f"    ✗ FAILED – {output[:120]}")

        rate = passed / len(challenges)
        self._log(f"\n  pass@1 = {passed}/{len(challenges)} = {rate:.2%}")
        return rate

    def self_correction_rate(
        self,
        challenges: list[dict] | None = None,
        max_retries: int = 3,
    ) -> float:
        """Fraction of challenges that *eventually* pass after feeding
        tracebacks back to the router.

        On failure the traceback is wrapped in a ``<|system|>`` token and
        sent back via ``chat()`` for up to *max_retries* additional
        attempts.
        """
        if challenges is None:
            challenges = CODING_CHALLENGES

        if not challenges:
            return 0.0

        passed = 0
        self._log("\n" + "=" * 60)
        self._log(f"  METRIC: self_correction_rate  (max_retries={max_retries})")
        self._log("=" * 60)

        for i, ch in enumerate(challenges, 1):
            prompt = ch["prompt"]
            assert_check = ch["assert_check"]
            self._log(f"\n  [{i}/{len(challenges)}] {prompt[:72]}…")

            solved = False
            current_prompt = prompt

            for attempt in range(1, max_retries + 1):
                try:
                    reply = self.router.chat(current_prompt)
                except Exception as exc:
                    self._log(f"    attempt {attempt}: Router error – {exc}")
                    break

                code = _extract_code_block(reply)
                if code is None:
                    self._log(f"    attempt {attempt}: No code block")
                    # Ask for a correction
                    current_prompt = (
                        f"<|system|> Your previous reply did not contain a "
                        f"```python code block. Please try again: {prompt}"
                    )
                    continue

                ok, output = _run_code_in_sandbox(code, assert_check)
                if ok:
                    self._log(f"    attempt {attempt}: ✓ PASSED")
                    solved = True
                    break
                else:
                    self._log(f"    attempt {attempt}: ✗ {output[:100]}")
                    # Feed the error back
                    current_prompt = (
                        f"<|system|> The code you wrote raised an error:\n"
                        f"{output[:500]}\n\nPlease fix the code for: {prompt}"
                    )

            if solved:
                passed += 1

        rate = passed / len(challenges)
        self._log(f"\n  self_correction_rate = {passed}/{len(challenges)} = {rate:.2%}")
        return rate

    # ══════════════════════════════════════════════════════════════════════
    # Conversational Metrics
    # ══════════════════════════════════════════════════════════════════════

    def active_teaching_trigger_rate(
        self,
        unknown_prompts: list[str] | None = None,
    ) -> float:
        """Fraction of truly-unknown prompts where the router correctly
        admits it does not know, rather than hallucinating a confident answer.

        Accepts any of the three architectural uncertainty signals:
          v0.1 legacy   → "I do not have enough context"
          v0.2 legacy   → "I do not know the answer to that"
          v0.3 CMOF     → "[Uncertain] ..."

        Target: 100 %.
        """
        if unknown_prompts is None:
            unknown_prompts = UNKNOWN_PROMPTS

        if not unknown_prompts:
            return 0.0

        triggered = 0
        self._log("\n" + "=" * 60)
        self._log("  METRIC: active_teaching_trigger_rate")
        self._log("=" * 60)

        for i, prompt in enumerate(unknown_prompts, 1):
            self._log(f"\n  [{i}/{len(unknown_prompts)}] {prompt[:72]}…")

            try:
                reply = self.router.chat(prompt)
            except Exception as exc:
                self._log(f"    ✗ Router error: {exc}")
                continue

            reply_str = str(reply) if reply else ""
            reply_lower = reply_str.lower()

            if any(signal in reply_lower for signal in _FALLBACK_SIGNALS):
                self._log("    ✓ Correctly triggered fallback")
                triggered += 1
            else:
                self._log(f"    ✗ Hallucinated: {reply_str[:80]}")

        rate = triggered / len(unknown_prompts)
        self._log(f"\n  active_teaching_trigger_rate = {triggered}/{len(unknown_prompts)} = {rate:.2%}")
        return rate

    def knowledge_recall_rate(
        self,
        qa_pairs: list[dict] | None = None,
    ) -> float:
        """Teach the router new facts, then query and check recall.

        Each element of *qa_pairs* must contain:

        - ``"teach"`` – a string of tokens to stream into the router
          (typically a ``<|user|> … <|assistant|> …`` turn).
        - ``"query"`` – a natural-language question to send via ``chat()``.
        - ``"expected_keyword"`` – a case-insensitive keyword that should
          appear in the reply if recall succeeded.
        """
        if qa_pairs is None:
            qa_pairs = QA_PAIRS

        if not qa_pairs:
            return 0.0

        recalled = 0
        self._log("\n" + "=" * 60)
        self._log("  METRIC: knowledge_recall_rate")
        self._log("=" * 60)

        for i, pair in enumerate(qa_pairs, 1):
            teach_tokens = pair["teach"].split()
            query = pair["query"]
            keyword = pair["expected_keyword"].lower()

            self._log(f"\n  [{i}/{len(qa_pairs)}] Teaching: {pair['teach'][:60]}…")

            # Teach by streaming the fact multiple times for reinforcement
            for _ in range(10):
                self.router.stream(teach_tokens)

            self._log(f"    Querying: {query}")

            try:
                reply = self.router.chat(query)
            except Exception as exc:
                self._log(f"    ✗ Router error: {exc}")
                continue

            reply_str = str(reply).lower() if reply else ""

            if keyword in reply_str:
                self._log(f"    ✓ Recalled (found '{keyword}')")
                recalled += 1
            else:
                self._log(f"    ✗ Missed (expected '{keyword}', got: {reply_str[:80]})")

        rate = recalled / len(qa_pairs)
        self._log(f"\n  knowledge_recall_rate = {recalled}/{len(qa_pairs)} = {rate:.2%}")
        return rate

    def average_prompt_entropy(
        self,
        prompts: list[str] | None = None,
    ) -> float:
        """Average bits-per-token (``predictor.score()``) across *prompts*.

        Lower values indicate the model is more confident / stable on
        the given prompts.  Very high values (> 12) indicate the prompt
        lies outside the model's learned distribution.
        """
        if prompts is None:
            prompts = ENTROPY_PROMPTS

        if not prompts:
            return float("inf")

        self._log("\n" + "=" * 60)
        self._log("  METRIC: average_prompt_entropy")
        self._log("=" * 60)

        scores: list[float] = []
        for i, prompt in enumerate(prompts, 1):
            tokens = prompt.split()
            # Tokenize through the router's pipeline so the predictor
            # sees the same representation it was trained on.
            concepts = self.router.tokenizer.tokenize(tokens, is_inference=True)
            full_tokens = ["<|user|>"] + list(concepts)

            try:
                entropy = self.router.predictor.score(full_tokens)
            except Exception as exc:
                self._log(f"  [{i}] Error scoring '{prompt[:40]}': {exc}")
                continue

            # Guard against pathological infinities
            if math.isinf(entropy) or math.isnan(entropy):
                self._log(f"  [{i}] '{prompt[:40]}' → inf/nan (skipped)")
                continue

            self._log(f"  [{i}] '{prompt[:40]}' → {entropy:.4f} bits/token")
            scores.append(entropy)

        if not scores:
            self._log("\n  average_prompt_entropy = N/A (no valid scores)")
            return float("inf")

        avg = sum(scores) / len(scores)
        self._log(f"\n  average_prompt_entropy = {avg:.4f} bits/token")
        return avg

    # ══════════════════════════════════════════════════════════════════════
    # CMOF / Vector-Routing Metrics  (added v0.3.0)
    # ══════════════════════════════════════════════════════════════════════

    def convergent_oracle_pass_rate(
        self,
        prompts: list[str] | None = None,
    ) -> float:
        """Fraction of prompts where ConvergentEngine returns kind != 'uncertain'.

        Directly measures whether the MCTS deliberation + oracle filter
        produces any usable candidates.  A value of 0.0 means every
        query fell through to the [Uncertain] fallback.

        Baseline (cold start): 0 – 30 %.
        Target (trained):      > 60 %.
        """
        if prompts is None:
            prompts = CONVERGENT_PROMPTS

        if not prompts:
            return 0.0

        passed = 0
        self._log("\n" + "=" * 60)
        self._log("  METRIC: convergent_oracle_pass_rate")
        self._log("=" * 60)

        for i, prompt in enumerate(prompts, 1):
            self._log(f"\n  [{i}/{len(prompts)}] {prompt[:72]}")
            try:
                concepts = self.router.tokenizer.tokenize(
                    prompt.split(), is_inference=True
                )
                kind, _, _ = self.router.convergent.generate(
                    list(concepts),
                    bootstrapped=self.router._knowledge_bootstrapped,
                )
                if kind != "uncertain":
                    self._log(f"    ✓ kind={kind}")
                    passed += 1
                else:
                    self._log("    ✗ uncertain")
            except Exception as exc:
                self._log(f"    ✗ error: {exc}")

        rate = passed / len(prompts)
        self._log(f"\n  convergent_oracle_pass_rate = {passed}/{len(prompts)} = {rate:.2%}")
        return rate

    def mcts_efficiency_score(
        self,
        teach_pairs: list[dict] | None = None,
        teach_reps: int = 20,
    ) -> float:
        """Average fraction of MAX_BUDGET rollouts consumed per known query.

        Lower is better: 0.0 = always exits at MIN_ROLLOUTS (perfectly
        converged), 1.0 = always exhausts the full budget (no convergence).

        Steps:
          1. Stream each teach pair *teach_reps* times to prime the trie.
          2. Run convergent.generate() on the known query while counting
             calls to predictor.generate via a lightweight wrapper.
          3. Compute consumed / MAX_BUDGET for each query.

        Baseline (cold start):  ~1.0  (no early stopping).
        Target (trained):       < 0.2 (exits at MIN_ROLLOUTS + a few).
        """
        from uchi.convergent_engine import MAX_BUDGET

        if teach_pairs is None:
            teach_pairs = EFFICIENCY_TEACH_PAIRS

        if not teach_pairs:
            return 1.0

        self._log("\n" + "=" * 60)
        self._log(f"  METRIC: mcts_efficiency_score  (teach_reps={teach_reps})")
        self._log("=" * 60)

        fractions: list[float] = []

        for i, pair in enumerate(teach_pairs, 1):
            teach_tokens = pair["teach"].split()
            query = pair["query"]

            self._log(f"\n  [{i}/{len(teach_pairs)}] Teaching and querying: '{query}'")

            # Prime the trie
            for _ in range(teach_reps):
                self.router.stream(teach_tokens)

            # Count predictor.generate() calls during one deliberation
            original_gen = self.router.predictor.generate
            call_count: list[int] = [0]

            def _counting_gen(*args, **kwargs):
                call_count[0] += 1
                return original_gen(*args, **kwargs)

            self.router.predictor.generate = _counting_gen
            try:
                concepts = self.router.tokenizer.tokenize(
                    query.split(), is_inference=True
                )
                self.router.convergent.generate(
                    list(concepts),
                    bootstrapped=self.router._knowledge_bootstrapped,
                )
            except Exception as exc:
                self._log(f"    error: {exc}")
            finally:
                self.router.predictor.generate = original_gen

            n = call_count[0]
            frac = min(n / max(MAX_BUDGET, 1), 1.0)
            fractions.append(frac)
            self._log(f"    rollouts={n} / budget={MAX_BUDGET}  →  {frac:.2%} of budget consumed")

        score = sum(fractions) / len(fractions)
        self._log(f"\n  mcts_efficiency_score = {score:.2%} of budget consumed (lower is better)")
        return score

    def tool_routing_precision_recall(
        self,
        labeled_queries: list[dict] | None = None,
    ) -> dict[str, float]:
        """Precision and recall of ConvergentEngine's tool dispatch.

        Each entry must have:
          "query"         — natural-language string
          "expected_tool" — True if a skill should be dispatched

        Precision = TP / (TP + FP)  — of dispatched calls, how many were correct?
        Recall    = TP / (TP + FN)  — of tool queries, how many were caught?

        At cold start, ABSOLUTE_FLOOR = 0.3 suppresses all tool dispatch:
        precision = 1.0 (undefined; no dispatches made → reported as 1.0),
        recall    = 0.0.

        Returns dict with keys: "precision", "recall", "tp", "fp", "fn".
        """
        if labeled_queries is None:
            labeled_queries = TOOL_ROUTING_QUERIES

        tp = fp = fn = 0
        self._log("\n" + "=" * 60)
        self._log("  METRIC: tool_routing_precision_recall")
        self._log("=" * 60)

        for entry in labeled_queries:
            query = entry["query"]
            expected = entry["expected_tool"]
            self._log(f"\n  [{'+' if expected else '-'}] {query[:60]}")
            try:
                concepts = self.router.tokenizer.tokenize(
                    query.split(), is_inference=True
                )
                kind, _, _ = self.router.convergent.generate(
                    list(concepts),
                    bootstrapped=self.router._knowledge_bootstrapped,
                )
                got_tool = (kind == "tool")
                if expected and got_tool:
                    tp += 1
                    self._log("      TP ✓ correctly dispatched")
                elif expected and not got_tool:
                    fn += 1
                    self._log(f"      FN ✗ missed tool (got kind={kind})")
                elif not expected and got_tool:
                    fp += 1
                    self._log("      FP ✗ false tool dispatch")
                else:
                    self._log(f"      TN ✓ correctly text/uncertain (kind={kind})")
            except Exception as exc:
                self._log(f"      error: {exc}")

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        self._log(
            f"\n  precision={precision:.2%}  recall={recall:.2%}"
            f"  (TP={tp} FP={fp} FN={fn})"
        )
        return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}

    def uncertain_canary_rate(
        self,
        prompts: list[str] | None = None,
    ) -> float:
        """Fraction of in-domain queries that return an [Uncertain] prefix.

        This is the canary metric: it should decrease over time as the trie
        learns.  A spike here after a code change indicates a broad regression
        (oracle became too strict, MCTS budget too low, trie overfit, etc.).

        Baseline (cold start):  50 – 80 %.
        Target (trained):       < 30 %.
        """
        if prompts is None:
            prompts = CANARY_PROMPTS

        if not prompts:
            return 0.0

        uncertain_count = 0
        self._log("\n" + "=" * 60)
        self._log("  METRIC: uncertain_canary_rate")
        self._log("=" * 60)

        for i, prompt in enumerate(prompts, 1):
            self._log(f"\n  [{i}/{len(prompts)}] {prompt[:72]}")
            try:
                reply = self.router.chat(prompt)
                reply_str = str(reply) if reply else ""
                if reply_str.startswith(_UNCERTAIN_PREFIX):
                    uncertain_count += 1
                    self._log(f"    [Uncertain] {reply_str[len(_UNCERTAIN_PREFIX):].strip()[:60]}")
                else:
                    self._log(f"    [OK] {reply_str[:60]}")
            except Exception as exc:
                self._log(f"    error: {exc}")

        rate = uncertain_count / len(prompts)
        self._log(f"\n  uncertain_canary_rate = {uncertain_count}/{len(prompts)} = {rate:.2%}")
        return rate

    def contrastive_loss_trend(self) -> dict[str, float]:
        """Summary statistics for the contrastive loss history buffer.

        Returns mean, last value, and whether loss is trending downward
        (last-quarter mean < first-quarter mean).  An empty or near-zero
        buffer means the background updater hasn't fired yet.

        Not added to the persistent JSON since it's in-memory only.
        """
        from uchi.vector_oracle import _contrastive_loss_history

        history = list(_contrastive_loss_history)
        if len(history) < 2:
            self._log("\n  contrastive_loss_trend: insufficient data (<2 samples)")
            return {"mean": float("nan"), "last": float("nan"), "trending_down": False, "n": len(history)}

        mean_val = sum(history) / len(history)
        last_val = history[-1]
        q = max(len(history) // 4, 1)
        first_q_mean = sum(history[:q]) / q
        last_q_mean  = sum(history[-q:]) / q
        trending_down = last_q_mean < first_q_mean

        self._log(
            f"\n  contrastive_loss_trend: n={len(history)}"
            f"  mean={mean_val:.4f}  last={last_val:.4f}"
            f"  trending={'↓' if trending_down else '→/↑'}"
        )
        return {
            "mean": mean_val,
            "last": last_val,
            "trending_down": trending_down,
            "n": len(history),
        }

    # ══════════════════════════════════════════════════════════════════════
    # Aggregate Evaluation
    # ══════════════════════════════════════════════════════════════════════

    def run_full_evaluation(self) -> dict[str, Any]:
        """Execute every metric with built-in benchmark data.

        Returns a flat dictionary of results suitable for JSON
        serialisation (via :meth:`save_metrics`).
        """
        self._log("\n" + "╔" + "═" * 58 + "╗")
        self._log("║" + "  UCHI OMNI-EVALUATOR  —  Full Evaluation Suite".center(58) + "║")
        self._log("╚" + "═" * 58 + "╝")

        results: dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "uchi_version": _get_version(),
        }

        # — Coding —
        results["pass_at_1"] = self.pass_at_1()
        results["self_correction_rate"] = self.self_correction_rate()

        # — Conversation —
        results["active_teaching_trigger_rate"] = self.active_teaching_trigger_rate()
        results["knowledge_recall_rate"] = self.knowledge_recall_rate()
        results["average_prompt_entropy"] = self.average_prompt_entropy()

        # — CMOF / Vector routing (v0.3.0) —
        results["convergent_oracle_pass_rate"] = self.convergent_oracle_pass_rate()
        results["mcts_efficiency_score"] = self.mcts_efficiency_score()
        routing = self.tool_routing_precision_recall()
        results["tool_routing_precision"] = routing["precision"]
        results["tool_routing_recall"]    = routing["recall"]
        results["uncertain_canary_rate"]  = self.uncertain_canary_rate()

        # — Summary —
        self._log("\n" + "╔" + "═" * 58 + "╗")
        self._log("║" + "  RESULTS SUMMARY".center(58) + "║")
        self._log("╠" + "═" * 58 + "╣")
        for key, label in [
            ("pass_at_1",                   "pass@1 (code)"),
            ("self_correction_rate",         "self-correction (code)"),
            ("active_teaching_trigger_rate", "active teaching trigger"),
            ("knowledge_recall_rate",        "knowledge recall"),
            ("average_prompt_entropy",       "avg prompt entropy"),
            ("convergent_oracle_pass_rate",  "convergent oracle pass"),
            ("mcts_efficiency_score",        "MCTS budget consumed"),
            ("tool_routing_precision",       "tool routing precision"),
            ("tool_routing_recall",          "tool routing recall"),
            ("uncertain_canary_rate",        "canary [Uncertain] rate"),
        ]:
            val = results.get(key, float("nan"))
            if key == "average_prompt_entropy":
                formatted = f"{val:.4f} bits/tok"
            elif key == "mcts_efficiency_score":
                formatted = f"{val:.2%} of budget"
            elif isinstance(val, float):
                formatted = f"{val:.2%}"
            else:
                formatted = str(val)
            self._log(f"║  {label:<40s} {formatted:>14s}  ║")
        self._log("╚" + "═" * 58 + "╝\n")

        return results

    # ══════════════════════════════════════════════════════════════════════
    # Persistence
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def save_metrics(results: dict[str, Any], path: str = "eval_metrics.json") -> None:
        """Append timestamped *results* to a JSON file at *path*.

        The file stores a JSON array; each call appends one entry.
        If the file does not exist it is created.
        """
        path_obj = Path(path)

        history: list[dict] = []
        if path_obj.exists():
            try:
                with open(path_obj, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        loaded = json.loads(content)
                        # Support both list and single-dict legacy formats
                        if isinstance(loaded, list):
                            history = loaded
                        elif isinstance(loaded, dict):
                            history = [loaded]
            except (json.JSONDecodeError, OSError):
                # Corrupted file – start fresh but keep a backup
                backup = path_obj.with_suffix(".json.bak")
                try:
                    path_obj.rename(backup)
                except OSError:
                    pass

        history.append(results)

        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=str)


# ─── Utilities ────────────────────────────────────────────────────────────────

def _get_version() -> str:
    try:
        import uchi
        return getattr(uchi, "__version__", "unknown")
    except Exception:
        return "unknown"
