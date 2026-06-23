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

import datetime
import json
import math
import os
import subprocess
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path
from typing import Any

from uchi.omni_router import OmniRouter


# ─── Constants ────────────────────────────────────────────────────────────────

_SANDBOX_TIMEOUT_SECONDS = 10
_FALLBACK_SENTINEL = "I do not have enough context"

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
                self._log(f"    ✗ No ```python block in reply")
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
        triggers its *"I do not have enough context"* fallback instead of
        hallucinating an answer.

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

            if _FALLBACK_SENTINEL.lower() in reply_str.lower():
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

        # — Summary —
        self._log("\n" + "╔" + "═" * 58 + "╗")
        self._log("║" + "  RESULTS SUMMARY".center(58) + "║")
        self._log("╠" + "═" * 58 + "╣")
        for key in (
            "pass_at_1",
            "self_correction_rate",
            "active_teaching_trigger_rate",
            "knowledge_recall_rate",
            "average_prompt_entropy",
        ):
            val = results[key]
            if isinstance(val, float) and key != "average_prompt_entropy":
                formatted = f"{val:.2%}"
            elif isinstance(val, float):
                formatted = f"{val:.4f} bits/token"
            else:
                formatted = str(val)
            self._log(f"║  {key:<40s} {formatted:>14s}  ║")
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
