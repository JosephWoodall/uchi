"""agentic_repair.py — 0.5.0 Item 7's agentic repair loop: a ReAct
(Thought -> Action -> Observation) agent that iterates against a real
sandboxed repo, with a tri-state outcome policy and a hard retry cap.

Ports the *pattern* of two prior-art files, fused into one module because
they solve the same problem for this repo's actual shape:
  - `efficient_llm_training/src/react_agent.py`'s ``ReActAgent`` — the
    Thought/Action/Observation regex parsing and loop structure (kept
    close to verbatim; it doesn't assume anything model-specific).
  - `efficient_llm_training/src/agentic_harness.py`'s ``AgenticHarness`` —
    the idea of a bounded tool-call loop over live model generation, but
    NOT its raw token-by-token decode loop or `<|action|>`/`<|observation|>`
    special tokens: FLUX was never trained with those tokens (confirmed:
    `uchi/flux/tokenizer_v2.py`'s special-token set has `<|think|>`/
    `<|assistant|>`/etc., no `<|action|>`), so this uses plain-text
    ``Thought:``/``Action: tool[arg]``/``Final Answer:`` markers instead —
    exactly what `react_agent.py`'s regexes already handle, and it costs
    nothing extra to train for.

Driven by ``uchi/flux/inference_engine.py``'s ``build_generate_fn()`` —
FLUX's actual production seam (``generate_fn(prompt, max_tokens, think) ->
str``) — not the foreign source's streaming multi-event
``InferenceEngine.respond_stream``, which doesn't exist in this codebase.

**Tri-state outcome policy** (todo.md): every repair attempt resolves to
exactly one of ``Outcome.PASS`` / ``Outcome.FAIL`` / ``Outcome.ABSTAIN``.
``ABSTAIN`` (todo.md's "timeout-or-flaky") means the sandbox itself
couldn't produce a trustworthy verdict — a timeout or an internal sandbox
exception, not "the patch is wrong." Per the plan: this does **not** retry
and does **not** fall back to any other oracle cascade layer (word-overlap/
entailment/numeric/relational) — none of them has an opinion about code
execution, so passing an ABSTAIN off to them would manufacture a false
signal, not recover one. Only ``FAIL`` retries, up to ``max_attempts``.

**Retry cap**: ``max_attempts`` defaults to 4 (todo.md's "3-5 attempts,
hard number") -- a whole-episode retry (fresh checkout, fresh ReAct
exploration, informed by the previous attempt's real failure text spliced
in as feedback), distinct from ``max_iterations`` (the ReAct loop's own
per-episode Thought/Action/Observation step cap, same role as
`react_agent.py`'s `max_iterations` / `agentic_harness.py`'s `max_steps`).

Exploration (``read_file``/``grep_repo``/``apply_patch``/``run_tests`` via
`uchi/tools.py`'s ``RepoToolRegistry``) happens against one working
checkout per attempt, which the model is free to mutate turn by turn — the
final verdict, though, always re-grades a *fresh* ``ExecutionSandbox.evaluate()``
call (its own clean checkout) against the accumulated diff, so a messy
exploration checkout (failed intermediate edits, partial reverts) can never
leak into the graded result.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from .execution_sandbox import ExecutionSandbox, SandboxEvalResult
from .tools import RepoToolRegistry

DEFAULT_MAX_ITERATIONS = 5
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_MAX_TOKENS = 300
_OBSERVATION_CHAR_CAP = 1500

# Same shape as react_agent.py's patterns -- plain-text markers, no special
# tokens FLUX was never trained on (see module docstring). One deliberate
# departure from the source: react_agent.py's single `tool[arg]` pattern
# breaks the moment `arg` contains a `]` -- fine for its calculator/
# physics_lookup args, but apply_patch's arg is a full unified diff, which
# routinely contains `]` in ordinary code (list literals, regex character
# classes, slice syntax). apply_patch therefore gets its own delimited
# block syntax instead of squeezing a multi-line diff into `[...]`; the
# other three tools (short, single-line args) keep the original bracket
# form. Both are spelled out in `_build_system_prompt` so the model is
# told the exact format, not left to guess it.
_THOUGHT_PATTERN = re.compile(
    r"Thought:\s*(.*?)(?=Action:|Final Answer:|\Z)", re.DOTALL,
)
_ACTION_PATTERN = re.compile(
    r"Action:\s*(read_file|grep_repo|run_tests)\[([^\n]*?)\]",
)
_PATCH_ACTION_PATTERN = re.compile(
    r"Action:\s*apply_patch\s*\n<<<PATCH>>>\n(.*?)\n<<<END_PATCH>>>", re.DOTALL,
)
_FINAL_PATTERN = re.compile(
    r"Final Answer:\s*(.*)", re.DOTALL,
)


class Outcome(Enum):
    PASS = "pass"
    FAIL = "fail"
    ABSTAIN = "abstain"  # todo.md's "timeout-or-flaky"


@dataclass
class RepairOutcome:
    outcome: Outcome
    patch: str
    attempts: int
    eval_result: SandboxEvalResult | None
    transcript: list[str] = field(default_factory=list)


def classify_outcome(result: SandboxEvalResult) -> Outcome:
    """The tri-state policy, applied to one `ExecutionSandbox.evaluate()`
    result. Timeout or an internal sandbox error is ABSTAIN regardless of
    what (if anything) the partial test results say -- an incomplete or
    untrustworthy run is not evidence either way.
    """
    if result.timed_out or result.error:
        return Outcome.ABSTAIN
    if result.resolved:
        return Outcome.PASS
    return Outcome.FAIL


def _diff_against_head(repo_dir: Path) -> str:
    """Unified diff of *repo_dir*'s working tree against its checked-out
    HEAD (== base_commit, since `ExecutionSandbox.checkout_repo` already
    reset there), including untracked new files. `git diff` alone misses
    untracked files regardless of working-tree state; staging first with
    `git add -A` is the standard way to fold them into one diff without
    needing a second untracked-files pass.
    """
    subprocess.run(["git", "add", "-A"], cwd=repo_dir, capture_output=True, text=True, timeout=30.0)
    result = subprocess.run(
        ["git", "diff", "--cached"], cwd=repo_dir, capture_output=True, text=True, timeout=30.0,
    )
    return result.stdout


def _format_failure_feedback(result: SandboxEvalResult) -> str:
    if not result.patch_applied:
        return f"Your previous patch failed to apply:\n{result.error}"
    lines = ["Your previous patch applied but did not resolve the issue:"]
    for test_id, passed in result.fail_to_pass.items():
        if not passed:
            lines.append(f"  still failing: {test_id}")
    for test_id, passed in result.pass_to_pass.items():
        if not passed:
            lines.append(f"  regressed (was passing): {test_id}")
    return "\n".join(lines)[:_OBSERVATION_CHAR_CAP]


class AgenticRepairAgent:
    """ReAct loop over `RepoToolRegistry`'s four tools, wrapped in a
    whole-episode retry loop with the tri-state outcome policy.
    """

    def __init__(
        self,
        generate_fn: Callable[..., str],
        sandbox: ExecutionSandbox,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.generate_fn = generate_fn
        self.sandbox = sandbox
        self.max_iterations = max_iterations
        self.max_attempts = max_attempts
        self.max_tokens = max_tokens

    def repair(
        self,
        repo_path: str,
        problem_statement: str,
        fail_to_pass: list[str],
        pass_to_pass: list[str],
        base_commit: str | None = None,
    ) -> RepairOutcome:
        feedback = ""
        patch_text = ""
        eval_result: SandboxEvalResult | None = None
        transcript: list[str] = []

        for attempt in range(1, self.max_attempts + 1):
            repo_dir = self.sandbox.checkout_repo(repo_path, base_commit=base_commit)
            tools = RepoToolRegistry(self.sandbox, repo_dir)
            transcript = self._run_react_loop(problem_statement, tools, feedback)
            patch_text = _diff_against_head(repo_dir)

            if not patch_text.strip():
                feedback = (
                    "No changes were made to the repository. You must call "
                    "apply_patch[<unified diff>] with a real diff before finishing."
                )
                continue

            eval_result = self.sandbox.evaluate(
                repo_path=repo_path, patch_text=patch_text,
                fail_to_pass=fail_to_pass, pass_to_pass=pass_to_pass,
                base_commit=base_commit,
            )
            outcome = classify_outcome(eval_result)
            if outcome is Outcome.PASS:
                return RepairOutcome(Outcome.PASS, patch_text, attempt, eval_result, transcript)
            if outcome is Outcome.ABSTAIN:
                # No retry, no fallback to any other oracle layer -- see
                # module docstring's tri-state policy section.
                return RepairOutcome(Outcome.ABSTAIN, patch_text, attempt, eval_result, transcript)
            feedback = _format_failure_feedback(eval_result)

        return RepairOutcome(Outcome.FAIL, patch_text, self.max_attempts, eval_result, transcript)

    # ── ReAct loop ────────────────────────────────────────────────────────

    def _build_system_prompt(self, tools: RepoToolRegistry) -> str:
        return (
            "You are a software engineer fixing a real bug in a real repository.\n"
            "Follow this format:\n\n"
            "Thought: reason about the current situation and what to do next\n"
            "Action: tool_name[input]\n"
            "Observation: (result from the tool will appear here)\n"
            "... (repeat Thought/Action/Observation as needed)\n"
            "Thought: I have applied a patch that should resolve the issue\n"
            "Final Answer: brief summary of the fix\n\n"
            "apply_patch's input is a full unified diff, which is not safe to "
            "put inside [...] (diffs commonly contain ']'). Use this form for "
            "apply_patch only:\n"
            "Action: apply_patch\n"
            "<<<PATCH>>>\n"
            "<the unified diff>\n"
            "<<<END_PATCH>>>\n\n"
            f"Available tools:\n{tools.descriptions()}\n\n"
            "You must call apply_patch with a real unified diff before giving "
            "your Final Answer -- reasoning alone does not change the repo."
        )

    def _run_react_loop(
        self, problem_statement: str, tools: RepoToolRegistry, feedback: str,
    ) -> list[str]:
        system_prompt = self._build_system_prompt(tools)
        memory: list[str] = []

        for _ in range(self.max_iterations):
            prompt = f"{system_prompt}\n\nIssue:\n{problem_statement}\n\n"
            if feedback:
                prompt += f"Feedback from your previous attempt:\n{feedback}\n\n"
            if memory:
                prompt += "\n".join(memory) + "\n"

            response = self.generate_fn(prompt, max_tokens=self.max_tokens, think=True)

            final_match = _FINAL_PATTERN.search(response)
            thought_match = _THOUGHT_PATTERN.search(response)
            patch_match = _PATCH_ACTION_PATTERN.search(response)
            action_match = patch_match or _ACTION_PATTERN.search(response)

            if thought_match:
                memory.append(f"Thought: {thought_match.group(1).strip()}")

            if final_match and (not action_match or final_match.start() < action_match.start()):
                memory.append(f"Final Answer: {final_match.group(1).strip()}")
                break

            if patch_match:
                tool_name, tool_arg = "apply_patch", patch_match.group(1)
                memory.append(f"Action: apply_patch[<{len(tool_arg)}-char diff>]")
                result = tools.execute(tool_name, tool_arg)
                memory.append(f"Observation: {result.output[:_OBSERVATION_CHAR_CAP]}")
            elif action_match:
                tool_name, tool_arg = action_match.group(1).strip(), action_match.group(2).strip()
                memory.append(f"Action: {tool_name}[{tool_arg}]")
                result = tools.execute(tool_name, tool_arg)
                memory.append(f"Observation: {result.output[:_OBSERVATION_CHAR_CAP]}")
            else:
                # No action, no final answer -- nothing left to drive the loop.
                break

        return memory
