"""grpo.py — 0.5.0 Item 6: execution-verified code self-play via GRPO
(Group Relative Policy Optimization).

Ports `efficient_llm_training/src/grpo.py`'s pure math (`grpo_advantage`,
`grpo_loss`, `grpo_agentic_advantage`, `AgenticBaseline`) near-verbatim —
none of it is model-specific, nothing to adapt. Reference: DeepSeek-R1:
Incentivizing Reasoning Capability in LLMs via RL (2025); advantage is
normalized within a group of same-episode samples rather than against a
stale cross-episode baseline (REINFORCE's failure mode the source's own
docstring names).

What's new here, not in the source, is everything that makes GRPO apply to
*this* repo's actual self-play shape:
  - The reward is `execution_sandbox.SandboxEvalResult.reward` (0.5.0
    Item 5) — real FAIL_TO_PASS/PASS_TO_PASS partial credit, not a
    synthetic single-assertion reward.
  - Branches are full ReAct episodes via `agentic_repair.run_react_episode`
    (0.5.0 Item 7), not single-shot completions — Item 8's action space is
    defined directly on top of the Thought-tagged trajectory these
    episodes produce (see `parse_transcript` below), so a branch has to be
    a real multi-step trajectory, not one string.
  - Curriculum bucketing by patch size (`todo.md`: "required to avoid
    degenerate zero-variance advantage on sparse reward" — an easy patch
    is far more likely to produce a non-all-zero reward group than a hard
    one, which is what GRPO's normalization actually needs to have a
    gradient at all).

Real training here needs a proposer coherent enough to occasionally solve
an instance, or every branch in a group gets reward 0, `grpo_advantage`
degenerates to `nan`/`0`, and there is no gradient. That is the Phase 1->4
decision gate this module deliberately does not try to get around — see
`tasks/todo.md`'s Item 6 section and the 0.5.0 project memory. This module
is buildable and testable now (against a scripted fake `generate_fn`, same
pattern as `tests/test_agentic_repair.py`); a real curriculum training run
against the live model waits for that gate to clear.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Literal

import torch
import torch.nn.functional as F

from .agentic_repair import run_react_episode
from .code_retrieval import CodeIndex, extract_patch_files
from .execution_sandbox import ExecutionSandbox, SandboxEvalResult
from .repo_fetch import RepoFetchError, ensure_local_clone
from .tools import RepoToolRegistry

# ── Pure math (ported near-verbatim from efficient_llm_training/src/grpo.py) ──


def grpo_advantage(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute group-relative advantages for a batch of rewards.

    Args:
        rewards: (n_branches,) float tensor of rewards for this episode's group
        eps:     numerical stability floor for std

    Returns:
        advantage: (n_branches,) normalized advantage tensor
    """
    mean = rewards.mean()
    std = rewards.std() + eps
    return (rewards - mean) / std


def grpo_loss(advantage: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
    """GRPO policy gradient loss.

    L = -mean(advantage_i * log_prob_i)

    Minimizing this loss increases log_prob for above-average responses
    and decreases it for below-average responses within the group.

    Args:
        advantage: (n_branches,) group-relative advantages
        log_probs: (n_branches,) per-branch mean log probabilities

    Returns:
        scalar loss tensor
    """
    return -(advantage * log_probs).mean()


def grpo_agentic_advantage(
    reward: float, running_mean: float, running_std: float, eps: float = 1e-8,
) -> float:
    """Single-sample advantage for agentic tasks (n_branches=1).

    Falls back to z-score against a running mean/std across recent
    agentic episodes since group normalization requires n > 1.

    Args:
        reward:       current episode reward
        running_mean: EWA mean of recent agentic rewards
        running_std:  EWA std of recent agentic rewards

    Returns:
        scalar advantage
    """
    return (reward - running_mean) / max(running_std, eps)


class AgenticBaseline:
    """Lightweight running statistics for agentic task advantage estimation.

    Maintains EWA mean and variance so single-response agentic episodes
    can still compute a meaningful advantage (unlike constitutional
    episodes which have n_branches=4 for within-group normalization).
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha  # slow decay -- agentic tasks appear infrequently
        self.mean = 0.0
        self.var = 1.0

    @property
    def std(self) -> float:
        return max(self.var ** 0.5, 1e-8)

    def update(self, reward: float):
        self.mean = (1 - self.alpha) * self.mean + self.alpha * reward
        self.var = (1 - self.alpha) * self.var + self.alpha * (reward - self.mean) ** 2

    def advantage(self, reward: float) -> float:
        return (reward - self.mean) / self.std


# ── Trajectory schema (shared by Item 8's MCTS action space) ─────────────

_STEP_PREFIXES: dict[str, str] = {
    "Thought: ": "thought",
    "Action: ": "action",
    "Observation: ": "observation",
    "Final Answer: ": "final",
}


@dataclass
class TrajectoryStep:
    """One Thought/Action/Observation/Final segment of a ReAct episode."""

    kind: Literal["thought", "action", "observation", "final"]
    text: str


def parse_transcript(lines: list[str]) -> list[TrajectoryStep]:
    """Convert `agentic_repair.run_react_episode`'s flat, prefixed
    transcript lines into structured `TrajectoryStep`s.

    Reuses the exact `"Thought: "`/`"Action: "`/`"Observation: "`/
    `"Final Answer: "` prefix convention `agentic_repair.py` already emits
    -- this does not change what that module produces, only gives Item 8
    (and this module's own reward bookkeeping) a structured view of it.
    """
    steps: list[TrajectoryStep] = []
    for line in lines:
        for prefix, kind in _STEP_PREFIXES.items():
            if line.startswith(prefix):
                steps.append(TrajectoryStep(kind=kind, text=line[len(prefix):]))
                break
        else:
            # Every line agentic_repair.py emits carries one of the four
            # prefixes above; this is a defensive fallback, not an
            # expected path.
            steps.append(TrajectoryStep(kind="thought", text=line))
    return steps


@dataclass
class Branch:
    """One GRPO branch: a sampled trajectory, its resulting patch, and its
    real execution-verified reward."""

    transcript: list[str]
    patch: str
    eval_result: SandboxEvalResult | None
    reward: float
    log_prob: torch.Tensor = field(repr=False)

    @property
    def steps(self) -> list[TrajectoryStep]:
        return parse_transcript(self.transcript)


# ── Sequence log-prob (reuses sft_train.py's masking pattern) ────────────


def sequence_log_prob(
    model, tokenizer, prompt: str, response: str,
    device: str = "cpu", max_length: int = 1024,
) -> torch.Tensor:
    """Mean per-token log-prob of *response* under teacher forcing, given
    *prompt* as a prefix.

    Same prompt/response split and masking idea as
    `uchi/flux/sft_train.py`'s `masked_ce_loss` (loss computed only on
    response tokens, prompt masked) -- this returns the differentiable
    log-prob directly rather than a loss, since GRPO's advantage multiplies
    log-probs, it doesn't target them against a label.
    """
    prompt_ids = tokenizer.encode_text(prompt, max_length=max_length) or [0]
    response_ids = tokenizer.encode_text(response, max_length=max_length) or [0]
    full_ids = (prompt_ids + response_ids)[:max_length]
    n_prompt = min(len(prompt_ids), len(full_ids))
    n_response = len(full_ids) - n_prompt
    if n_response <= 0 or len(full_ids) < 2:
        return torch.zeros((), device=device)

    x = torch.tensor([full_ids], dtype=torch.long, device=device)
    lang_logits, _ = model(x[:, :-1])
    targets = x[:, 1:]

    mask = torch.zeros_like(targets, dtype=torch.float32)
    mask[:, -n_response:] = 1.0

    logits_flat = lang_logits.reshape(-1, lang_logits.size(-1))
    targets_flat = targets.reshape(-1)
    mask_flat = mask.reshape(-1)

    nll_per_token = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    masked_nll = (nll_per_token * mask_flat).sum() / mask_flat.sum().clamp(min=1)
    return -masked_nll


# ── Curriculum (bucket by patch size, easy -> hard) ───────────────────────


def _patch_line_count(patch: str) -> int:
    """Lines actually changed (added/removed), not the full diff's line
    count (hunk headers/context lines aren't a size signal)."""
    return sum(
        1 for line in patch.splitlines()
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith(("+++", "---"))
    )


# Thresholds are a first-pass split of "small enough to plausibly one-shot"
# vs "clearly multi-file/multi-concern" -- not derived from a distribution
# fit; revisit once real SWE-Gym patch-size stats are pulled for this
# corpus specifically, same caveat todo.md already flags for Item 4's small
# samples.
SMALL_PATCH_MAX_LINES = 10
MEDIUM_PATCH_MAX_LINES = 50


def load_curriculum(jsonl_path: str) -> list[list[dict]]:
    """Load SWE-Gym-shaped records from *jsonl_path* (the schema
    `.uchi/corpus/swe_gym_full.jsonl` already uses: `repo`, `instance_id`,
    `problem_statement`, `patch`, `base_commit`, `fail_to_pass`,
    `pass_to_pass`) and bucket them by patch size, easy -> hard.

    Returns `[small_bucket, medium_bucket, large_bucket]`, each a list of
    raw records in file order (no reshuffling within a bucket -- the
    trainer decides sampling order).
    """
    small: list[dict] = []
    medium: list[dict] = []
    large: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            n = _patch_line_count(record.get("patch", ""))
            if n <= SMALL_PATCH_MAX_LINES:
                small.append(record)
            elif n <= MEDIUM_PATCH_MAX_LINES:
                medium.append(record)
            else:
                large.append(record)
    return [small, medium, large]


# ── test-impact-analysis fallback (0.5.0 Item 5 Stage 2) ──────────────────


def derive_test_lists(
    sandbox: ExecutionSandbox, repo_path: str, base_commit: str | None, patch: str,
) -> tuple[list[str], list[str]]:
    """For curriculum records with no pre-annotated FAIL_TO_PASS/PASS_TO_PASS
    (`tasks/todo.md`'s Item 5 Stage 2: "test-impact-analysis subsetting for
    the broader training corpus, where no ... annotation exists") --
    derives the same shape from real execution instead of a pre-supplied
    list.

    Builds a `CodeIndex` over a clean checkout at *base_commit* (the same
    checkout the baseline run below uses, not a second one), computes
    `impacted_tests` for the patch's changed files
    (`code_retrieval.extract_patch_files`), then runs exactly those tests
    BEFORE the patch is applied: tests already failing at baseline become
    the FAIL_TO_PASS candidates (the same "must transition from fail to
    pass" idea real SWE-bench annotation encodes, just derived instead of
    supplied); tests already passing become the PASS_TO_PASS candidates
    (no-regression check). Returns `([], [])` if the patch touches no file
    with any discoverable impacted test -- callers get the same
    all-empty-lists behavior `SandboxEvalResult.reward` already defines
    for that case, not new handling here.

    **File-level granularity, not per-test-function**: `impacted_tests`
    resolves whole files (import graph/naming convention operate at file
    grain), so the returned lists are file-shaped test ids
    (`"test_foo.py"`), coarser than real SWE-bench annotation's
    `"test_foo.py::test_specific_case"`. `ExecutionSandbox.run_tests`
    accepts a whole file as a node id exactly the way it accepts a single
    test, so this grades correctly, just at a coarser pass/fail
    resolution than curated, function-annotated instances get -- an
    honest, buildable-now tradeoff, not a bug.
    """
    changed_files = extract_patch_files(patch)
    if not changed_files:
        return [], []

    baseline_dir = sandbox.checkout_repo(repo_path, base_commit=base_commit)
    index = CodeIndex.build(baseline_dir)
    impacted = sorted(index.impacted_tests(changed_files))
    if not impacted:
        return [], []

    baseline = sandbox.run_tests(baseline_dir, impacted)
    fail_to_pass = [t for t in impacted if not baseline.passed(t)]
    pass_to_pass = [t for t in impacted if baseline.passed(t)]
    return fail_to_pass, pass_to_pass


# ── GRPOTrainer ────────────────────────────────────────────────────────────


class GRPOTrainer:
    """Samples `n_branches` independent ReAct episodes per curriculum
    instance, grades each with real test execution, and applies a GRPO
    update to the policy.

    One instance = one "group" in GRPO's sense: `n_branches` branches all
    attempt the same instance, rewards are normalized within that group
    (`grpo_advantage`), and the resulting loss is backpropagated through
    each branch's own `sequence_log_prob`.
    """

    def __init__(
        self,
        model,
        tokenizer,
        sandbox: ExecutionSandbox,
        generate_fn: Callable[..., str],
        optimizer: torch.optim.Optimizer,
        n_branches: int = 4,
        max_iterations: int = 5,
        max_tokens: int = 300,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sandbox = sandbox
        self.generate_fn = generate_fn
        self.optimizer = optimizer
        self.n_branches = n_branches
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.device = device

    def _sample_branch(
        self, repo_path: str, problem_statement: str,
        fail_to_pass: list[str], pass_to_pass: list[str],
        base_commit: str | None,
    ) -> Branch:
        repo_dir = self.sandbox.checkout_repo(repo_path, base_commit=base_commit)
        tools = RepoToolRegistry(self.sandbox, repo_dir)

        # Wrap generate_fn to capture the *raw* per-turn completions.
        # run_react_episode's returned transcript deliberately abbreviates
        # apply_patch's diff to a `<N-char diff>` placeholder (kept short
        # for the next turn's prompt) -- scoring log-prob against that
        # summary would throw away exactly the tokens that differ between
        # a correct and an incorrect patch, since two diffs of equal
        # length produce an identical placeholder. GRPO needs the log-prob
        # of what the policy actually generated, not a display summary of
        # it, so the raw responses (not the parsed memory) are what
        # sequence_log_prob scores.
        raw_responses: list[str] = []

        def _tracking_generate(prompt, **kwargs):
            response = self.generate_fn(prompt, **kwargs)
            raw_responses.append(response)
            return response

        transcript = run_react_episode(
            _tracking_generate, tools, problem_statement,
            feedback="", max_iterations=self.max_iterations, max_tokens=self.max_tokens,
        )

        subprocess.run(["git", "add", "-A"], cwd=repo_dir, capture_output=True, text=True, timeout=30.0)
        patch = subprocess.run(
            ["git", "diff", "--cached"], cwd=repo_dir, capture_output=True, text=True, timeout=30.0,
        ).stdout

        if not patch.strip():
            eval_result = None
            reward = 0.0
        else:
            f2p, p2p = fail_to_pass, pass_to_pass
            if not f2p and not p2p:
                # No pre-annotated test lists (broader-corpus records past
                # curated SWE-Gym) -- derive them from real execution
                # instead of skipping grading entirely. Existing callers
                # that already pass real annotations never hit this path.
                f2p, p2p = derive_test_lists(self.sandbox, repo_path, base_commit, patch)
            eval_result = self.sandbox.evaluate(
                repo_path=repo_path, patch_text=patch,
                fail_to_pass=f2p, pass_to_pass=p2p,
                base_commit=base_commit,
            )
            reward = eval_result.reward

        prompt = f"Issue:\n{problem_statement}\n\n"
        response = "\n".join(raw_responses)
        log_prob = sequence_log_prob(self.model, self.tokenizer, prompt, response, device=self.device)

        return Branch(transcript=transcript, patch=patch, eval_result=eval_result,
                       reward=reward, log_prob=log_prob)

    def train_step(
        self, repo_path: str, problem_statement: str,
        fail_to_pass: list[str], pass_to_pass: list[str],
        base_commit: str | None = None,
    ) -> tuple[list[Branch], torch.Tensor]:
        """Run one GRPO group (`n_branches` episodes against the same
        instance), apply the policy-gradient update, return the branches
        and the scalar loss actually backpropagated."""
        branches = [
            self._sample_branch(repo_path, problem_statement, fail_to_pass, pass_to_pass, base_commit)
            for _ in range(self.n_branches)
        ]

        rewards = torch.tensor([b.reward for b in branches], dtype=torch.float32, device=self.device)
        log_probs = torch.stack([b.log_prob for b in branches])
        advantage = grpo_advantage(rewards)
        loss = grpo_loss(advantage, log_probs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return branches, loss.detach()

    def run_curriculum(self, jsonl_path: str, max_instances: int | None = None) -> list[tuple[list[Branch], torch.Tensor]]:
        """Iterate `load_curriculum`'s buckets easy -> hard, running one
        `train_step` per instance. `max_instances=None` runs everything --
        callers doing a real multi-day run should pass an explicit cap.

        Each record's `repo` field is `"owner/name"`, not a local path --
        resolved to a cached local clone via `repo_fetch.ensure_local_clone`
        (same as `benchmarks/retrieval_scaling_benchmark.py`'s code track),
        not passed to the sandbox directly.
        """
        buckets = load_curriculum(jsonl_path)
        results = []
        n_done = 0
        for bucket in buckets:
            for record in bucket:
                if max_instances is not None and n_done >= max_instances:
                    return results
                try:
                    repo_path = ensure_local_clone(record["repo"], record["base_commit"])
                except RepoFetchError:
                    continue
                result = self.train_step(
                    repo_path=repo_path,
                    problem_statement=record["problem_statement"],
                    fail_to_pass=record["fail_to_pass"],
                    pass_to_pass=record["pass_to_pass"],
                    base_commit=record.get("base_commit"),
                )
                results.append(result)
                n_done += 1
        return results
