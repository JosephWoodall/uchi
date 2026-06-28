"""GRPO — Group Relative Policy Optimization.

Replaces REINFORCE + EWA-baseline in rl_trainer.py.

Key difference from REINFORCE:
  REINFORCE: advantage = reward - EWA_baseline  (cross-episode, stale)
  GRPO:      advantage = (reward - group_mean) / group_std  (within-episode, fresh)

Benefits:
  - No stale baselines causing exploded negative RL losses
  - Variance reduction through within-group normalization
  - No per-task state to maintain
  - Used by DeepSeek-R1, proven effective for diverse task RL

Reference: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL (2025)
"""

import ast
import math
import os
import re
import subprocess
import sys
import tempfile
from typing import List, Optional

import torch

# Detects raw trie vocabulary that should never reach the user
_SYNSET_TOKEN  = re.compile(r"^[\w'-]+\.(n|v|a|r|s)\.\d{2}$")
_CONTROL_TOKEN = re.compile(r"^<\|.+\|>$|^\[(?:Uncertain|uncertain|UNCERTAIN)\]$")


# ── format reward ─────────────────────────────────────────────────────────────

def format_reward(
    response_tokens: List[str],
    query_tokens: List[str],
    intent_key: Optional[str] = None,
) -> float:
    """Score a response on structural quality, independent of factual content.

    Components and weights:
      0.30  length_score   — penalises too-short or too-long responses
      0.30  coherence_score — penalises trigram repetition (degenerate loops)
      0.25  overlap_score  — penalises high query-echo (parrot responses)
      0.15  structure_score — rewards intent-appropriate vocabulary

    Returns a float in [-1.0, +1.0]. Positive = well-formatted.
    """
    if not response_tokens:
        return -1.0

    # MCQ: only the first token matters — A/B/C/D or not.
    if intent_key == "mcq":
        first = response_tokens[0].lower().strip(".,):") if response_tokens else ""
        return 1.0 if first in {"a", "b", "c", "d"} else -1.0

    n = len(response_tokens)

    # ── length score ──────────────────────────────────────────────────────────
    # Optimal range: 3–50 tokens. Penalise <3 (truncated) and >80 (padded).
    if n < 3:
        length_score = -1.0
    elif n <= 50:
        length_score = 1.0
    else:
        # Sigmoid decay past 50: reaches −0.5 at 100 tokens.
        length_score = 1.0 - 2.0 / (1.0 + math.exp(-0.05 * (n - 50)))

    # ── coherence score (trigram repetition) ─────────────────────────────────
    if n < 3:
        coherence_score = 0.0
    else:
        trigrams = [tuple(response_tokens[i:i+3]) for i in range(n - 2)]
        unique_tg = len(set(trigrams))
        repeat_rate = 1.0 - (unique_tg / len(trigrams))
        # >40% repeated trigrams = degenerate; 0% = perfect
        coherence_score = 1.0 - 2.5 * min(repeat_rate, 0.4) / 0.4

    # ── overlap score (anti-echo) ─────────────────────────────────────────────
    q_set = set(t.lower() for t in query_tokens if len(t) > 2)
    r_set = set(t.lower() for t in response_tokens if len(t) > 2)
    if not r_set:
        overlap_score = 0.0
    else:
        overlap_frac = len(q_set & r_set) / len(r_set)
        # 0% overlap = good; >60% = near-echo = bad
        overlap_score = 1.0 - 2.0 * min(overlap_frac / 0.6, 1.0)

    # ── structure score (intent-aligned vocabulary) ───────────────────────────
    r_text = " ".join(response_tokens).lower()
    if intent_key == "code":
        _CODE_KW = {"def", "return", "import", "class", "for", "while", "if",
                    "print", "true", "false", "none", "self", "else", "elif"}
        kw_hits = sum(1 for kw in _CODE_KW if kw in r_text)
        kw_score = min(kw_hits / 3.0, 1.0)
        try:
            ast.parse(" ".join(response_tokens))
            syntax_bonus = 0.5
        except SyntaxError:
            syntax_bonus = 0.0
        structure_score = min(kw_score + syntax_bonus, 1.0)
    elif intent_key == "math":
        has_digit = bool(re.search(r"\d", r_text))
        has_op    = any(op in r_text for op in ["+", "-", "*", "/", "=", "^"])
        structure_score = 0.5 * float(has_digit) + 0.5 * float(has_op)
    else:
        # Type-Token Ratio: vocabulary richness (standard NLP metric)
        unique_t = len(set(t.lower() for t in response_tokens))
        ttr = unique_t / max(n, 1)
        # Sentence completeness: terminal punctuation in last 3 tokens
        has_terminal = any(
            response_tokens[i].endswith((".", "!", "?"))
            for i in range(max(0, n - 3), n)
        )
        # Content density: alphabetic tokens longer than 3 chars (excludes function words/artifacts)
        content = sum(1 for t in response_tokens if t.isalpha() and len(t) > 3)
        content_density = content / max(n, 1)
        structure_score = 0.40 * ttr + 0.30 * float(has_terminal) + 0.30 * content_density

    # ── naturalness penalty (synset / control token leakage) ─────────────────
    # Counts raw trie vocabulary tokens that should never reach the user.
    # Each leaked token is evidence the policy is not learning clean output.
    # Full penalty (-1.0 additive) kicks in at ≥25% leakage rate.
    leaked = sum(
        1 for t in response_tokens
        if _SYNSET_TOKEN.match(t) or _CONTROL_TOKEN.match(t)
    )
    naturalness_penalty = -1.0 * min(leaked / max(n * 0.25, 1), 1.0) if leaked else 0.0

    total = (
        0.30 * length_score
        + 0.30 * coherence_score
        + 0.25 * overlap_score
        + 0.15 * structure_score
        + naturalness_penalty
    )
    return float(max(-1.0, min(1.0, total)))


_FENCED_CODE_RE = re.compile(r'```(?:python)?\s*(.*?)```', re.DOTALL)


def _extract_code_only(text: str) -> str:
    """Extract just the code portion from a mixed natural-language + code response.

    Priority: fenced code block > def/class lines > raw text.
    """
    m = _FENCED_CODE_RE.search(text)
    if m:
        return m.group(1).strip()
    lines = text.split("\n")
    code_lines: list[str] = []
    in_block = False
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith(("def ", "class ", "import ", "from ")):
            in_block = True
        if in_block:
            code_lines.append(ln)
    return "\n".join(code_lines) if code_lines else text


def _run_in_sandbox(script: str, timeout: float) -> float:
    fname = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False, dir=tempfile.gettempdir()
        ) as f:
            f.write(script)
            fname = f.name
        result = subprocess.run([sys.executable, fname], timeout=timeout, capture_output=True)
        return 1.0 if result.returncode == 0 else -0.5
    except subprocess.TimeoutExpired:
        return -0.25
    except Exception:
        return 0.0
    finally:
        if fname:
            try:
                os.unlink(fname)
            except OSError:
                pass


def execution_reward(code: str, assertion: str, timeout: float = 5.0) -> float:
    """Execute generated code + assertion in a subprocess sandbox.

    Handles mixed natural-language + code responses by trying to extract
    just the code block if the raw text fails execution.

    Returns:
      +1.0  all assertions passed
      -0.5  assertion failed (wrong output, NameError, etc.)
      -0.25 timeout (likely infinite loop)
       0.0  empty input or unexpected error
    """
    if not code.strip() or not assertion.strip():
        return 0.0
    result = _run_in_sandbox(f"{code}\n{assertion}\n", timeout)
    if result == 1.0:
        return result
    extracted = _extract_code_only(code)
    if extracted.strip() and extracted != code:
        result2 = _run_in_sandbox(f"{extracted}\n{assertion}\n", timeout)
        if result2 > result:
            return result2
    return result


def dpo_preference_signal(
    preferred_reward: float,
    rejected_reward: float,
    beta: float = 0.1,
) -> tuple:
    """DPO-inspired preference signal for SSM training.

    Computes a contrastive training signal: how strongly to push toward
    the preferred response and away from the rejected one. Beta controls
    sharpness — lower beta (0.1) gives smoother gradients than standard DPO (0.5).

    Returns (push_signal, pull_signal) where:
      push_signal > 0  — use as reward for the preferred sequence
      pull_signal < 0  — use as reward for the rejected sequence

    Reference: Rafailov et al. 2023, "Direct Preference Optimization" (NeurIPS).
    """
    margin = preferred_reward - rejected_reward
    dpo_val = float(torch.sigmoid(torch.tensor(beta * margin)).item())
    return +dpo_val, -(1.0 - dpo_val)


def grpo_advantage(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute group-relative advantages for a batch of rewards.

    Args:
        rewards: (n_branches,) float tensor of rewards for this episode's group
        eps:     numerical stability floor for std

    Returns:
        advantage: (n_branches,) normalized advantage tensor
    """
    mean = rewards.mean()
    std  = rewards.std() + eps
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


def grpo_agentic_advantage(reward: float, running_mean: float,
                            running_std: float, eps: float = 1e-8) -> float:
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
        self.alpha = alpha   # slow decay — agentic tasks appear infrequently
        self.mean  = 0.0
        self.var   = 1.0

    @property
    def std(self) -> float:
        return max(self.var ** 0.5, 1e-8)

    def update(self, reward: float):
        self.mean = (1 - self.alpha) * self.mean + self.alpha * reward
        self.var  = (1 - self.alpha) * self.var  + self.alpha * (reward - self.mean) ** 2

    def advantage(self, reward: float) -> float:
        return (reward - self.mean) / self.std
