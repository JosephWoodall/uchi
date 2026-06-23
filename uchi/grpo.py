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

import torch


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
