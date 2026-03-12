"""Core math for the native GRPO-like update."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StepMetrics:
    """Metrics reported by one training step."""

    loss: float
    mean_raw_reward: float
    mean_shaped_reward: float
    reward_std: float
    mean_kl: float
    mean_ratio: float
    clip_fraction: float
    mean_advantage: float
    accuracy: float


def compute_group_advantages(group_rewards: list[list[float]]) -> list[list[float]]:
    """Normalize rewards inside each prompt group.

    This is the core educational idea behind GRPO: compare completions relative to
    other completions sampled from the same prompt.
    """

    grouped_advantages: list[list[float]] = []
    for rewards in group_rewards:
        if not rewards:
            grouped_advantages.append([])
            continue
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
        std_reward = variance**0.5
        if std_reward < 1e-6:
            grouped_advantages.append([0.0 for _ in rewards])
            continue
        grouped_advantages.append([(reward - mean_reward) / std_reward for reward in rewards])
    return grouped_advantages


def flatten_groups(groups: list[list[float]]) -> list[float]:
    """Flatten prompt groups into one list."""

    return [item for group in groups for item in group]


def apply_kl_penalty(
    grouped_rewards: list[list[float]],
    grouped_kls: list[list[float]],
    *,
    kl_coef: float,
) -> list[list[float]]:
    """Apply reward shaping with a reference-model KL penalty."""

    shaped_groups: list[list[float]] = []
    for rewards, kls in zip(grouped_rewards, grouped_kls):
        shaped_groups.append([reward - kl_coef * kl for reward, kl in zip(rewards, kls)])
    return shaped_groups


def clipped_surrogate_objective(ratio: float, advantage: float, *, clip_range: float) -> float:
    """Compute the PPO-style clipped surrogate for one sample."""

    clipped_ratio = min(max(ratio, 1.0 - clip_range), 1.0 + clip_range)
    return min(ratio * advantage, clipped_ratio * advantage)
