"""Reward composition for the native engine."""

from __future__ import annotations

from dataclasses import dataclass

from nano_verl.native.data import PromptRecord
from nano_verl.real_reward import math_accuracy_reward, think_tag_reward


@dataclass(slots=True)
class RewardBreakdown:
    """Reward outputs for one prompt group."""

    totals: list[float]
    accuracy: list[float]
    think: list[float]


def compute_group_rewards(
    record: PromptRecord,
    completions: list[str],
    *,
    enable_think_reward: bool,
) -> RewardBreakdown:
    """Compute scalar rewards for one prompt's sampled completions."""

    accuracy_scores = [
        score or 0.0
        for score in math_accuracy_reward(
            completions=completions,
            solution=[record.solution] * len(completions),
            task=[record.task] * len(completions),
        )
    ]
    think_scores = think_tag_reward(completions) if enable_think_reward else [0.0] * len(completions)
    totals = [acc + 0.1 * think for acc, think in zip(accuracy_scores, think_scores)]
    return RewardBreakdown(totals=totals, accuracy=accuracy_scores, think=think_scores)

