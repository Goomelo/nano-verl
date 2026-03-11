"""Reward stack composition for GRPO."""

from __future__ import annotations

from typing import Callable

from nano_verl.real_reward import math_accuracy_reward, think_tag_reward
from nano_verl.trainer.config import GRPOExperimentConfig


def build_reward_stack(config: GRPOExperimentConfig) -> list[Callable]:
    """Compose reward functions in the order used during training."""

    reward_funcs: list[Callable] = [math_accuracy_reward]
    if config.enable_think_reward:
        reward_funcs.append(think_tag_reward)
    return reward_funcs

