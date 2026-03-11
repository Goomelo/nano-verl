"""Reward functions for real GRPO training."""

from __future__ import annotations

import re
from typing import Any


def math_accuracy_reward(
    completions,
    solution,
    task=None,
    **_: Any,
) -> list[float | None]:
    """Reward exact math answers with 1.0 and wrong answers with 0.0.

    This follows TRL's custom reward function contract and returns one reward per completion.
    Non-math tasks can be skipped by returning ``None``.
    """

    rewards: list[float | None] = []
    tasks = task if task is not None else ["math"] * len(completions)

    for completion, expected, task_name in zip(completions, solution, tasks):
        if task_name != "math":
            rewards.append(None)
            continue

        predicted = _extract_last_number(_completion_to_text(completion))
        gold = _normalize_number(expected)
        rewards.append(1.0 if predicted == gold else 0.0)

    return rewards


def think_tag_reward(completions, **_: Any) -> list[float]:
    """Encourage Qwen-style reasoning traces wrapped in <think> tags."""

    rewards: list[float] = []
    for completion in completions:
        text = _completion_to_text(completion)
        has_open = "<think>" in text
        has_close = "</think>" in text
        rewards.append(1.0 if has_open and has_close else 0.0)
    return rewards


def _completion_to_text(completion: Any) -> str:
    """Flatten TRL completion payloads into a plain string."""

    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def _extract_last_number(text: str) -> str | None:
    """Extract the final numeric token from a free-form completion."""

    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    return _normalize_number(matches[-1])


def _normalize_number(value: str) -> str | None:
    """Normalize 12, 12.0, and 12.000 to the same string representation."""

    try:
        parsed = float(value)
    except ValueError:
        return None

    if parsed.is_integer():
        return str(int(parsed))
    return f"{parsed:.6g}"

