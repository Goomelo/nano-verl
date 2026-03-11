"""Reward functions that score rollout candidates."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from nano_verl.types import PromptExample, RewardResult, RolloutCandidate


class RewardFunction(ABC):
    """Abstract interface for scoring a rollout candidate."""

    @abstractmethod
    def score(self, prompt: PromptExample, candidate: RolloutCandidate) -> RewardResult:
        """Return a reward value and a human-readable reason."""


class RuleReward(RewardFunction):
    """Simple rule-based reward for math and lightweight QA tasks."""

    def score(self, prompt: PromptExample, candidate: RolloutCandidate) -> RewardResult:
        if prompt.task_type == "math":
            return self._score_math(prompt, candidate)
        return self._score_keywords(prompt, candidate)

    def _score_math(self, prompt: PromptExample, candidate: RolloutCandidate) -> RewardResult:
        """Give reward 1.0 for correct math answers, otherwise 0.0."""

        expected = _normalize_number(prompt.reference_answer)
        predicted = _extract_last_number(candidate.text)

        if predicted is None:
            return RewardResult(
                prompt_id=prompt.prompt_id,
                sample_id=candidate.sample_id,
                reward=0.0,
                is_correct=False,
                reason="no numeric answer found",
            )

        is_correct = predicted == expected
        return RewardResult(
            prompt_id=prompt.prompt_id,
            sample_id=candidate.sample_id,
            reward=1.0 if is_correct else 0.0,
            is_correct=is_correct,
            reason=f"expected={expected}, predicted={predicted}",
        )

    def _score_keywords(self, prompt: PromptExample, candidate: RolloutCandidate) -> RewardResult:
        """Assign fractional reward based on keyword coverage."""

        keywords = [str(item).lower() for item in prompt.metadata.get("keywords", [])]
        if not keywords:
            return RewardResult(
                prompt_id=prompt.prompt_id,
                sample_id=candidate.sample_id,
                reward=0.0,
                is_correct=False,
                reason="no keywords configured for non-math task",
            )

        text = candidate.text.lower()
        matched = [keyword for keyword in keywords if keyword in text]
        reward = len(matched) / len(keywords)

        return RewardResult(
            prompt_id=prompt.prompt_id,
            sample_id=candidate.sample_id,
            reward=reward,
            is_correct=reward == 1.0,
            reason=f"matched {len(matched)}/{len(keywords)} keywords: {matched}",
        )


def _normalize_number(value: str) -> str:
    """Normalize numeric strings so 12 and 12.0 compare equal."""

    as_float = float(value)
    if as_float.is_integer():
        return str(int(as_float))
    return f"{as_float:.6g}"


def _extract_last_number(text: str) -> str | None:
    """Extract the last number-like token from free-form model output."""

    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    return _normalize_number(matches[-1])

