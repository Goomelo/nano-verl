"""Shared backend interfaces used across nano_verl."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from nano_verl.types import PromptExample, RolloutSample


@dataclass(slots=True)
class RolloutRequest:
    """Normalized rollout request passed to backend implementations."""

    prompt: PromptExample
    num_samples: int
    model_name: str | None = None
    max_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    timeout_s: float = 60.0
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingBackendPlan:
    """A lightweight plan describing how a training backend should be invoked."""

    backend_name: str
    summary: str
    launch_example: str
    notes: list[str] = field(default_factory=list)


class RolloutBackend(ABC):
    """Interface for any candidate-generation backend."""

    backend_name: str

    @abstractmethod
    async def generate(self, request: RolloutRequest) -> RolloutSample:
        """Generate a rollout sample for one prompt."""


class TrainingBackend(ABC):
    """Interface for training backends used by the real RL path."""

    backend_name: str

    @abstractmethod
    def plan(self, model_name: str, output_dir: str) -> TrainingBackendPlan:
        """Return an execution plan for the backend."""

    @abstractmethod
    def validate(self) -> None:
        """Validate the current runtime, or raise a clear error."""

