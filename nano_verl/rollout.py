"""Rollout engines built on top of pluggable rollout backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from nano_verl.backends.base import RolloutBackend, RolloutRequest
from nano_verl.backends.mock import MockRolloutBackend
from nano_verl.backends.vllm import VLLMServerConfig, VLLMServerRolloutBackend
from nano_verl.types import PromptExample, RolloutCandidate, RolloutSample


class RolloutEngine(ABC):
    """Abstract rollout interface used by the main pipeline."""

    @abstractmethod
    async def generate(self, prompt: PromptExample, num_samples: int) -> RolloutSample:
        """Generate multiple candidates for a single prompt."""


class BackendRolloutEngine(RolloutEngine):
    """Thin adapter that turns a rollout backend into the engine interface."""

    def __init__(
        self,
        backend: RolloutBackend,
        *,
        model_name: str | None = None,
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        timeout_s: float = 60.0,
        seed: int | None = None,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout_s = timeout_s
        self.seed = seed

    async def generate(self, prompt: PromptExample, num_samples: int) -> RolloutSample:
        return await self.backend.generate(
            RolloutRequest(
                prompt=prompt,
                num_samples=num_samples,
                model_name=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                timeout_s=self.timeout_s,
                seed=self.seed,
            )
        )


class MockRolloutEngine(BackendRolloutEngine):
    """Engine wrapper around the local mock rollout backend."""

    def __init__(self, seed: int = 7, sleep_scale: float = 0.0) -> None:
        super().__init__(MockRolloutBackend(seed=seed, sleep_scale=sleep_scale), seed=seed)


class VLLMServerRolloutEngine(BackendRolloutEngine):
    """Engine wrapper around the vLLM OpenAI-compatible server backend."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str = "EMPTY",
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        timeout_s: float = 60.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            VLLMServerRolloutBackend(VLLMServerConfig(base_url=base_url, api_key=api_key)),
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout_s=timeout_s,
            seed=seed,
        )
