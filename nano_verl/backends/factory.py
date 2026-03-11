"""Factories for backend selection."""

from __future__ import annotations

from nano_verl.backends.base import RolloutBackend, TrainingBackend
from nano_verl.backends.megatron import MegatronTrainingBackendStub
from nano_verl.backends.mock import MockRolloutBackend
from nano_verl.backends.vllm import VLLMServerConfig, VLLMServerRolloutBackend


def create_rollout_backend(
    backend_name: str,
    *,
    seed: int = 7,
    sleep_scale: float = 0.0,
    vllm_server_base_url: str | None = None,
    vllm_api_key: str = "EMPTY",
) -> RolloutBackend:
    """Construct a rollout backend from CLI/runtime config."""

    if backend_name == "mock":
        return MockRolloutBackend(seed=seed, sleep_scale=sleep_scale)
    if backend_name == "vllm-server":
        if not vllm_server_base_url:
            raise ValueError("rollout backend 'vllm-server' requires --vllm-server-base-url")
        return VLLMServerRolloutBackend(
            VLLMServerConfig(base_url=vllm_server_base_url, api_key=vllm_api_key)
        )
    if backend_name == "megatron":
        raise NotImplementedError(
            "Megatron is not a rollout backend in nano_verl. Use vLLM or mock for generation."
        )
    raise ValueError(f"Unsupported rollout backend: {backend_name}")


def create_training_backend(
    backend_name: str,
    *,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
) -> TrainingBackend:
    """Construct a training backend or stub from CLI/runtime config."""

    if backend_name == "megatron":
        return MegatronTrainingBackendStub(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )
    raise ValueError(f"Unsupported training backend: {backend_name}")

