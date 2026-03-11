"""Training backend resolution.

The goal is to keep backend-specific decisions away from the CLI entrypoint.
"""

from __future__ import annotations

from dataclasses import dataclass

from nano_verl.backends import create_training_backend
from nano_verl.trainer.actor import build_model_init_kwargs
from nano_verl.trainer.config import GRPOExperimentConfig


@dataclass(slots=True)
class GenerationBackendConfig:
    """Backend-specific knobs that feed into `trl.GRPOConfig`."""

    config_kwargs: dict[str, object]
    summary: str


def resolve_generation_backend(config: GRPOExperimentConfig) -> GenerationBackendConfig:
    """Translate a high-level backend choice into TRL config kwargs."""

    if config.generation_backend == "megatron":
        backend = create_training_backend(
            "megatron",
            tensor_parallel_size=config.megatron_tensor_parallel_size,
            pipeline_parallel_size=config.megatron_pipeline_parallel_size,
        )
        plan = backend.plan(config.model_name, config.output_dir)
        raise NotImplementedError(
            f"{backend.backend_name} backend is not implemented in nano_verl.\n"
            f"summary: {plan.summary}\n"
            f"launch_example: {plan.launch_example}\n"
            f"notes: {' | '.join(plan.notes)}"
        )

    use_vllm = config.generation_backend in {"vllm-server", "vllm-colocate"}
    summary = "Transformers generation inside the trainer process."
    backend_kwargs: dict[str, object] = {"use_vllm": False}

    if use_vllm:
        summary = "vLLM handles rollout generation while TRL keeps the policy update loop."
        backend_kwargs = {
            "use_vllm": True,
            "vllm_mode": "server" if config.generation_backend == "vllm-server" else "colocate",
            "vllm_model_impl": config.vllm_model_impl,
            "vllm_server_base_url": config.vllm_server_base_url
            or f"http://{config.vllm_server_host}:{config.vllm_server_port}",
            "vllm_gpu_memory_utilization": config.vllm_gpu_memory_utilization,
            "vllm_tensor_parallel_size": config.vllm_tensor_parallel_size,
            "vllm_max_model_length": config.vllm_max_model_length,
        }

    backend_kwargs["model_init_kwargs"] = build_model_init_kwargs(config)
    return GenerationBackendConfig(config_kwargs=backend_kwargs, summary=summary)

