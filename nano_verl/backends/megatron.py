"""Megatron backend stubs.

This module intentionally does not pretend Megatron is already wired into the local
training loop. Instead it exposes a stable interface and a concrete launch plan so the
repo can grow toward a distributed backend without polluting the single-node path.
"""

from __future__ import annotations

from nano_verl.backends.base import TrainingBackend, TrainingBackendPlan


class MegatronTrainingBackendStub(TrainingBackend):
    """Stub backend that documents the expected Megatron integration boundary."""

    backend_name = "megatron"

    def __init__(self, tensor_parallel_size: int = 1, pipeline_parallel_size: int = 1) -> None:
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size

    def plan(self, model_name: str, output_dir: str) -> TrainingBackendPlan:
        launch_example = (
            "torchrun --nproc_per_node=<gpus> pretrain_gpt.py "
            f"--load {model_name} --save {output_dir} "
            f"--tensor-model-parallel-size {self.tensor_parallel_size} "
            f"--pipeline-model-parallel-size {self.pipeline_parallel_size}"
        )
        return TrainingBackendPlan(
            backend_name=self.backend_name,
            summary="Distributed training backend for large-scale post-training.",
            launch_example=launch_example,
            notes=[
                "Use this backend only after separating actor, rollout, and reward workers.",
                "Keep vLLM as the rollout path; Megatron should own only the policy optimization side.",
                "A real integration needs checkpoint conversion, distributed optimizer wiring, and launcher scripts.",
            ],
        )

    def validate(self) -> None:
        raise NotImplementedError(
            "Megatron backend is a stub in nano_verl. "
            "Add a dedicated distributed trainer package before enabling this path."
        )

