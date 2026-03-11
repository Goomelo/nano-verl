"""Dataset loading for the real GRPO path."""

from __future__ import annotations

from dataclasses import dataclass

from nano_verl.real_data import build_hf_dataset
from nano_verl.trainer.config import GRPOExperimentConfig


@dataclass(slots=True)
class DatasetBundle:
    """Train/eval datasets passed into the trainer."""

    train_dataset: object
    eval_dataset: object | None


def build_dataset_bundle(config: GRPOExperimentConfig) -> DatasetBundle:
    """Load datasets from JSONL and return them in one object."""

    return DatasetBundle(
        train_dataset=build_hf_dataset(config.train_data),
        eval_dataset=build_hf_dataset(config.eval_data) if config.eval_data else None,
    )

