"""High-level orchestration for the real GRPO path."""

from __future__ import annotations

from dataclasses import dataclass

from nano_verl.trainer.actor import build_peft_config, build_processing_class
from nano_verl.trainer.backends import resolve_generation_backend
from nano_verl.trainer.config import GRPOExperimentConfig
from nano_verl.trainer.dataflow import (
    build_dataflow,
    build_dataflow_diagram,
    build_stage_traces,
    format_dataflow,
    format_stage_traces,
)
from nano_verl.trainer.dataset import build_dataset_bundle
from nano_verl.trainer.rewards import build_reward_stack


@dataclass(slots=True)
class TrainerComponents:
    """All objects assembled before calling `trainer.train()`."""

    train_dataset: object
    eval_dataset: object | None
    training_args: object
    processing_class: object
    peft_config: object | None
    reward_funcs: list[object]


def build_trainer_components(config: GRPOExperimentConfig) -> TrainerComponents:
    """Assemble the GRPO training objects in a readable order."""

    dataset_bundle = build_dataset_bundle(config)
    backend_config = resolve_generation_backend(config)
    training_args = build_training_args(config, backend_config.config_kwargs)
    processing_class = build_processing_class(config.model_name)
    peft_config = build_peft_config(config)
    reward_funcs = build_reward_stack(config)

    print("=" * 88)
    print("nano_verl real GRPO training plan")
    print("=" * 88)
    print(f"model_name={config.model_name}")
    print(f"generation_backend={config.generation_backend}")
    print(f"backend_summary={backend_config.summary}")
    print(f"use_peft={config.use_peft}")
    print(f"load_in_4bit={config.load_in_4bit}")
    print(build_dataflow_diagram(config))
    print()
    print(format_dataflow(build_dataflow(config)))
    print()
    print(
        format_stage_traces(
            build_stage_traces(
                config,
                train_dataset=dataset_bundle.train_dataset,
                eval_dataset=dataset_bundle.eval_dataset,
                processing_class=processing_class,
                peft_config=peft_config,
                reward_funcs=reward_funcs,
                training_args=training_args,
            )
        )
    )
    print()

    return TrainerComponents(
        train_dataset=dataset_bundle.train_dataset,
        eval_dataset=dataset_bundle.eval_dataset,
        training_args=training_args,
        processing_class=processing_class,
        peft_config=peft_config,
        reward_funcs=reward_funcs,
    )


def build_training_args(config: GRPOExperimentConfig, backend_kwargs: dict[str, object]):
    """Build the TRL `GRPOConfig` from the high-level experiment config."""

    try:
        from trl import GRPOConfig
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "trl is required for real GRPO training. Install requirements-real.txt first."
        ) from exc

    config_kwargs = dict(
        output_dir=config.output_dir,
        max_steps=config.max_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        top_p=config.top_p,
        beta=config.beta,
        loss_type=config.loss_type,
        bf16=config.bf16,
        report_to=config.report_to,
        seed=config.seed,
        remove_unused_columns=False,
        gradient_checkpointing=config.gradient_checkpointing,
    )
    config_kwargs.update(backend_kwargs)
    return GRPOConfig(**config_kwargs)


def run_grpo_experiment(config: GRPOExperimentConfig) -> None:
    """Run the actual GRPO training loop."""

    try:
        from trl import GRPOTrainer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "trl is required for real GRPO training. Install requirements-real.txt first."
        ) from exc

    components = build_trainer_components(config)
    trainer = GRPOTrainer(
        model=config.model_name,
        reward_funcs=components.reward_funcs,
        args=components.training_args,
        train_dataset=components.train_dataset,
        eval_dataset=components.eval_dataset,
        processing_class=components.processing_class,
        peft_config=components.peft_config,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    components.processing_class.save_pretrained(config.output_dir)

    print("=" * 88)
    print("nano_verl real GRPO training finished")
    print("=" * 88)
    print(f"model_name={config.model_name}")
    print(f"output_dir={config.output_dir}")
    print(f"generation_backend={config.generation_backend}")
    print(f"use_peft={config.use_peft}")
    print(f"load_in_4bit={config.load_in_4bit}")
