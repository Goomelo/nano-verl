"""Human-readable dataflow and trace logging for the real training path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nano_verl.trainer.config import GRPOExperimentConfig


@dataclass(slots=True)
class DataflowStage:
    """One stage in the real training pipeline."""

    name: str
    inputs: str
    outputs: str
    purpose: str


@dataclass(slots=True)
class StageTrace:
    """Trace record for one stage's key runtime objects."""

    name: str
    key_objects: list[str]


def build_dataflow(config: GRPOExperimentConfig) -> list[DataflowStage]:
    """Return the end-to-end flow that this training run will execute."""

    rollout_system = (
        "TRL + vLLM rollout backend"
        if config.generation_backend in {"vllm-server", "vllm-colocate"}
        else "TRL + transformers.generate"
    )
    actor_stack = "base model + LoRA adapters" if config.use_peft else "base model full weights"

    return [
        DataflowStage(
            name="load_dataset",
            inputs="JSONL rows with prompt, solution, task",
            outputs="HF Dataset with chat-format prompt column",
            purpose="Normalize raw records into the structure expected by GRPOTrainer.",
        ),
        DataflowStage(
            name="actor_setup",
            inputs=f"model={config.model_name}, actor_stack={actor_stack}",
            outputs="tokenizer, model loading kwargs, optional PEFT config",
            purpose="Prepare the policy actor that will generate online completions.",
        ),
        DataflowStage(
            name="rollout",
            inputs=f"prompt batch + policy actor via {rollout_system}",
            outputs=f"{config.num_generations} sampled completions per prompt",
            purpose="Generate on-policy samples that GRPO will compare inside each group.",
        ),
        DataflowStage(
            name="reward",
            inputs="sampled completions + reference solution + task metadata",
            outputs="per-completion scalar rewards",
            purpose="Score correctness and optional formatting behavior such as <think> tags.",
        ),
        DataflowStage(
            name="group_relative_update",
            inputs="grouped rewards + old policy outputs",
            outputs="GRPO loss and updated policy weights",
            purpose="Optimize the actor using relative advantages instead of a separate critic.",
        ),
        DataflowStage(
            name="checkpoint_eval",
            inputs="updated policy + eval dataset",
            outputs="saved checkpoints, trainer logs, optional eval metrics",
            purpose="Persist the run and make the training trace inspectable.",
        ),
    ]


def build_dataflow_diagram(config: GRPOExperimentConfig) -> str:
    """Render a compact ASCII diagram of the real GRPO pipeline."""

    rollout_node = "vLLM rollout" if config.generation_backend in {"vllm-server", "vllm-colocate"} else "transformers rollout"
    actor_node = "actor + LoRA" if config.use_peft else "actor"
    return "\n".join(
        [
            "[dataflow graph]",
            "JSONL dataset",
            "  |",
            "  v",
            "HF Dataset -> tokenizer",
            "  |",
            "  v",
            f"{actor_node} -> {rollout_node}",
            "  |",
            "  v",
            "sampled completions -> reward funcs",
            "  |",
            "  v",
            "grouped rewards -> GRPO update",
            "  |",
            "  v",
            "checkpoint + eval",
        ]
    )


def format_dataflow(stages: list[DataflowStage]) -> str:
    """Render the dataflow in a console-friendly format."""

    lines = ["[dataflow]"]
    for index, stage in enumerate(stages, start=1):
        lines.append(
            f"{index}. {stage.name}: inputs={stage.inputs} -> outputs={stage.outputs} | purpose={stage.purpose}"
        )
    return "\n".join(lines)


def build_stage_traces(
    config: GRPOExperimentConfig,
    *,
    train_dataset: object,
    eval_dataset: object | None,
    processing_class: object,
    peft_config: object | None,
    reward_funcs: list[object],
    training_args: object,
) -> list[StageTrace]:
    """Build a readable trace log for the major training stages."""

    train_len = _safe_len(train_dataset)
    eval_len = _safe_len(eval_dataset)
    train_row = _safe_first_row(train_dataset)
    reward_names = [getattr(func, "__name__", func.__class__.__name__) for func in reward_funcs]

    rollout_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
    rollout_shape = (
        f"completions ~= [{rollout_batch} prompts x {config.num_generations} generations x <= {config.max_completion_length} tokens]"
    )

    traces = [
        StageTrace(
            name="load_dataset",
            key_objects=[
                f"train_dataset={_describe_dataset(train_dataset, train_len)}",
                f"eval_dataset={_describe_dataset(eval_dataset, eval_len)}",
                f"train_example={_describe_mapping_shape(train_row)}",
            ],
        ),
        StageTrace(
            name="actor_setup",
            key_objects=[
                f"processing_class={processing_class.__class__.__name__}",
                f"tokenizer_vocab_size={_safe_vocab_size(processing_class)}",
                f"peft_config={_describe_peft(peft_config)}",
            ],
        ),
        StageTrace(
            name="rollout",
            key_objects=[
                f"generation_backend={config.generation_backend}",
                rollout_shape,
                f"sampling=(temperature={config.temperature}, top_p={config.top_p})",
            ],
        ),
        StageTrace(
            name="reward",
            key_objects=[
                f"reward_funcs={reward_names}",
                "reward_shape=[num_prompts x num_generations]",
                "reward_inputs=[completion text, solution, task metadata]",
            ],
        ),
        StageTrace(
            name="group_relative_update",
            key_objects=[
                f"loss_type={config.loss_type}",
                f"beta={config.beta}",
                f"train_batch=(per_device={config.per_device_train_batch_size}, grad_accum={config.gradient_accumulation_steps})",
            ],
        ),
        StageTrace(
            name="checkpoint_eval",
            key_objects=[
                f"output_dir={config.output_dir}",
                f"save_steps={config.save_steps}",
                f"training_args={_describe_training_args(training_args)}",
            ],
        ),
    ]
    return traces


def format_stage_traces(traces: list[StageTrace]) -> str:
    """Render stage traces in a console-friendly format."""

    lines = ["[trace]"]
    for trace in traces:
        lines.append(f"- {trace.name}")
        for item in trace.key_objects:
            lines.append(f"    {item}")
    return "\n".join(lines)


def _safe_len(value: object | None) -> int | None:
    if value is None:
        return None
    try:
        return len(value)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return None


def _safe_first_row(dataset: object | None) -> Any:
    if dataset is None:
        return None
    try:
        if len(dataset) == 0:  # type: ignore[arg-type]
            return None
        return dataset[0]  # type: ignore[index]
    except (TypeError, KeyError, IndexError, AttributeError):
        return None


def _describe_dataset(dataset: object | None, dataset_len: int | None) -> str:
    if dataset is None:
        return "None"
    dataset_name = dataset.__class__.__name__
    if dataset_len is None:
        return dataset_name
    return f"{dataset_name}(rows={dataset_len})"


def _describe_mapping_shape(value: Any) -> str:
    if not isinstance(value, dict):
        return repr(value)
    keys = list(value.keys())
    parts = [f"{key}={_infer_shape(value[key])}" for key in keys]
    return "{" + ", ".join(parts) + "}"


def _infer_shape(value: Any) -> str:
    if isinstance(value, str):
        return "str"
    if isinstance(value, (int, float, bool)):
        return type(value).__name__
    if isinstance(value, list):
        if not value:
            return "list[0]"
        first = value[0]
        if isinstance(first, dict):
            return f"list[{len(value)}] of dict(keys={list(first.keys())})"
        return f"list[{len(value)}] of {type(first).__name__}"
    if isinstance(value, dict):
        return f"dict(keys={list(value.keys())})"
    return type(value).__name__


def _safe_vocab_size(processing_class: object) -> int | str:
    try:
        return len(processing_class)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return "unknown"


def _describe_peft(peft_config: object | None) -> str:
    if peft_config is None:
        return "None"
    target_modules = getattr(peft_config, "target_modules", None)
    return f"{peft_config.__class__.__name__}(target_modules={target_modules})"


def _describe_training_args(training_args: object) -> str:
    fields = ["max_steps", "learning_rate", "num_generations", "max_completion_length"]
    pairs: list[str] = []
    for field_name in fields:
        if hasattr(training_args, field_name):
            pairs.append(f"{field_name}={getattr(training_args, field_name)}")
    return ", ".join(pairs) if pairs else training_args.__class__.__name__
