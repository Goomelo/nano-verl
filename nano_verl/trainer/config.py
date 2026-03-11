"""Configuration for the real GRPO training path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TRAIN_PATH = Path(__file__).resolve().parents[2] / "data" / "grpo_math_train.jsonl"
DEFAULT_EVAL_PATH = Path(__file__).resolve().parents[2] / "data" / "grpo_math_eval.jsonl"


@dataclass(slots=True)
class GRPOExperimentConfig:
    """All user-facing knobs for a GRPO training run."""

    model_name: str
    train_data: str
    eval_data: str | None
    output_dir: str
    max_steps: int
    save_steps: int
    logging_steps: int
    learning_rate: float
    weight_decay: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_generations: int
    max_completion_length: int
    temperature: float
    top_p: float
    seed: int
    beta: float
    loss_type: str
    use_peft: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    load_in_4bit: bool
    bf16: bool
    gradient_checkpointing: bool
    generation_backend: str
    vllm_server_base_url: str | None
    vllm_server_host: str
    vllm_server_port: int
    vllm_gpu_memory_utilization: float
    vllm_tensor_parallel_size: int
    vllm_max_model_length: int | None
    vllm_model_impl: str
    megatron_tensor_parallel_size: int
    megatron_pipeline_parallel_size: int
    enable_think_reward: bool
    report_to: str


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser once so it stays readable."""

    parser = argparse.ArgumentParser(description="Train a small real GRPO experiment for nano_verl.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--eval-data", default=str(DEFAULT_EVAL_PATH))
    parser.add_argument("--output-dir", default="outputs/grpo-run")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--loss-type", default="dr_grpo", choices=("grpo", "dapo", "dr_grpo", "sapo"))
    parser.add_argument("--use-peft", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--generation-backend",
        choices=("transformers", "vllm-server", "vllm-colocate", "megatron"),
        default="transformers",
        help="Generation path used during online RL.",
    )
    parser.add_argument("--vllm-server-base-url", default=None)
    parser.add_argument("--vllm-server-host", default="127.0.0.1")
    parser.add_argument("--vllm-server-port", type=int, default=8000)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-max-model-length", type=int, default=None)
    parser.add_argument("--vllm-model-impl", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--megatron-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--megatron-pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--enable-think-reward", action="store_true")
    parser.add_argument("--report-to", default="none")
    return parser


def parse_grpo_args() -> GRPOExperimentConfig:
    """Parse CLI args into a typed config object."""

    args = build_arg_parser().parse_args()
    return GRPOExperimentConfig(
        model_name=args.model_name,
        train_data=args.train_data,
        eval_data=args.eval_data,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        beta=args.beta,
        loss_type=args.loss_type,
        use_peft=args.use_peft,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=list(args.lora_target_modules),
        load_in_4bit=args.load_in_4bit,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        generation_backend=args.generation_backend,
        vllm_server_base_url=args.vllm_server_base_url,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_max_model_length=args.vllm_max_model_length,
        vllm_model_impl=args.vllm_model_impl,
        megatron_tensor_parallel_size=args.megatron_tensor_parallel_size,
        megatron_pipeline_parallel_size=args.megatron_pipeline_parallel_size,
        enable_think_reward=args.enable_think_reward,
        report_to=args.report_to,
    )

