"""Configuration for the native GRPO-like training path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TRAIN_PATH = Path(__file__).resolve().parents[2] / "data" / "grpo_math_train.jsonl"
DEFAULT_EVAL_PATH = Path(__file__).resolve().parents[2] / "data" / "grpo_math_eval.jsonl"


@dataclass(slots=True)
class NativeGRPOConfig:
    """Config for the native single-process GRPO-like engine.

    This path is intentionally small and readable. It does not try to cover the full
    feature set of TRL or verl.
    """

    model_name: str
    train_data: str
    eval_data: str | None
    output_dir: str
    steps: int
    batch_size: int
    num_generations: int
    max_new_tokens: int
    learning_rate: float
    weight_decay: float
    kl_coef: float
    clip_range: float
    temperature: float
    top_p: float
    seed: int
    device: str
    reference_model_name: str | None
    log_interval: int
    eval_interval: int
    save_interval: int
    enable_think_reward: bool


def build_native_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the native engine."""

    parser = argparse.ArgumentParser(description="Run a native single-process GRPO-like experiment.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--eval-data", default=str(DEFAULT_EVAL_PATH))
    parser.add_argument("--output-dir", default="outputs/native-grpo")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--kl-coef", type=float, default=0.02, help="KL penalty coefficient against the reference model.")
    parser.add_argument("--clip-range", type=float, default=0.2, help="Clip range for the policy ratio objective.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    parser.add_argument(
        "--reference-model-name",
        default=None,
        help="Frozen reference model used for KL penalty. Defaults to --model-name.",
    )
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--enable-think-reward", action="store_true")
    return parser


def parse_native_grpo_args() -> NativeGRPOConfig:
    """Parse CLI args into a typed config object."""

    args = build_native_arg_parser().parse_args()
    return NativeGRPOConfig(
        model_name=args.model_name,
        train_data=args.train_data,
        eval_data=args.eval_data,
        output_dir=args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        kl_coef=args.kl_coef,
        clip_range=args.clip_range,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        device=args.device,
        reference_model_name=args.reference_model_name,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        enable_think_reward=args.enable_think_reward,
    )
