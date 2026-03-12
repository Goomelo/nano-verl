"""Artifacts and reporting for native training runs."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nano_verl.native.config import NativeGRPOConfig
from nano_verl.native.grpo import StepMetrics


class RunArtifactWriter:
    """Write run artifacts in simple text formats.

    Files:
    - run_config.json
    - metrics.jsonl
    - benchmark_report.md
    - benchmark_summary.json
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.config_path = self.output_dir / "run_config.json"
        self.report_path = self.output_dir / "benchmark_report.md"
        self.summary_path = self.output_dir / "benchmark_summary.json"

    def write_config(self, config: NativeGRPOConfig) -> None:
        """Persist the run configuration."""

        self.config_path.write_text(
            json.dumps(_to_jsonable(config), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def append_metric(
        self,
        *,
        step: int,
        metrics: StepMetrics,
        eval_accuracy: float | None = None,
    ) -> None:
        """Append one step metric record to metrics.jsonl."""

        record = {
            "timestamp_utc": _utc_now(),
            "step": step,
            **_to_jsonable(metrics),
            "eval_accuracy": eval_accuracy,
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def write_benchmark_report(
        self,
        *,
        config: NativeGRPOConfig,
        train_records: int,
        eval_records: int,
        final_metrics: StepMetrics | None,
        final_eval_accuracy: float | None,
        final_checkpoint: str,
        total_runtime_s: float,
    ) -> None:
        """Write a Markdown report plus a JSON summary."""

        report = render_benchmark_report(
            config=config,
            train_records=train_records,
            eval_records=eval_records,
            final_metrics=final_metrics,
            final_eval_accuracy=final_eval_accuracy,
            final_checkpoint=final_checkpoint,
            total_runtime_s=total_runtime_s,
        )
        self.report_path.write_text(report, encoding="utf-8")
        summary = build_benchmark_summary(
            config=config,
            train_records=train_records,
            eval_records=eval_records,
            final_metrics=final_metrics,
            final_eval_accuracy=final_eval_accuracy,
            final_checkpoint=final_checkpoint,
            total_runtime_s=total_runtime_s,
        )
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def render_benchmark_report(
    *,
    config: NativeGRPOConfig,
    train_records: int,
    eval_records: int,
    final_metrics: StepMetrics | None,
    final_eval_accuracy: float | None,
    final_checkpoint: str,
    total_runtime_s: float,
) -> str:
    """Render a Markdown report for one run."""

    lines = [
        "# Native GRPO Benchmark Report",
        "",
        "## Run",
        "",
        f"- model_name: `{config.model_name}`",
        f"- reference_model_name: `{config.reference_model_name or config.model_name}`",
        f"- train_records: `{train_records}`",
        f"- eval_records: `{eval_records}`",
        f"- steps: `{config.steps}`",
        f"- batch_size: `{config.batch_size}`",
        f"- num_generations: `{config.num_generations}`",
        f"- max_new_tokens: `{config.max_new_tokens}`",
        f"- kl_coef: `{config.kl_coef}`",
        f"- clip_range: `{config.clip_range}`",
        f"- total_runtime_s: `{total_runtime_s:.4f}`",
        "",
        "## Final Metrics",
        "",
    ]

    if final_metrics is None:
        lines.append("- no training steps were recorded")
    else:
        lines.extend(
            [
                f"- loss: `{final_metrics.loss:.6f}`",
                f"- mean_raw_reward: `{final_metrics.mean_raw_reward:.6f}`",
                f"- mean_shaped_reward: `{final_metrics.mean_shaped_reward:.6f}`",
                f"- reward_std: `{final_metrics.reward_std:.6f}`",
                f"- mean_kl: `{final_metrics.mean_kl:.6f}`",
                f"- mean_ratio: `{final_metrics.mean_ratio:.6f}`",
                f"- clip_fraction: `{final_metrics.clip_fraction:.6f}`",
                f"- mean_advantage: `{final_metrics.mean_advantage:.6f}`",
                f"- accuracy: `{final_metrics.accuracy:.6f}`",
            ]
        )

    lines.extend(
        [
            f"- final_eval_accuracy: `{0.0 if final_eval_accuracy is None else final_eval_accuracy:.6f}`",
            f"- final_checkpoint: `{final_checkpoint}`",
            "",
            "## Artifacts",
            "",
            "- `run_config.json`",
            "- `metrics.jsonl`",
            "- `benchmark_summary.json`",
            "- `benchmark_report.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_benchmark_summary(
    *,
    config: NativeGRPOConfig,
    train_records: int,
    eval_records: int,
    final_metrics: StepMetrics | None,
    final_eval_accuracy: float | None,
    final_checkpoint: str,
    total_runtime_s: float,
) -> dict[str, Any]:
    """Build a machine-readable summary of the run."""

    return {
        "model_name": config.model_name,
        "reference_model_name": config.reference_model_name or config.model_name,
        "train_records": train_records,
        "eval_records": eval_records,
        "steps": config.steps,
        "batch_size": config.batch_size,
        "num_generations": config.num_generations,
        "max_new_tokens": config.max_new_tokens,
        "kl_coef": config.kl_coef,
        "clip_range": config.clip_range,
        "total_runtime_s": total_runtime_s,
        "final_metrics": None if final_metrics is None else _to_jsonable(final_metrics),
        "final_eval_accuracy": final_eval_accuracy,
        "final_checkpoint": final_checkpoint,
    }


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()

