import json
import tempfile
import unittest
from pathlib import Path

from nano_verl.native.config import NativeGRPOConfig
from nano_verl.native.grpo import StepMetrics
from nano_verl.native.reporting import RunArtifactWriter, render_benchmark_report


def _config() -> NativeGRPOConfig:
    return NativeGRPOConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        train_data="data/grpo_math_train.jsonl",
        eval_data="data/grpo_math_eval.jsonl",
        output_dir="outputs/native-grpo",
        steps=10,
        batch_size=1,
        num_generations=4,
        max_new_tokens=64,
        learning_rate=1e-5,
        weight_decay=0.0,
        kl_coef=0.02,
        clip_range=0.2,
        temperature=0.8,
        top_p=0.95,
        seed=7,
        device="cpu",
        reference_model_name=None,
        log_interval=1,
        eval_interval=5,
        save_interval=10,
        enable_think_reward=False,
    )


def _metrics() -> StepMetrics:
    return StepMetrics(
        loss=0.123,
        mean_raw_reward=0.5,
        mean_shaped_reward=0.45,
        reward_std=0.2,
        mean_kl=0.05,
        mean_ratio=1.02,
        clip_fraction=0.1,
        mean_advantage=0.0,
        accuracy=0.5,
    )


class NativeReportingTests(unittest.TestCase):
    def test_render_benchmark_report_mentions_artifacts(self) -> None:
        report = render_benchmark_report(
            config=_config(),
            train_records=10,
            eval_records=4,
            final_metrics=_metrics(),
            final_eval_accuracy=0.75,
            final_checkpoint="outputs/native-grpo/final",
            total_runtime_s=1.5,
        )

        self.assertIn("Native GRPO Benchmark Report", report)
        self.assertIn("metrics.jsonl", report)
        self.assertIn("final_eval_accuracy", report)

    def test_artifact_writer_writes_metrics_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            writer = RunArtifactWriter(tmp_dir)
            writer.write_config(_config())
            writer.append_metric(step=1, metrics=_metrics(), eval_accuracy=0.5)
            writer.write_benchmark_report(
                config=_config(),
                train_records=10,
                eval_records=4,
                final_metrics=_metrics(),
                final_eval_accuracy=0.75,
                final_checkpoint="outputs/native-grpo/final",
                total_runtime_s=1.5,
            )

            metrics_lines = writer.metrics_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(metrics_lines), 1)
            record = json.loads(metrics_lines[0])
            self.assertEqual(record["step"], 1)
            self.assertAlmostEqual(record["mean_kl"], 0.05)

            summary = json.loads(writer.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["train_records"], 10)
            self.assertAlmostEqual(summary["final_eval_accuracy"], 0.75)

            report_text = writer.report_path.read_text(encoding="utf-8")
            self.assertIn("benchmark_report.md", str(writer.report_path))
            self.assertIn("Final Metrics", report_text)
