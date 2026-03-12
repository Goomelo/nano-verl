"""Native single-process GRPO-like training loop."""

from __future__ import annotations

import random
from pathlib import Path
from time import perf_counter

from nano_verl.native.config import NativeGRPOConfig
from nano_verl.native.data import PromptRecord, load_prompt_records, sample_batch
from nano_verl.native.grpo import (
    StepMetrics,
    apply_kl_penalty,
    compute_group_advantages,
    flatten_groups,
)
from nano_verl.native.policy import NativePolicy
from nano_verl.native.reporting import RunArtifactWriter
from nano_verl.native.rewards import compute_group_rewards


def run_native_grpo(config: NativeGRPOConfig) -> None:
    """Run a readable, native GRPO-like loop."""

    started_at = perf_counter()
    rng = random.Random(config.seed)
    train_records = load_prompt_records(config.train_data)
    eval_records = load_prompt_records(config.eval_data) if config.eval_data else []
    policy = NativePolicy.from_pretrained(
        model_name=config.model_name,
        reference_model_name=config.reference_model_name,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
        seed=config.seed,
    )
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    artifacts = RunArtifactWriter(config.output_dir)
    artifacts.write_config(config)
    last_step_metrics: StepMetrics | None = None
    last_eval_accuracy: float | None = None

    print("=" * 88)
    print("nano_verl native GRPO-like engine")
    print("=" * 88)
    print(f"model_name={config.model_name}")
    print(f"reference_model_name={config.reference_model_name or config.model_name}")
    print(f"train_records={len(train_records)}")
    print(f"eval_records={len(eval_records)}")
    print(f"steps={config.steps}, batch_size={config.batch_size}, num_generations={config.num_generations}")
    print(f"kl_coef={config.kl_coef:.4f}")
    print(f"clip_range={config.clip_range:.4f}")
    print(
        "dataflow: dataset -> sample completions -> raw reward -> reference KL penalty "
        "-> group normalize -> old-policy ratio objective -> eval"
    )
    print("note: this is a native educational GRPO-like loop with frozen reference and old-policy anchors.")
    print()

    for step in range(1, config.steps + 1):
        batch = sample_batch(train_records, config.batch_size, rng)
        step_metrics = _run_training_step(policy, batch, config)
        last_step_metrics = step_metrics
        eval_accuracy: float | None = None

        if step % config.log_interval == 0:
            print(
                f"[step {step:04d}] "
                f"loss={step_metrics.loss:.4f} "
                f"mean_raw_reward={step_metrics.mean_raw_reward:.4f} "
                f"mean_shaped_reward={step_metrics.mean_shaped_reward:.4f} "
                f"reward_std={step_metrics.reward_std:.4f} "
                f"mean_kl={step_metrics.mean_kl:.4f} "
                f"mean_ratio={step_metrics.mean_ratio:.4f} "
                f"clip_fraction={step_metrics.clip_fraction:.4f} "
                f"mean_advantage={step_metrics.mean_advantage:.4f} "
                f"accuracy={step_metrics.accuracy:.4f}"
            )

        if eval_records and step % config.eval_interval == 0:
            eval_accuracy = evaluate_policy(policy, eval_records, max_new_tokens=config.max_new_tokens)
            last_eval_accuracy = eval_accuracy
            print(f"[eval {step:04d}] greedy_accuracy={eval_accuracy:.4f}")

        artifacts.append_metric(step=step, metrics=step_metrics, eval_accuracy=eval_accuracy)

        if step % config.save_interval == 0:
            checkpoint_dir = Path(config.output_dir) / f"step_{step:04d}"
            policy.save(checkpoint_dir)
            print(f"[save {step:04d}] checkpoint={checkpoint_dir}")

    final_dir = Path(config.output_dir) / "final"
    policy.save(final_dir)
    if eval_records:
        last_eval_accuracy = evaluate_policy(policy, eval_records, max_new_tokens=config.max_new_tokens)
        print(f"[final eval] greedy_accuracy={last_eval_accuracy:.4f}")
    total_runtime_s = perf_counter() - started_at
    artifacts.write_benchmark_report(
        config=config,
        train_records=len(train_records),
        eval_records=len(eval_records),
        final_metrics=last_step_metrics,
        final_eval_accuracy=last_eval_accuracy,
        final_checkpoint=str(final_dir),
        total_runtime_s=total_runtime_s,
    )
    print(f"[done] final_checkpoint={final_dir}")
    print(f"[artifacts] metrics={artifacts.metrics_path}")
    print(f"[artifacts] report={artifacts.report_path}")
    print(f"[artifacts] summary={artifacts.summary_path}")


def evaluate_policy(policy: NativePolicy, records: list[PromptRecord], *, max_new_tokens: int) -> float:
    """Evaluate greedy accuracy on held-out prompts."""

    if not records:
        return 0.0

    correct = 0
    for record in records:
        completion = policy.greedy_completion(record.prompt, max_new_tokens=max_new_tokens)
        rewards = compute_group_rewards(record, [completion], enable_think_reward=False)
        if rewards.accuracy and rewards.accuracy[0] >= 1.0:
            correct += 1
    return correct / len(records)


def _run_training_step(
    policy: NativePolicy,
    batch: list[PromptRecord],
    config: NativeGRPOConfig,
) -> StepMetrics:
    policy.sync_old_policy()
    grouped_samples = []
    grouped_diagnostics = []
    grouped_raw_rewards = []
    grouped_kls = []
    grouped_accuracy = []

    for record in batch:
        samples = policy.sample_group(
            record.prompt,
            num_generations=config.num_generations,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        reward_breakdown = compute_group_rewards(
            record,
            [sample.completion_text for sample in samples],
            enable_think_reward=config.enable_think_reward,
        )
        diagnostics = policy.score_samples([samples])[0]
        grouped_samples.append(samples)
        grouped_diagnostics.append(diagnostics)
        grouped_raw_rewards.append(reward_breakdown.totals)
        grouped_kls.append([item.approx_kl for item in diagnostics])
        grouped_accuracy.append(reward_breakdown.accuracy)

    grouped_shaped_rewards = apply_kl_penalty(
        grouped_raw_rewards,
        grouped_kls,
        kl_coef=config.kl_coef,
    )
    grouped_advantages = compute_group_advantages(grouped_shaped_rewards)
    update_stats = policy.update_step(
        grouped_samples,
        grouped_advantages,
        grouped_diagnostics,
        clip_range=config.clip_range,
    )

    flat_raw_rewards = flatten_groups(grouped_raw_rewards)
    flat_shaped_rewards = flatten_groups(grouped_shaped_rewards)
    flat_kls = flatten_groups(grouped_kls)
    flat_advantages = flatten_groups(grouped_advantages)
    flat_accuracy = flatten_groups(grouped_accuracy)
    mean_raw_reward = sum(flat_raw_rewards) / len(flat_raw_rewards) if flat_raw_rewards else 0.0
    mean_shaped_reward = sum(flat_shaped_rewards) / len(flat_shaped_rewards) if flat_shaped_rewards else 0.0
    mean_kl = sum(flat_kls) / len(flat_kls) if flat_kls else 0.0
    reward_variance = (
        sum((reward - mean_shaped_reward) ** 2 for reward in flat_shaped_rewards) / len(flat_shaped_rewards)
        if flat_shaped_rewards
        else 0.0
    )
    mean_advantage = sum(flat_advantages) / len(flat_advantages) if flat_advantages else 0.0
    accuracy = sum(flat_accuracy) / len(flat_accuracy) if flat_accuracy else 0.0

    return StepMetrics(
        loss=update_stats.loss,
        mean_raw_reward=mean_raw_reward,
        mean_shaped_reward=mean_shaped_reward,
        reward_std=reward_variance**0.5,
        mean_kl=mean_kl,
        mean_ratio=update_stats.mean_ratio,
        clip_fraction=update_stats.clip_fraction,
        mean_advantage=mean_advantage,
        accuracy=accuracy,
    )
