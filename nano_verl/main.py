"""Command-line entry point for the nano_verl prototype pipeline."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from statistics import mean

from nano_verl.backends import create_rollout_backend
from nano_verl.eval import evaluate_run
from nano_verl.metrics import MetricsTracker
from nano_verl.prompt_source import load_prompts
from nano_verl.reward import RuleReward
from nano_verl.rollout import BackendRolloutEngine
from nano_verl.selector import select_samples
from nano_verl.types import ExperimentConfig, RewardResult, RolloutSample, SelectedSample, UpdateResult

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional.
    torch = None


DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parents[1] / "data" / "prompts.jsonl"


def parse_args() -> ExperimentConfig:
    """Parse CLI arguments into a strongly typed experiment config."""

    parser = argparse.ArgumentParser(description="Run a local nano_verl prototype.")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS_PATH), help="Path to prompt JSONL.")
    parser.add_argument(
        "--rollout-backend",
        choices=("mock", "vllm-server", "megatron"),
        default="mock",
        help="Candidate generation backend.",
    )
    parser.add_argument("--rollout-model-name", default=None, help="Model name used by the rollout backend.")
    parser.add_argument("--vllm-server-base-url", default=None, help="Base URL of the vLLM server.")
    parser.add_argument("--vllm-api-key", default="EMPTY", help="Bearer token for the vLLM server.")
    parser.add_argument("--rollout-max-tokens", type=int, default=128, help="Max generated tokens per sample.")
    parser.add_argument("--rollout-temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--rollout-top-p", type=float, default=0.95, help="Top-p sampling value.")
    parser.add_argument("--rollout-timeout-s", type=float, default=60.0, help="Per-request timeout in seconds.")
    parser.add_argument("--num-samples", type=int, default=4, help="Candidates generated per prompt.")
    parser.add_argument(
        "--strategy",
        choices=("best_of_n", "rejection"),
        default="best_of_n",
        help="Selection strategy used after reward scoring.",
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=1.0,
        help="Reward threshold for rejection sampling.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Maximum concurrent prompts processed by rollout.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Seed used by the mock rollout engine.")
    parser.add_argument(
        "--rollout-sleep-scale",
        type=float,
        default=0.0,
        help="Scale simulated rollout latency. Keep 0.0 for fast local runs.",
    )
    args = parser.parse_args()

    return ExperimentConfig(
        prompts_path=args.prompts,
        rollout_backend=args.rollout_backend,
        rollout_model_name=args.rollout_model_name,
        vllm_server_base_url=args.vllm_server_base_url,
        vllm_api_key=args.vllm_api_key,
        rollout_max_tokens=args.rollout_max_tokens,
        rollout_temperature=args.rollout_temperature,
        rollout_top_p=args.rollout_top_p,
        rollout_timeout_s=args.rollout_timeout_s,
        num_samples=args.num_samples,
        strategy=args.strategy,
        min_reward=args.min_reward,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        rollout_sleep_scale=args.rollout_sleep_scale,
    )


async def run_experiment(config: ExperimentConfig) -> None:
    """Execute the full prompt -> rollout -> reward -> select -> update -> eval pipeline."""

    metrics = MetricsTracker()

    with metrics.timed("prompt_load"):
        prompts = load_prompts(config.prompts_path)
    metrics.increment("num_prompts", len(prompts))

    rollout_backend = create_rollout_backend(
        config.rollout_backend,
        seed=config.seed,
        sleep_scale=config.rollout_sleep_scale,
        vllm_server_base_url=config.vllm_server_base_url,
        vllm_api_key=config.vllm_api_key,
    )
    rollout_engine = BackendRolloutEngine(
        rollout_backend,
        model_name=config.rollout_model_name,
        max_tokens=config.rollout_max_tokens,
        temperature=config.rollout_temperature,
        top_p=config.rollout_top_p,
        timeout_s=config.rollout_timeout_s,
        seed=config.seed,
    )
    reward_fn = RuleReward()

    with metrics.timed("rollout", items=len(prompts) * config.num_samples):
        rollouts = await run_rollout_stage(
            prompts=prompts,
            engine=rollout_engine,
            num_samples=config.num_samples,
            max_concurrency=config.max_concurrency,
        )
    metrics.increment("num_candidates", sum(len(rollout.candidates) for rollout in rollouts))

    with metrics.timed("reward", items=sum(len(rollout.candidates) for rollout in rollouts)):
        reward_lookup = run_reward_stage(rollouts, reward_fn)
    metrics.increment("num_reward_calls", len(reward_lookup))

    with metrics.timed("select", items=len(rollouts)):
        selections = select_samples(
            rollouts=rollouts,
            reward_lookup=reward_lookup,
            strategy=config.strategy,
            min_reward=config.min_reward,
        )
    metrics.increment("num_selected", sum(1 for item in selections if item.accepted))

    with metrics.timed("update", items=sum(1 for item in selections if item.accepted)):
        update_result = run_update_stage(selections)

    with metrics.timed("eval", items=len(rollouts)):
        evaluation = evaluate_run(rollouts, reward_lookup, selections)

    print_report(
        config=config,
        rollouts=rollouts,
        reward_lookup=reward_lookup,
        selections=selections,
        update_result=update_result,
        evaluation=evaluation,
        metrics=metrics,
    )


async def run_rollout_stage(
    prompts,
    engine: BackendRolloutEngine,
    num_samples: int,
    max_concurrency: int,
) -> list[RolloutSample]:
    """Run the rollout stage with bounded concurrency."""

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run_single(prompt):
        async with semaphore:
            return await engine.generate(prompt, num_samples)

    tasks = [_run_single(prompt) for prompt in prompts]
    return list(await asyncio.gather(*tasks))


def run_reward_stage(
    rollouts: list[RolloutSample],
    reward_fn: RuleReward,
) -> dict[str, RewardResult]:
    """Score every rollout candidate and store results by sample id."""

    reward_lookup: dict[str, RewardResult] = {}
    for rollout in rollouts:
        for candidate in rollout.candidates:
            reward_lookup[candidate.sample_id] = reward_fn.score(rollout.prompt, candidate)
    return reward_lookup


def run_update_stage(selections: list[SelectedSample]) -> UpdateResult:
    """Simulate a lightweight update step from accepted samples.

    In real VERL-style training this stage would optimize a policy.
    Here we convert selected samples into a compact training summary
    so the end-to-end data flow remains easy to inspect.
    """

    accepted = [selection for selection in selections if selection.accepted]
    if not accepted:
        return UpdateResult(
            num_training_samples=0,
            mean_reward=0.0,
            pseudo_loss=1.0,
            note="No samples passed selection, so the update step is skipped.",
        )

    rewards = [selection.reward.reward for selection in accepted]
    if torch is not None:
        mean_reward = float(torch.tensor(rewards, dtype=torch.float32).mean().item())
    else:
        mean_reward = mean(rewards)

    pseudo_loss = max(0.0, 1.0 - mean_reward)
    note = (
        "Mock update completed by treating accepted samples as supervised targets. "
        "This keeps the pipeline structure close to post-training without implementing RL optimization."
    )
    return UpdateResult(
        num_training_samples=len(accepted),
        mean_reward=mean_reward,
        pseudo_loss=pseudo_loss,
        note=note,
    )


def print_report(
    config: ExperimentConfig,
    rollouts: list[RolloutSample],
    reward_lookup: dict[str, RewardResult],
    selections: list[SelectedSample],
    update_result,
    evaluation,
    metrics: MetricsTracker,
) -> None:
    """Print a detailed, resume-friendly experiment report to stdout."""

    selection_by_prompt = {selection.prompt.prompt_id: selection for selection in selections}

    print("=" * 88)
    print("nano_verl: local VERL-style post-training pipeline prototype")
    print("=" * 88)
    print(
        f"config: prompts={config.prompts_path}, rollout_backend={config.rollout_backend}, "
        f"rollout_model_name={config.rollout_model_name}, num_samples={config.num_samples}, "
        f"strategy={config.strategy}, min_reward={config.min_reward}, "
        f"max_concurrency={config.max_concurrency}, seed={config.seed}"
    )
    print()

    print("[stage metrics]")
    for stage_metric in metrics.stage_metrics():
        print(
            f"- {stage_metric.name:<12} "
            f"time={stage_metric.duration_s:>7.4f}s "
            f"items={stage_metric.items:>3d} "
            f"throughput={stage_metric.throughput:>8.2f}/s"
        )
    print(f"- {'total':<12} time={metrics.total_duration_s():>7.4f}s")
    print()

    print("[update summary]")
    print(
        f"- training_samples={update_result.num_training_samples}, "
        f"mean_reward={update_result.mean_reward:.3f}, "
        f"pseudo_loss={update_result.pseudo_loss:.3f}"
    )
    print(f"- note={update_result.note}")
    print()

    print("[evaluation summary]")
    print(
        f"- prompts={evaluation.num_prompts}, candidates={evaluation.num_candidates}, "
        f"accepted={evaluation.num_selected}"
    )
    print(
        f"- avg_candidate_reward={evaluation.avg_candidate_reward:.3f}, "
        f"avg_selected_reward={evaluation.avg_selected_reward:.3f}"
    )
    print(
        f"- selected_accuracy={evaluation.selected_accuracy:.3f}, "
        f"end_to_end_accuracy={evaluation.end_to_end_accuracy:.3f}, "
        f"oracle_accuracy={evaluation.oracle_accuracy:.3f}"
    )
    print(
        f"- acceptance_rate={evaluation.acceptance_rate:.3f}, "
        f"avg_selected_tokens={evaluation.avg_selected_tokens:.2f}, "
        f"avg_selected_latency_ms={evaluation.avg_selected_latency_ms:.2f}"
    )
    print()

    print("[per-prompt details]")
    for rollout in rollouts:
        selection = selection_by_prompt[rollout.prompt.prompt_id]
        status = "accepted" if selection.accepted else "rejected"
        print(
            f"{rollout.prompt.prompt_id} | task={rollout.prompt.task_type} | "
            f"reference={rollout.prompt.reference_answer} | selection={status}"
        )
        print(f"prompt: {rollout.prompt.prompt}")
        for index, candidate in enumerate(rollout.candidates, start=1):
            reward = reward_lookup[candidate.sample_id]
            marker = " <- selected" if candidate.sample_id == selection.candidate.sample_id else ""
            print(
                f"  [{index}] reward={reward.reward:.2f} correct={int(reward.is_correct)} "
                f"tokens={candidate.token_count:>2d} latency={candidate.latency_ms:>6.2f}ms{marker}"
            )
            print(f"      text: {candidate.text}")
            print(f"      why : {reward.reason}")
        print()


def main() -> None:
    """Entrypoint used by `python -m nano_verl.main`."""

    config = parse_args()
    asyncio.run(run_experiment(config))


if __name__ == "__main__":
    main()
