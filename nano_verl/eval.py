"""Evaluation helpers for summarizing experiment quality."""

from __future__ import annotations

from statistics import mean

from nano_verl.types import EvaluationSummary, RewardResult, RolloutSample, SelectedSample


def evaluate_run(
    rollouts: list[RolloutSample],
    reward_lookup: dict[str, RewardResult],
    selections: list[SelectedSample],
) -> EvaluationSummary:
    """Compute prompt-level and sample-level quality metrics."""

    all_rewards = [
        reward_lookup[candidate.sample_id].reward
        for rollout in rollouts
        for candidate in rollout.candidates
    ]
    accepted = [selection for selection in selections if selection.accepted]
    selected_rewards = [selection.reward.reward for selection in accepted]
    selected_correct = [selection.reward.is_correct for selection in accepted]

    oracle_hits = 0
    for rollout in rollouts:
        if any(reward_lookup[candidate.sample_id].is_correct for candidate in rollout.candidates):
            oracle_hits += 1

    avg_candidate_reward = mean(all_rewards) if all_rewards else 0.0
    avg_selected_reward = mean(selected_rewards) if selected_rewards else 0.0
    selected_accuracy = mean(1.0 if flag else 0.0 for flag in selected_correct) if selected_correct else 0.0
    end_to_end_accuracy = (
        sum(1 for selection in accepted if selection.reward.is_correct) / len(selections)
        if selections
        else 0.0
    )
    acceptance_rate = len(accepted) / len(selections) if selections else 0.0
    avg_selected_tokens = mean(selection.candidate.token_count for selection in accepted) if accepted else 0.0
    avg_selected_latency_ms = mean(selection.candidate.latency_ms for selection in accepted) if accepted else 0.0

    return EvaluationSummary(
        num_prompts=len(rollouts),
        num_candidates=sum(len(rollout.candidates) for rollout in rollouts),
        num_selected=len(accepted),
        avg_candidate_reward=avg_candidate_reward,
        avg_selected_reward=avg_selected_reward,
        selected_accuracy=selected_accuracy,
        end_to_end_accuracy=end_to_end_accuracy,
        oracle_accuracy=oracle_hits / len(rollouts) if rollouts else 0.0,
        acceptance_rate=acceptance_rate,
        avg_selected_tokens=avg_selected_tokens,
        avg_selected_latency_ms=avg_selected_latency_ms,
    )

