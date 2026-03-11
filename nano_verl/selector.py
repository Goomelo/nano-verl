"""Selection strategies for turning rollouts into update data."""

from __future__ import annotations

from nano_verl.types import RewardResult, RolloutSample, SelectedSample, SelectionStrategy


def select_samples(
    rollouts: list[RolloutSample],
    reward_lookup: dict[str, RewardResult],
    strategy: SelectionStrategy,
    min_reward: float = 1.0,
) -> list[SelectedSample]:
    """Select the best candidate for each prompt.

    Strategies:
    - best_of_n: always take the highest-reward sample
    - rejection: take the highest-reward sample only if reward >= min_reward
    """

    selected: list[SelectedSample] = []

    for rollout in rollouts:
        best_candidate = max(
            rollout.candidates,
            key=lambda candidate: reward_lookup[candidate.sample_id].reward,
        )
        best_reward = reward_lookup[best_candidate.sample_id]
        accepted = strategy == "best_of_n" or best_reward.reward >= min_reward

        selected.append(
            SelectedSample(
                prompt=rollout.prompt,
                candidate=best_candidate,
                reward=best_reward,
                accepted=accepted,
                strategy=strategy,
            )
        )

    return selected

