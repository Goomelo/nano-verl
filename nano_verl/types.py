"""Shared data structures used across the nano_verl prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SelectionStrategy = Literal["best_of_n", "rejection"]
RolloutBackendName = Literal["mock", "vllm-server", "megatron"]


@dataclass(slots=True)
class PromptExample:
    """A single training or evaluation prompt loaded from JSONL."""

    prompt_id: str
    prompt: str
    task_type: str
    reference_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RolloutCandidate:
    """One candidate answer produced by a rollout engine."""

    sample_id: str
    prompt_id: str
    text: str
    token_count: int
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RolloutSample:
    """All candidates generated for a prompt in a single rollout pass."""

    prompt: PromptExample
    candidates: list[RolloutCandidate]


@dataclass(slots=True)
class RewardResult:
    """Reward score assigned to one candidate."""

    prompt_id: str
    sample_id: str
    reward: float
    is_correct: bool
    reason: str


@dataclass(slots=True)
class SelectedSample:
    """Best candidate chosen for a prompt, with optional rejection."""

    prompt: PromptExample
    candidate: RolloutCandidate
    reward: RewardResult
    accepted: bool
    strategy: SelectionStrategy


@dataclass(slots=True)
class UpdateResult:
    """A small mock update summary that imitates a post-training step."""

    num_training_samples: int
    mean_reward: float
    pseudo_loss: float
    note: str


@dataclass(slots=True)
class EvaluationSummary:
    """Aggregate evaluation metrics across the full experiment."""

    num_prompts: int
    num_candidates: int
    num_selected: int
    avg_candidate_reward: float
    avg_selected_reward: float
    selected_accuracy: float
    end_to_end_accuracy: float
    oracle_accuracy: float
    acceptance_rate: float
    avg_selected_tokens: float
    avg_selected_latency_ms: float


@dataclass(slots=True)
class ExperimentConfig:
    """Runtime configuration parsed from the command line."""

    prompts_path: str
    rollout_backend: RolloutBackendName = "mock"
    rollout_model_name: str | None = None
    vllm_server_base_url: str | None = None
    vllm_api_key: str = "EMPTY"
    rollout_max_tokens: int = 128
    rollout_temperature: float = 0.8
    rollout_top_p: float = 0.95
    rollout_timeout_s: float = 60.0
    num_samples: int = 4
    strategy: SelectionStrategy = "best_of_n"
    min_reward: float = 1.0
    max_concurrency: int = 8
    seed: int = 7
    rollout_sleep_scale: float = 0.0
