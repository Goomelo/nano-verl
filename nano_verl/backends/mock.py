"""Mock rollout backend for local pipeline exploration."""

from __future__ import annotations

import asyncio
import random

from nano_verl.backends.base import RolloutBackend, RolloutRequest
from nano_verl.types import RolloutCandidate, RolloutSample


class MockRolloutBackend(RolloutBackend):
    """Deterministic backend that simulates rollout candidates locally."""

    backend_name = "mock"

    def __init__(self, seed: int = 7, sleep_scale: float = 0.0) -> None:
        self.seed = seed
        self.sleep_scale = sleep_scale

    async def generate(self, request: RolloutRequest) -> RolloutSample:
        prompt_rng = random.Random(f"{self.seed}:{request.prompt.prompt_id}")
        candidates: list[RolloutCandidate] = []

        for sample_index in range(request.num_samples):
            text = self._generate_candidate_text(request.prompt, sample_index, prompt_rng)
            token_count = max(4, len(text.split()) + prompt_rng.randint(0, 4))
            latency_ms = round(prompt_rng.uniform(8.0, 24.0) + token_count * 0.75, 2)

            if self.sleep_scale > 0:
                await asyncio.sleep(latency_ms / 1000.0 * self.sleep_scale)

            candidates.append(
                RolloutCandidate(
                    sample_id=f"{request.prompt.prompt_id}-cand-{sample_index:02d}",
                    prompt_id=request.prompt.prompt_id,
                    text=text,
                    token_count=token_count,
                    latency_ms=latency_ms,
                    metadata={
                        "sample_index": sample_index,
                        "backend": self.backend_name,
                    },
                )
            )

        return RolloutSample(prompt=request.prompt, candidates=candidates)

    def _generate_candidate_text(self, prompt, sample_index: int, rng: random.Random) -> str:
        if prompt.task_type == "math":
            return self._generate_math_candidate(prompt.reference_answer.strip(), sample_index, rng)
        return self._generate_keyword_candidate(prompt.metadata, sample_index, rng)

    def _generate_math_candidate(self, reference_text: str, sample_index: int, rng: random.Random) -> str:
        reference_value = _safe_int(reference_text)
        wrong_offsets = (-3, -2, -1, 1, 2, 3)

        if reference_value is None:
            templates = [
                f"My best guess is {reference_text}.",
                f"Final answer: {reference_text}",
                "I could not parse the number, so I will answer unknown.",
            ]
            return templates[sample_index % len(templates)]

        wrong_value = reference_value + wrong_offsets[sample_index % len(wrong_offsets)]
        correct_templates = [
            f"{reference_value}",
            f"The answer is {reference_value}.",
            f"I computed it carefully. Final answer: {reference_value}.",
        ]
        wrong_templates = [
            f"{wrong_value}",
            f"I think the answer is {wrong_value}.",
            f"After a quick estimate, final answer: {wrong_value}.",
        ]

        if sample_index % 4 in (0, 1):
            return correct_templates[sample_index % len(correct_templates)]
        if sample_index % 4 == 2:
            return wrong_templates[sample_index % len(wrong_templates)]
        return (
            f"I add the numbers and get {reference_value if rng.random() > 0.4 else wrong_value}. "
            f"Final answer: {wrong_value}."
        )

    def _generate_keyword_candidate(self, metadata: dict[str, object], sample_index: int, rng: random.Random) -> str:
        keywords = [str(item) for item in metadata.get("keywords", [])]
        if not keywords:
            generic_outputs = [
                "This is a short generic answer.",
                "I would explain the idea step by step.",
                "Here is a concise response with one key point.",
            ]
            return generic_outputs[sample_index % len(generic_outputs)]

        included = keywords[: max(1, (sample_index % len(keywords)) + 1)]
        if sample_index % 3 == 2:
            included = included[:-1]
        rng.shuffle(included)
        return f"Relevant concepts: {', '.join(included)}."


def _safe_int(value: str) -> int | None:
    try:
        return int(float(value))
    except ValueError:
        return None

