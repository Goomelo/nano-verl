"""Dataset helpers for the native training path."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PromptRecord:
    """One prompt/solution pair used by the native engine."""

    prompt: str
    solution: str
    task: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_prompt_records(path: str | Path) -> list[PromptRecord]:
    """Load JSONL rows into prompt records."""

    file_path = Path(path)
    records: list[PromptRecord] = []

    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            metadata = dict(payload.get("metadata", {}))
            records.append(
                PromptRecord(
                    prompt=str(payload["prompt"]),
                    solution=str(payload["solution"]),
                    task=str(payload.get("task", "math")),
                    metadata=metadata,
                )
            )

    if not records:
        raise ValueError(f"No prompt records were found in {file_path}.")
    return records


def sample_batch(records: list[PromptRecord], batch_size: int, rng: random.Random) -> list[PromptRecord]:
    """Sample one training batch with replacement for simplicity."""

    return [rng.choice(records) for _ in range(batch_size)]

