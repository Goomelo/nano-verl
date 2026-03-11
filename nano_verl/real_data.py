"""Dataset helpers for the real GRPO training path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_rl_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL RL dataset and normalize it to TRL-friendly columns.

    Required input keys per line:
    - prompt: raw user prompt string
    - solution: expected answer used by the reward function

    Optional keys:
    - task: task family, defaults to "math"
    - metadata: arbitrary extra columns forwarded to reward functions
    """

    file_path = Path(path)
    records: list[dict[str, Any]] = []

    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            payload = json.loads(line)
            prompt = str(payload["prompt"])
            solution = str(payload["solution"])
            task = str(payload.get("task", "math"))
            metadata = dict(payload.get("metadata", {}))

            # Conversational format plays better with instruct/chat models like Qwen.
            records.append(
                {
                    "prompt": [{"role": "user", "content": prompt}],
                    "solution": solution,
                    "task": task,
                    **metadata,
                }
            )

    if not records:
        raise ValueError(f"No training records were found in {file_path}.")

    return records


def build_hf_dataset(path: str | Path):
    """Create a Hugging Face Dataset lazily so this module imports without extras."""

    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError(
            "datasets is required for real GRPO training. Install requirements-real.txt first."
        ) from exc

    return Dataset.from_list(load_rl_jsonl(path))

