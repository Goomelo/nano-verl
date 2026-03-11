"""Utilities for loading prompts from a JSONL file."""

from __future__ import annotations

import json
from pathlib import Path

from nano_verl.types import PromptExample


def load_prompts(path: str | Path) -> list[PromptExample]:
    """Read prompt examples from a JSONL file.

    Expected JSON keys:
    - id: unique prompt identifier
    - prompt: user-facing prompt text
    - task_type: task family such as "math" or "qa"
    - reference_answer: ground-truth answer used by reward/eval
    - metadata: optional extra task-specific fields
    """

    file_path = Path(path)
    prompts: list[PromptExample] = []

    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            payload = json.loads(line)
            prompt_id = str(payload.get("id") or f"example-{line_no:03d}")
            prompt_text = str(payload["prompt"])
            task_type = str(payload.get("task_type", "general"))
            reference_answer = str(payload.get("reference_answer", "")).strip()
            metadata = dict(payload.get("metadata", {}))

            if "keywords" in payload and "keywords" not in metadata:
                metadata["keywords"] = payload["keywords"]

            prompts.append(
                PromptExample(
                    prompt_id=prompt_id,
                    prompt=prompt_text,
                    task_type=task_type,
                    reference_answer=reference_answer,
                    metadata=metadata,
                )
            )

    return prompts

