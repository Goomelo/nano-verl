"""CLI entrypoint for the real GRPO training path.

Keep this file thin. The learning-oriented logic lives in `nano_verl.trainer.*`.
"""

from __future__ import annotations

from nano_verl.trainer import parse_grpo_args, run_grpo_experiment


def main() -> None:
    """Parse config and hand off to the orchestrator."""

    config = parse_grpo_args()
    run_grpo_experiment(config)


if __name__ == "__main__":
    main()
