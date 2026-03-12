"""CLI entrypoint for the native GRPO-like engine."""

from __future__ import annotations

from nano_verl.native import parse_native_grpo_args, run_native_grpo


def main() -> None:
    """Parse config and run the native engine."""

    config = parse_native_grpo_args()
    run_native_grpo(config)


if __name__ == "__main__":
    main()

