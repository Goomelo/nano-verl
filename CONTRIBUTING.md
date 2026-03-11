# Contributing

## Scope

This repository is a learning-oriented open-source project.

Good contributions are:

- small and readable
- explicit about dataflow
- local-first by default
- honest about what is real and what is still a stub

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Install heavier runtime dependencies only when you need the GRPO path:

```bash
pip install -e ".[real]"
```

## Before Opening a PR

Run:

```bash
make compile
make test
```

If you touch the training path, also verify:

```bash
python -m nano_verl.train_grpo --help
```

## Contribution Guidelines

- Prefer small PRs over broad rewrites.
- Keep modules focused and readable.
- Preserve the learning value of the codebase.
- Document new dataflow stages and backend boundaries.
- Do not present stubs as finished integrations.

## Areas That Need Work

- native training loops beyond `TRL`
- better reward examples
- richer evaluation and logging
- distributed execution design
- real Megatron integration
- examples and tutorials
