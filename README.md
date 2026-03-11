# nano_verl

`nano_verl` is a small, readable learning project for understanding LLM post-training dataflow.

It is not a reimplementation of full `verl`. The goal is narrower:

- make the core stages visible
- keep the code local-first and hackable
- provide one toy pipeline and one real training entrypoint
- stay structured enough to evolve in public on GitHub

## What This Repo Contains

Two paths live in the same repository:

1. `python -m nano_verl.main`
   A toy pipeline that makes the stages explicit:
   `prompt -> rollout -> reward -> select -> update -> eval`

2. `accelerate launch -m nano_verl.train_grpo`
   A real GRPO-oriented training entrypoint built around `TRL`, with optional `PEFT/QLoRA`
   and optional `vLLM` generation acceleration.

## Why This Exists

`verl` is a production-grade distributed RL/post-training framework.

This repo is intentionally smaller. It focuses on:

- understanding the shape of the pipeline
- reading the code end-to-end in one sitting
- experimenting locally before touching cluster-scale systems
- growing toward a better open-source learning project over time

## Current Status

Implemented today:

- toy post-training pipeline with metrics and evaluation
- typed prompt / rollout / reward / selection data structures
- mock rollout backend
- `vLLM` server rollout adapter
- real GRPO entrypoint using `TRL`
- backend separation for rollout and training concerns
- stage-by-stage dataflow trace for learning
- `Megatron` backend stub with a clear boundary, not a fake implementation

Not implemented yet:

- native PPO / GRPO training loop
- distributed actor / rollout / reward workers
- critic / reference model / KL controller stack
- real Megatron training backend
- cluster orchestration comparable to `verl`

## Repository Layout

```text
.
├── data/
├── nano_verl/
│   ├── backends/
│   ├── trainer/
│   ├── main.py
│   └── train_grpo.py
├── scripts/
├── tests/
├── .github/workflows/
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── ROADMAP.md
└── pyproject.toml
```

## Code Map

Start here if you want to learn the codebase:

1. `nano_verl/trainer/dataflow.py`
2. `nano_verl/trainer/config.py`
3. `nano_verl/trainer/dataset.py`
4. `nano_verl/trainer/actor.py`
5. `nano_verl/trainer/rewards.py`
6. `nano_verl/trainer/backends.py`
7. `nano_verl/trainer/orchestrator.py`

Toy pipeline modules:

- `nano_verl/main.py`
- `nano_verl/rollout.py`
- `nano_verl/reward.py`
- `nano_verl/selector.py`
- `nano_verl/eval.py`

Backend modules:

- `nano_verl/backends/mock.py`
- `nano_verl/backends/vllm.py`
- `nano_verl/backends/megatron.py`

## Quick Start

### 1. Install the lightweight learning setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Run the toy pipeline

```bash
python -m nano_verl.main --num-samples 4 --strategy best_of_n
```

### 3. Run tests

```bash
python3 -m unittest discover -s tests
```

## Real Training Setup

Install the heavier dependencies only when you want the GRPO path:

```bash
pip install -e ".[real]"
```

Single-GPU baseline:

```bash
accelerate launch -m nano_verl.train_grpo \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --train-data data/grpo_math_train.jsonl \
  --eval-data data/grpo_math_eval.jsonl \
  --output-dir outputs/grpo-qwen25-0p5b \
  --use-peft \
  --load-in-4bit \
  --max-steps 50
```

The GRPO startup path prints:

- a compact ASCII dataflow graph
- stage-by-stage input/output descriptions
- key object shapes such as dataset schema, tokenizer size, reward stack, and rollout shape

## vLLM Rollout

You can swap the toy rollout engine with a real `vLLM` server:

```bash
python -m nano_verl.main \
  --rollout-backend vllm-server \
  --rollout-model-name Qwen/Qwen3-8B \
  --vllm-server-base-url http://127.0.0.1:8000 \
  --num-samples 4
```

You can also use the helper scripts in `scripts/`.

## Development Commands

```bash
make test
make compile
make toy
```

## Project Direction

This repo is intended to grow in small, readable steps.

Short-term direction is tracked in [ROADMAP.md](ROADMAP.md). Contribution guidelines are in
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
