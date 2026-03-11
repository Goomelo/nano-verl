#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/Users/gupengcheng/Documents/nano-verl"

accelerate launch -m nano_verl.train_grpo \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --train-data "${ROOT_DIR}/data/grpo_math_train.jsonl" \
  --eval-data "${ROOT_DIR}/data/grpo_math_eval.jsonl" \
  --output-dir "${ROOT_DIR}/outputs/grpo-qwen25-0p5b" \
  --use-peft \
  --load-in-4bit \
  --gradient-checkpointing \
  --max-steps 50 \
  --num-generations 4 \
  --logging-steps 1 \
  --save-steps 25

