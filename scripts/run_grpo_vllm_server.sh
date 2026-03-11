#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/Users/gupengcheng/Documents/nano-verl"

accelerate launch -m nano_verl.train_grpo \
  --model-name Qwen/Qwen3-8B \
  --train-data "${ROOT_DIR}/data/grpo_math_train.jsonl" \
  --eval-data "${ROOT_DIR}/data/grpo_math_eval.jsonl" \
  --output-dir "${ROOT_DIR}/outputs/grpo-qwen3-8b" \
  --use-peft \
  --generation-backend vllm-server \
  --vllm-server-base-url http://127.0.0.1:8000 \
  --max-steps 50 \
  --num-generations 4 \
  --logging-steps 1 \
  --save-steps 25
