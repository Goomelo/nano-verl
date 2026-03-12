#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/Users/gupengcheng/Documents/nano-verl"

python3 -m nano_verl.train_native \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --train-data "${ROOT_DIR}/data/grpo_math_train.jsonl" \
  --eval-data "${ROOT_DIR}/data/grpo_math_eval.jsonl" \
  --output-dir "${ROOT_DIR}/outputs/native-grpo-demo" \
  --steps 20 \
  --batch-size 1 \
  --num-generations 4 \
  --kl-coef 0.02 \
  --clip-range 0.2
