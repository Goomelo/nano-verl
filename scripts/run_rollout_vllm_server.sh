#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/Users/gupengcheng/Documents/nano-verl"

python3 -m nano_verl.main \
  --prompts "${ROOT_DIR}/data/prompts.jsonl" \
  --rollout-backend vllm-server \
  --rollout-model-name Qwen/Qwen3-8B \
  --vllm-server-base-url http://127.0.0.1:8000 \
  --num-samples 4 \
  --strategy best_of_n
