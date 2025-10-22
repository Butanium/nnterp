#!/bin/bash

models=(
  "yujiepan/qwen1.5-moe-tiny-random"
  "yujiepan/gpt-oss-tiny-random"
  "yujiepan/dbrx-tiny-random"
  "yujiepan/stablelm-2-tiny-random"
)

for model in "${models[@]}"; do
  echo ""
  echo "████████████████████████████████████████████████████████████████"
  echo "Testing: $model"
  echo "████████████████████████████████████████████████████████████████"
  uv run python find_eager_attention.py "$model" 2>&1 | head -150
  echo ""
done
