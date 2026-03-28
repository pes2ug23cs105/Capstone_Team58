#!/usr/bin/env bash
# Phase 3: Evaluation
set -e
cd "$(dirname "$0")/.."
python -m pipelines.phase3_evaluate \
  --model-config  config/model_config.yaml \
  --eval-config   config/eval_config.yaml \
  --adapter-path  outputs/checkpoints/final_adapter
