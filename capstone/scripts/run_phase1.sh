#!/usr/bin/env bash
# Phase 1: Offline Teacher Distillation
set -e
cd "$(dirname "$0")/.."
python -m pipelines.phase1_teacher_distill \
  --model-config config/model_config.yaml \
  --train-config config/training_config.yaml
