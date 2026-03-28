#!/usr/bin/env bash
# Phase 2: Student Training
set -e
cd "$(dirname "$0")/.."
python -m pipelines.phase2_student_train \
  --model-config config/model_config.yaml \
  --train-config config/training_config.yaml
