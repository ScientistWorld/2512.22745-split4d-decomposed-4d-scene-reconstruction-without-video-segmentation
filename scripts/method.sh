#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

mkdir -p outputs/split4d
python -m method.split4d \
  --output outputs/split4d/results.json \
  --scenes linear_motion interaction_motion \
  --steps "${SPLIT4D_STEPS:-1400}" \
  --feature-dim 32 \
  --samples-per-mask "${SPLIT4D_SAMPLES_PER_MASK:-32}" \
  --lr 2.5e-3 \
  --lambda-align "${SPLIT4D_LAMBDA_ALIGN:-1.0}" \
  --lambda-dino "${SPLIT4D_LAMBDA_DINO:-0.08}" \
  --dino-fraction 0.35 \
  --streaming

bash scripts/evaluate.sh
