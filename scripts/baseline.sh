#!/bin/bash
# Run a baseline method from the paper.
#
# Baselines are simple reference methods the paper compares against. 
# Implement this only if applicable, i.e. the paper has baselines.
# You do not need to implement all baselines to begin — 
# just enough to validate that the evaluation pipeline works 
# and provide a reference point. You can decide to expand more baselines later.
# Each baseline must be actually run and scored through evaluate.sh
# so the numbers are verified, not self-reported.
# You can decide what this baseline script should do yourself.
#
# Usage:
#   bash scripts/baseline.sh <name>

set -e

cd /home/user

mkdir -p outputs/baseline
python -m method.split4d \
  --output outputs/baseline/random_order.json \
  --method-name split4d_random_order \
  --scenes linear_motion interaction_motion \
  --steps "${SPLIT4D_BASELINE_STEPS:-1400}" \
  --feature-dim 32 \
  --samples-per-mask "${SPLIT4D_SAMPLES_PER_MASK:-32}" \
  --lr 2.5e-3 \
  --lambda-align "${SPLIT4D_LAMBDA_ALIGN:-1.0}" \
  --lambda-dino "${SPLIT4D_LAMBDA_DINO:-0.03}" \
  --lambda-stream "${SPLIT4D_LAMBDA_STREAM:-0.25}" \
  --dino-fraction 0.35 \
  --streaming-window "${SPLIT4D_STREAMING_WINDOW:-3}" \
  --no-streaming

bash scripts/evaluate.sh --inputs outputs/baseline/random_order.json --output tmp/baseline_scores.json
