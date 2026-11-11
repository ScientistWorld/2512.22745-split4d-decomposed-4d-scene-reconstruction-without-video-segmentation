#!/bin/bash
# Run the paper's method (Split4D streaming feature learning) on every public
# scene (development and test) and score the development scenes.
#
# Paper settings: 10,000 iterations per scene, 32-d features, Adam with the
# learning rate decayed from 2.5e-3 to 2e-6, lambda_1 = 100, lambda_2 = 500
# (10 for the multi-person interaction profile), DINO alignment only early,
# first-frame HDBSCAN + dynamic-perception filtering at inference.
#
# Usage:
#   bash scripts/method.sh

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/split4d
python -m method.split4d \
  --output outputs/split4d/results.json \
  --method-name split4d_streaming \
  --steps "${SPLIT4D_STEPS:-10000}" \
  --seed "${SPLIT4D_SEED:-0}" \
  --streaming

bash scripts/evaluate.sh
