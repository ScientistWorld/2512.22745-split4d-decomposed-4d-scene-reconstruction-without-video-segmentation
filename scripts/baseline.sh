#!/bin/bash
# Run the paper's "w/o streaming learning" ablation: the same Split4D feature
# objective, representation, iterations, and inference, but observations are
# sampled uniformly at random instead of in temporal order.
#
# Usage:
#   bash scripts/baseline.sh

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/baseline
python -m baseline.random_order \
  --output outputs/baseline/random_order.json \
  --method-name split4d_random_order \
  --steps "${SPLIT4D_STEPS:-10000}" \
  --seed "${SPLIT4D_SEED:-0}" \
  --no-streaming
