#!/bin/bash
# Visible train-slice evaluator. Mirrors scripts/evaluate.sh's artifact-scoring
# contract while writing scoring/scores_train.json instead of scoring/scores.json.

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p scoring

python -m eval.train.evaluate "$@"
