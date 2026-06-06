#!/bin/bash
# Visible train-slice evaluator. Mirrors scripts/evaluate.sh's artifact-scoring
# contract while writing scoring/scores_train.json instead of scoring/scores.json.

set -e

cd /home/user

mkdir -p scoring

python -m eval.train.evaluate "$@"
