#!/bin/bash
# Regenerate the accepted scaled Split4D artifacts, validate the standard
# scores, then score the visible train and held-out test slices.

set -e

cd /home/user

export SPLIT4D_BASELINE_STEPS="${SPLIT4D_BASELINE_STEPS:-3200}"
export SPLIT4D_STEPS="${SPLIT4D_STEPS:-3200}"
export SPLIT4D_SAMPLES_PER_MASK="${SPLIT4D_SAMPLES_PER_MASK:-32}"
export SPLIT4D_LAMBDA_ALIGN="${SPLIT4D_LAMBDA_ALIGN:-1.0}"
export SPLIT4D_LAMBDA_DINO="${SPLIT4D_LAMBDA_DINO:-0.08}"

bash scripts/reproduce.sh
python -m eval.metrics --include-exploratory --output tmp/exploratory_scores.json
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python validate.py --post-design
python -m eval.check_core_claim
