#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -e

cd /home/user

export SPLIT4D_BASELINE_STEPS="${SPLIT4D_BASELINE_STEPS:-3200}"
export SPLIT4D_STEPS="${SPLIT4D_STEPS:-3200}"
export SPLIT4D_SAMPLES_PER_MASK="${SPLIT4D_SAMPLES_PER_MASK:-32}"
export SPLIT4D_LAMBDA_ALIGN="${SPLIT4D_LAMBDA_ALIGN:-1.0}"
export SPLIT4D_LAMBDA_DINO="${SPLIT4D_LAMBDA_DINO:-0.08}"

bash scripts/download.sh
rm -f outputs/split4d/results.json outputs/baseline/random_order.json scoring/scores.json tmp/exploratory_scores.json
bash scripts/baseline.sh
bash scripts/method.sh
bash scripts/evaluate.sh
