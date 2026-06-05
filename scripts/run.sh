#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

export SPLIT4D_BASELINE_STEPS="${SPLIT4D_BASELINE_STEPS:-1400}"
export SPLIT4D_STEPS="${SPLIT4D_STEPS:-2200}"

bash scripts/reproduce.sh
python validate.py --compare
