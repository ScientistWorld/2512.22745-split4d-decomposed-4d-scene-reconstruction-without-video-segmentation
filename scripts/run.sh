#!/bin/bash
# CPU-only packaging check: regenerate canonical scores from frozen method
# artifacts, then verify the core-claim gate and train/test split scores.

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python -m eval.check_core_claim --score-files scoring/scores.json scoring/scores_train.json scoring/scores_test.json
python validate.py --post-design
