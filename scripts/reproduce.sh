#!/bin/bash
# Reproduce the reference results end to end (runs inside the container):
# the random-order ablation baseline, the paper's streaming method, and the
# visible dev-scene evaluation. The hidden test evaluation is run separately
# with `bash scripts/evaluate_test.sh` where eval/test/ is available.

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
bash scripts/download.sh
rm -f outputs/split4d/results.json outputs/baseline/random_order.json
bash scripts/baseline.sh
bash scripts/method.sh
bash scripts/evaluate_train.sh
if [ -d eval/test/truth ]; then
  bash scripts/evaluate_test.sh
fi
