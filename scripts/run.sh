#!/bin/bash
# Score the visible train and held-out test slices from the frozen method
# artifacts under outputs/.

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
