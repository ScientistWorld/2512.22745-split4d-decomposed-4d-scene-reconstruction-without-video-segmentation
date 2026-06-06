#!/bin/bash
# Held-out test-slice evaluator. It mirrors the method keys in
# scoring/scores_train.json and writes scoring/scores_test.json.

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p scoring

python -m eval.test.evaluate "$@"
