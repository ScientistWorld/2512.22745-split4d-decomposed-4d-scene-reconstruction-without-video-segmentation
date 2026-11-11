#!/bin/bash
# Hidden test-split evaluator: scores every method output under outputs/ on
# the held-out test scenes (answers only in eval/test/truth) and writes
# scoring/scores_test.json. Runs with no arguments and never retrains.

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p scoring
python -m eval.test.evaluate "$@"
