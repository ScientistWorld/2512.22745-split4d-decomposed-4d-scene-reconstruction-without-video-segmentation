#!/bin/bash
# Visible train-split evaluator: scores every method output under outputs/ on
# the development scenes (answers in eval/train/truth) and writes
# scoring/scores_train.json. Runs with no arguments.

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p scoring
python -m eval.train.evaluate "$@"
