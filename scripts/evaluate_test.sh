#!/bin/bash
# Held-out test-slice evaluator. It mirrors the method keys in
# scoring/scores_train.json and writes scoring/scores_test.json.

set -e

cd /home/user

mkdir -p scoring

python -m eval.test.evaluate "$@"
