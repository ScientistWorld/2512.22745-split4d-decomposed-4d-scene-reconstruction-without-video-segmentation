#!/bin/bash
# Score the existing final method artifacts on the visible train slice and the
# held-out test slice. Method discovery happens inside the evaluator scripts.

set -e

cd /home/user

bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
