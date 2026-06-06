#!/bin/bash
# Score the visible train and held-out test slices from the frozen method
# artifacts under outputs/.

set -e

cd /home/user

bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
