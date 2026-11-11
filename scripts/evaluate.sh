#!/bin/bash
# Standard evaluation entry point (visible dev scenes).
#
# Scores every method output under outputs/ (*.json with a "scenes" object of
# per-primitive "pred_labels") on the development scenes, using only code and
# answers in eval/train/, and writes scoring/scores.json:
#   {"experiments": {"scaled_dynamic_scene_segmentation": {"results": {"<method>": {"mIoU": ...}}}, ...}}
# It never imports method/. Extra arguments are passed to eval.train.evaluate.
#
# Usage:
#   bash scripts/evaluate.sh
#   bash scripts/evaluate.sh --outputs outputs

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p scoring
python -m eval.train.evaluate --output scoring/scores.json "$@"
