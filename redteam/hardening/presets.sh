#!/bin/bash
# Diagnostic only: how the reference, the random-order ablation and the mask
# co-membership graph respond to two alternative scene presets (dev scenes only).
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
mkdir -p redteam/hardening/presets
for P in clean_masks long_lived; do
  D=tmp/presets/$P
  [ -f "$D/scenes/manifest.json" ] || python -m eval.test.scene_builder --preset "$P" --only-split dev \
      --public-dir "$D/scenes" --train-truth-dir "$D/truth" --test-truth-dir "$D/unused" || continue
  [ -f "$D/outputs/split4d/results.json" ] || python -m method.split4d --scene-dir "$D/scenes" \
      --output "$D/outputs/split4d/results.json" --method-name split4d_streaming
  [ -f "$D/outputs/baseline/random_order.json" ] || python -m method.split4d --scene-dir "$D/scenes" \
      --output "$D/outputs/baseline/random_order.json" --method-name split4d_random_order --no-streaming
  [ -f "$D/outputs/graph/results.json" ] || python -m baseline.simple_grouping mask_graph_spectral \
      --scene-dir "$D/scenes" --output "$D/outputs/graph/results.json"
  python -m eval.train.evaluate --truth-dir "$D/truth" --fixed-baseline "$D/fixed_baseline.json" \
      --freeze-baseline "$D/outputs/baseline/random_order.json"
  python -m eval.train.evaluate --truth-dir "$D/truth" --fixed-baseline "$D/fixed_baseline.json" \
      --outputs "$D/outputs" --output "redteam/hardening/presets/$P.json" > /dev/null
  echo "preset $P done"
done
