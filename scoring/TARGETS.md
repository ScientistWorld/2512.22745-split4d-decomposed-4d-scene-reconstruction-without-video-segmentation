# Evaluation Targets

## Primary: linear_motion_segmentation
Tests whether a method can recover temporally consistent instance labels on a dynamic multi-view scene with moderate linear motion and inconsistent per-image mask ids.
- **Metric**: `mIoU` (higher is better)
- **Secondary metrics**: `mAcc`, `Recall`, `F1`, `Recalldyn` (higher is better)
- **How to run**: write method outputs in the JSON format used by `outputs/split4d/results.json`, then run `bash scripts/evaluate.sh`.

## Primary: interaction_motion_segmentation
Tests whether the same task remains successful when objects have more interaction-like motion and semantic ambiguity.
- **Metric**: `mIoU` (higher is better)
- **Secondary metrics**: `mAcc`, `Recall`, `F1`, `Recalldyn` (higher is better)
- **How to run**: write method outputs in the JSON format used by `outputs/split4d/results.json`, then run `bash scripts/evaluate.sh`.

## Ablation Constraint: streaming_ablation
Tests whether temporally coherent learning improves dynamic-instance segmentation relative to the same feature-learning task with randomized temporal order.
- **Metric**: `mIoU` (higher is better)
- **Constraint metric**: `Recalldyn` (higher is better; low dynamic recall means the central temporal consistency claim is not supported)
- **How to run**: run both `bash scripts/baseline.sh` and `bash scripts/method.sh`, then inspect `scoring/scores.json`.
