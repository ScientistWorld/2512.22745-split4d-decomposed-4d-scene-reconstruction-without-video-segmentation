# Evaluation Targets

## Primary: linear_motion_segmentation
Tests whether a method can recover temporally consistent instance labels on a dynamic multi-view scene with moderate linear motion and inconsistent per-image mask ids.
- **Metric**: `mIoU` (higher is better)
- **Constraint metric**: `Recalldyn` (higher is better; the method should maintain strong dynamic-object recovery rather than improving static objects only)
- **Secondary metrics**: `mAcc`, `Recall`, `F1` (higher is better)
- **How to run**: write method outputs in the JSON format used by `outputs/split4d/results.json`, then run `bash scripts/evaluate.sh`.
