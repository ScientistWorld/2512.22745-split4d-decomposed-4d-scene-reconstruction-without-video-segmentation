# Evaluation Targets

## Primary: scaled_linear_motion_segmentation
Tests whether a method can recover temporally consistent instance labels on a dynamic multi-view scene with moderate linear motion and inconsistent per-image mask ids.
- **Metric**: `mIoU` (higher is better)
- **Constraint metric**: `Recalldyn` (higher is better; the method should maintain strong dynamic-object recovery rather than improving static objects only)
- **Secondary metrics**: `mAcc`, `Recall`, `F1` (higher is better)
- **How to run**: write method outputs in the JSON format used by `outputs/split4d/results.json`, then run `bash scripts/evaluate.sh`.

## Constraint/Ablation: scaled_streaming_ablation
Tests whether temporally ordered training improves over the same feature learner run with random temporal sampling on the harder interaction-motion scene.
- **Metric**: `mIoU` gap between `split4d_streaming` and `split4d_random_order` (higher streaming score is better)
- **Constraint metric**: `Recalldyn` (streaming should preserve moving-object recall rather than improving only static regions)
- **Secondary metrics**: `mAcc`, `Recall`, `F1`
- **How to run**: `bash scripts/reproduce.sh`; `scripts/evaluate.sh` reads both method and random-order output artifacts.

## Paper Provenance
Full-scale paper values are retained in `reference.json` under `paper_fullscale_*` keys. Accepted scores use `scaled_*` keys because this gym evaluates a deterministic reduced synthetic benchmark, not the paper's unreleased full-scale reconstruction/data pipeline.
