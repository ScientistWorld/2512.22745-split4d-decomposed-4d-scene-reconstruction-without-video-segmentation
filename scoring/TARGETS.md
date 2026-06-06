# Evaluation Targets

## Primary: scaled_linear_motion_segmentation
Tests whether a method can recover temporally consistent instance labels on a dynamic multi-view scene with moderate linear motion and inconsistent per-image mask ids.
- **Metric**: `mIoU` (higher is better)
- **Constraint metric**: `Recalldyn` (higher is better; the method should maintain strong dynamic-object recovery rather than improving static objects only)
- **Secondary metrics**: `mAcc`, `Recall`, `F1`, `Recallstatic`, `MaskFeatureAgreement`, and `TrackingFeatureMargin` (higher is better), plus `ClusterSizeError` (lower is better)
- **How to run**: write method outputs in the JSON format consumed by `eval/metrics.py`, then run `bash scripts/evaluate.sh --inputs outputs/<method>/results.json`.

## Constraint/Ablation: scaled_streaming_ablation
Tests whether a candidate method maintains coherent object identities on the harder interaction-motion scene relative to the included reference artifact.
- **Metric**: `F1` gap between the candidate run and the reference artifact (higher candidate score is better)
- **Constraint metric**: `Recalldyn` (the candidate should preserve moving-object recall rather than improving only static regions)
- **Secondary metrics**: `mAcc`, `Recall`, `F1`, `Recallstatic`, `MaskFeatureAgreement`, and `TrackingFeatureMargin` (higher is better), plus `ClusterSizeError` (lower is better)
- **How to run**: `bash scripts/reproduce.sh`; `scripts/evaluate.sh` reads the candidate and reference output artifacts.

## Paper Provenance
Full-scale paper values are retained in `reference.json` under `paper_fullscale_*` keys. Accepted scores use `scaled_*` keys because this gym evaluates a deterministic reduced synthetic benchmark, not the paper's unreleased full-scale reconstruction/data pipeline.
