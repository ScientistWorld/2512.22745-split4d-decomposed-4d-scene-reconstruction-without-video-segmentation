# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-05 05:56] - method_runs
- Implemented the Split4D feature-stage algorithm on fixed dynamic Gaussian-style scenes.
- Submitted a container job that ran the method and random-order ablation end-to-end.

### [2026-06-05 06:01] - method_runs
- A milestone audit found that the previous scores used method-side oracle cluster-id alignment from evaluation labels.
- Removed evaluation labels from method outputs and moved generic Hungarian cluster matching into `eval/metrics.py`.
- Updated inference to use the paper's HDBSCAN clustering step when available, with deterministic KMeans only as a small-sample fallback.
- Removed hidden-label use from 3D tracking regularization; adjacent-frame primitive pairs now use spatial nearest-neighbor matching, matching the paper's regularizer.

### [2026-06-05 06:15] - method_runs
- Job `6a29ddef-acf` completed corrected non-oracle validation and passed `python validate.py --compare`.
- Corrected linear-motion score: mIoU 0.694, F1 0.819, dynamic recall 0.833.
- Kept milestone at `method_runs` because mIoU is slightly below the strongest baseline in the accepted target.

### [2026-06-06 16:39] - core_claim
- Job `030a094a-79e` completed the fixed non-oracle Split4D run and passed `python validate.py --compare`.
- Accepted linear-motion score improved to mIoU 0.709, F1 0.824, and dynamic recall 0.833.
- This clears the strongest paper baseline mIoU target in the accepted scaled benchmark while preserving dynamic-object recall.

## Stop Justification
- Completed at milestone `core_claim`.
