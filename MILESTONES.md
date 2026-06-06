# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-06 16:44] - none
- Started from an existing partial Split4D reproduction workspace.
- Audited the paper, implementation, scoring setup, and prior job notes.

### [2026-06-06 16:44] - method_runs
- Confirmed the scaled Split4D feature-stage implementation runs end-to-end in prior compute jobs.
- Verified the method implements fixed linear-motion Gaussian primitives with learnable features, contrastive mask learning, streaming temporal sampling, tracking alignment, semantic regularization, and clustering/filtering inference.

### [2026-06-06 16:44] - core_claim
- Validated `scoring/scores.json` with `python validate.py --compare`.
- Accepted target `linear_motion_segmentation` reports Split4D streaming mIoU 0.709, clearing the strongest listed baseline mIoU 0.700, with dynamic recall 0.833 matching the paper's proposed target on the scaled benchmark.
- Kept harder interaction and streaming-ablation results documented as exploratory because they are underpowered relative to the paper's reported interaction setting.
