# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-06 00:00] - method_runs
- Implemented the paper's feature-stage Split4D algorithm at reduced scale with fixed linear-motion Gaussian primitives, rendered feature maps, temporally inconsistent per-image masks, contrastive mask learning, streaming sampling, tracking regularization, semantic alignment, and clustering/filtering inference.
- Submitted container jobs that ran the method and random-order ablation end-to-end and wrote method output artifacts consumed by `scripts/evaluate.sh`.

### [2026-06-06 00:00] - core_claim
- Job `82b7b084-80f` reproduced the central same-benchmark claim on deterministic reduced scenes: temporally ordered Split4D feature learning beats the random-order ablation on the interaction-motion setting while maintaining stronger dynamic-object recall.
- Accepted scores are recorded under `scaled_linear_motion_segmentation` and `scaled_streaming_ablation`; full-scale paper Table 1/Table 2 values are preserved separately in `scoring/reference.json` as provenance-only `paper_fullscale_*` groups.

## Stop Justification
- Completed at milestone `core_claim`.
- Further work toward `core_claim_plus` would require finding a new configuration that improves an additional setting without breaking the streaming-ablation constraint; the current packaged gym is validated and usable.
