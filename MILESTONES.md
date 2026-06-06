# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-06 00:00] - method_runs
- Implemented the paper's feature-stage Split4D algorithm at reduced scale with fixed linear-motion Gaussian primitives, rendered feature maps, temporally inconsistent per-image masks, contrastive mask learning, streaming sampling, tracking regularization, semantic alignment, and clustering/filtering inference.
- Submitted container jobs that ran the method and random-order ablation end-to-end and wrote method output artifacts consumed by `scripts/evaluate.sh`.

### [2026-06-06 00:00] - core_claim
- Job `82b7b084-80f` reproduced the central same-benchmark claim on deterministic reduced scenes: temporally ordered Split4D feature learning beats the random-order ablation on the interaction-motion setting while maintaining stronger dynamic-object recall.
- Accepted scores are recorded under `scaled_linear_motion_segmentation` and `scaled_streaming_ablation`; full-scale paper Table 1/Table 2 values are preserved separately in `scoring/reference.json` as provenance-only `paper_fullscale_*` groups.

### [2026-06-06 17:59] - core_claim
- Continuation audit after the reset re-read `method/`, `eval/`, `data/`, scripts, briefing, scoring docs, and the current score files.
- `python validate.py --compare` passed; the implementation remains a real scaled Split4D feature-stage reproduction and the evaluator reads method artifacts without importing `method/`.
- No new compute job is submitted because the accepted configuration already supports `core_claim`, while the documented longer run improved one score but broke the streaming-ablation constraint.

## Stop Justification
- Completed at milestone `core_claim`.
