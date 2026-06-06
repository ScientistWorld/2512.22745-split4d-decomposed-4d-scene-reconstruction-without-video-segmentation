# Evaluation

This file defines how candidate outputs are scored. It describes the target metrics and output contract only; method design choices belong outside the problem definition.

## Metrics

Evaluation uses instance segmentation metrics over annotated objects:

- `mIoU`: mean intersection-over-union between predicted and target masks or primitive sets, averaged over instances. Higher is better.
- `mAcc`: mean one-vs-rest accuracy per instance. Higher is better.
- `Recall`: mean foreground recall per instance. Higher is better.
- `F1`: mean per-instance harmonic mean of precision and recall. Higher is better.
- `Recalldyn`: recall restricted to dynamic instances. Higher is better and is the key constraint against methods that only handle static or slow objects.
- `Recallstatic`: recall restricted to static instances. Higher is better.
- `ClusterSizeError`: mean relative error in predicted instance sizes. Lower is better.
- `MaskFeatureAgreement`: separation margin between candidate-provided element representations for elements that agree or disagree with observed masks. Higher is better.
- `TrackingFeatureMargin`: representation consistency margin for geometrically matched elements across adjacent frames. Higher is better.

## Baselines and Targets

The paper reports comparisons against OmniSeg3D, DGD, SA4D, and SADG on Neural3DV, Multi-Human, and SelfCap. Its reported headline mIoU values are 0.801 on Neural3DV, 0.893 on Multi-Human, and 0.882 on SelfCap. This gym keeps those full-scale numbers in `scoring/reference.json` for provenance, but the accepted scored experiments use deterministic reduced scenes so the environment is reproducible within the available budget.

## Evaluation Protocol

Accepted runs evaluate on deterministic dynamic scenes with fixed geometry, fixed observation masks, and hidden instance labels. Methods write primitive-level predicted instance labels and per-primitive representation vectors to JSON. `scripts/evaluate.sh` reads those artifacts and computes metrics without importing method code. Scores are written to `scoring/scores.json`.
