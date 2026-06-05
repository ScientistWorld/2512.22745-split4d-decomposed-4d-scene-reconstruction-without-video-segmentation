# Evaluation

How success is measured. Describe only the metrics, targets, and evaluation procedure — do NOT describe the paper's method or how they achieved these results. Think of this as a general evaluation protocal for other scientists trying to solve the same problem, where they should not see this paper's solution.

## Metrics

Evaluation uses instance segmentation metrics over annotated objects:

- `mIoU`: mean intersection-over-union between predicted and target masks or primitive sets, averaged over instances. Higher is better.
- `mAcc`: mean one-vs-rest accuracy per instance. Higher is better.
- `Recall`: mean foreground recall per instance. Higher is better.
- `F1`: mean per-instance harmonic mean of precision and recall. Higher is better.
- `Recalldyn`: recall restricted to dynamic instances. Higher is better and is the key constraint against methods that only handle static or slow objects.

## Baselines and Targets

The paper reports comparisons against OmniSeg3D, DGD, SA4D, and SADG on Neural3DV, Multi-Human, and SelfCap. Its reported headline mIoU values are 0.801 on Neural3DV, 0.893 on Multi-Human, and 0.882 on SelfCap. The most important ablation is replacing temporally ordered training with random sampling, which drops mIoU from 0.895 to 0.552 on the Multi-Human Juggle sequence and strongly reduces dynamic recall.

## Evaluation Protocol

This reproduction evaluates on deterministic scaled dynamic scenes with fixed 4D primitive geometry and fixed per-image inconsistent supervision. Methods write predicted primitive-level instance labels to JSON. `scripts/evaluate.sh` reads those artifacts and computes metrics without importing method code. Scores are written to `scoring/scores.json`.
