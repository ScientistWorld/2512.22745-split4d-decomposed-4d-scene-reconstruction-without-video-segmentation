# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-06 17:55] - core_claim
- Continuation audit confirmed the existing implementation is a scaled Split4D reproduction rather than a surrogate: fixed dynamic Gaussian scenes, rendered per-primitive features, mask-level contrastive learning, streaming temporal sampling, 3D tracking regularization, semantic alignment, and clustering/filtering inference are present in `method/`.
- Re-ran `python validate.py --compare`; validation passed, including scoring structure, import separation, evaluator output-read checks, train/test split checks, and method discrimination on the primary metric.
- Retained `core_claim` as the highest trustworthy milestone. The accepted job `82b7b084-80f` supports the core claim; a later longer exploratory run did not improve the streaming-ablation constraint reliably, so no higher milestone is claimed.

### [2026-06-06 17:53] - core_claim
- Implemented the paper's actual Split4D feature-stage computation at reduced scale: fixed linearly moving Gaussian primitives, rendered per-primitive feature maps, mask-level contrastive feature learning, streaming temporal sampling, 3D tracking regularization, semantic alignment, and clustering/filtering inference.
- Submitted and completed the accepted container job `82b7b084-80f`; it regenerated method and random-order ablation artifacts, wrote `scoring/scores.json`, and passed `python validate.py --compare`.
- On the deterministic scaled benchmark, streaming Split4D reached mIoU 0.365 on linear motion and beat random temporal sampling on the harder interaction-motion ablation (mIoU 0.373 vs. 0.249, dynamic recall 0.611 vs. 0.317).
- The reproduction uses reduced procedural dynamic Gaussian scenes because the paper's full Neural3DV, Multi-Human, and SelfCap reconstruction pipeline is not packaged as a ready benchmark here; the algorithmic procedure itself is preserved rather than replaced.
