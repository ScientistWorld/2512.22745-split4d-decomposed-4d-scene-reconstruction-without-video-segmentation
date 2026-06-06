# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-06 17:53] - core_claim
- Implemented the paper's actual Split4D feature-stage computation at reduced scale: fixed linearly moving Gaussian primitives, rendered per-primitive feature maps, mask-level contrastive feature learning, streaming temporal sampling, 3D tracking regularization, semantic alignment, and clustering/filtering inference.
- Submitted and completed the accepted container job `82b7b084-80f`; it regenerated method and random-order ablation artifacts, wrote `scoring/scores.json`, and passed `python validate.py --compare`.
- On the deterministic scaled benchmark, streaming Split4D reached mIoU 0.365 on linear motion and beat random temporal sampling on the harder interaction-motion ablation (mIoU 0.373 vs. 0.249, dynamic recall 0.611 vs. 0.317).
- The reproduction uses reduced procedural dynamic Gaussian scenes because the paper's full Neural3DV, Multi-Human, and SelfCap reconstruction pipeline is not packaged as a ready benchmark here; the algorithmic procedure itself is preserved rather than replaced.
