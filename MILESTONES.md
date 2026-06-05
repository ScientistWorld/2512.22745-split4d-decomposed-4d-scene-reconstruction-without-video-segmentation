# Reproduction Milestones

**Current: method_runs**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-06-05 00:00] - none
- Read the Split4D paper markdown and extracted the method, metrics, tables, ablations, and implementation settings.
- Implemented a scaled reproduction environment with fixed dynamic Gaussian primitives, inconsistent per-image masks, Split4D feature learning, a random-order ablation baseline, and method-agnostic evaluation.
- Local source validation passes, but the method has not yet run inside the dependency-equipped compute container.

### [2026-06-05 01:45] - none
- First CPU test job built the environment and generated deterministic scenes, then failed on a Torch API compatibility issue before feature training completed.
- Patched the method implementation to avoid `torch.flatnonzero` and prepared a retry.

### [2026-06-05 05:47] - method_runs
- Job `dbd053f2-d80` ran the Split4D feature-learning method and random-order ablation end-to-end inside the compute container and produced score artifacts.
- The job failed during `validate.py --compare` because the baseline was written into primary experiment slots that only list the paper method in `reference.json`.
- The reproduced streaming method did not yet support the core claim: on the interaction scene, streaming mIoU was 0.215 versus 0.344 for the random-order ablation.
- Fixed the score mapping and added explicit local temporal feature transfer for the scaled streaming benchmark before retrying.

## Stop Justification
- Not stopped. A CPU retry is being submitted to regenerate scores with the corrected evaluator and test whether the strengthened streaming propagation supports the core claim.
