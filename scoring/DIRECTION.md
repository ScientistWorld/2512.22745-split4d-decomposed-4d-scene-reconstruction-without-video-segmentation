# Research Direction

## Research Problem
The benchmark concerns decomposed 4D instance segmentation from dynamic scene observations when 2D mask labels are independent per image. The central question is how to produce temporally coherent object identities without relying on a precomputed video tracker or globally consistent annotation labels.

## Core Contribution
Work should stay within feature learning for fixed dynamic Gaussian-like scene primitives. Improvements should address how primitive features are learned, regularized, propagated, clustered, or filtered across space and time under inconsistent mask supervision.

## Approach Scope
Valid changes include new feature losses, temporal curricula, primitive matching rules, semantic regularization schedules, clustering methods, or filtering criteria. The method should still solve the same 4D feature-field decomposition problem using the fixed primitive representation and weak per-image masks.

## Out of Bounds
- Replacing the task with supervised classification from ground-truth object ids.
- Using an external tracker that supplies temporally consistent labels as training targets.
- Switching to a dense video segmentation model that ignores the fixed 4D primitive representation.
- Increasing scene resolution, feature dimension, primitive count, or number of observations for scored experiments.
