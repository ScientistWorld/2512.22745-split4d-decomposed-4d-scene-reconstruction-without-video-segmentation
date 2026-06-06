# Research Direction

## Research Problem
The benchmark concerns decomposed 4D instance segmentation from dynamic scene observations when 2D mask labels are independent per image. The central question is how to produce temporally coherent object identities without relying on a precomputed video tracker or globally consistent annotation labels.

## Core Contribution
Work should stay within feature learning for fixed dynamic scene primitives. Improvements should address how primitive features are estimated and grouped across space and time under inconsistent mask supervision.

## Approach Scope
Valid changes include new feature objectives, observation-ordering policies, primitive matching rules, semantic consistency terms, clustering methods, or filtering criteria. The method should still solve the same 4D primitive-level decomposition problem using the fixed representation and weak per-image masks.

## Out of Bounds
- Replacing the task with supervised classification from ground-truth object ids.
- Using an external tracker that supplies temporally consistent labels as training targets.
- Switching to a dense video segmentation model that ignores the fixed 4D primitive representation.
- Increasing scene resolution, feature dimension, primitive count, or number of observations for scored experiments.
