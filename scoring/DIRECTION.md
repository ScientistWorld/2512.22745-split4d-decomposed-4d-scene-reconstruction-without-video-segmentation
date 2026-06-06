# Research Direction

## Research Problem
The benchmark concerns decomposed 4D instance segmentation from dynamic scene observations when 2D mask labels are independent per image. The central question is how to produce temporally coherent object identities without relying on a precomputed video tracker or globally consistent annotation labels.

## Research Focus
Work should stay within decomposed 4D instance grouping for fixed dynamic scene elements. Improvements should address how spatial, temporal, and observation-level evidence is combined to assign stable object identities under inconsistent mask supervision.

## Approach Scope
Valid changes include alternative uses of the provided observations, temporal and spatial consistency cues, grouping or refinement strategies, and representation choices that keep the fixed scene elements and output contract. The method should still solve the same 4D element-level decomposition problem using the benchmark observations and weak per-image masks.

## Out of Bounds
- Replacing the task with supervised classification from ground-truth object ids.
- Using an external tracker that supplies temporally consistent labels as training targets.
- Switching to a dense video segmentation model that ignores the fixed 4D primitive representation.
- Increasing scene resolution, feature dimension, primitive count, or number of observations for scored experiments.
