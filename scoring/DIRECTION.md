# Research Direction

## Research Problem
Decomposed 4D instance segmentation of dynamic scenes from per-image automatic masks whose labels are independent across views and frames and contain the segmenter's errors. The central question is how to produce object identities that stay coherent over a whole sequence, including through contacts between objects, without a video tracker supplying consistent labels.

## Research Focus
Work should stay within instance grouping of the fixed 4D scene primitives. Improvements should address how spatial, temporal, observation-level, and semantic evidence is combined to assign stable identities. Moving and interacting objects are the hard part; static objects are comparatively easy.

## Approach Scope
Valid changes include how the observations are used, how evidence propagates through time, representation learning on the primitives, grouping and inference procedures, robustness to segmenter merges and splits, and refinement of assignments. The output must remain one instance label per given primitive.

## Out of Bounds
- Training on development or test answers, or hard-coding them.
- Using external trackers or segmenters that supply temporally consistent labels.
- Editing the evaluator, truth files, fixed reference artifact, or scene inputs.
- Changing the number of primitives, frames, or views, or re-rendering scenes.
