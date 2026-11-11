# Scientist Constraints

## Data
Use only the scene inputs in `data/scenes/` as provided (primitive geometry, semantic descriptors, training-view pixels with blend weights and per-image mask ids). Do not edit those files to change what is scored. The evaluator does not read them, so edits only change what your method sees. Do not add external videos, labels, or extra generated scenes as supervision.

## Answers
Development-scene answers in `eval/train/truth/` are for evaluation and model selection only. Do not use them as training targets for a grouping model, and do not hard-code anything derived from them. Test-scene answers are hidden.

## Evaluation Artifacts
Submit per-primitive integer `pred_labels` for every development and test scene, in `outputs/**/*.json`, following `briefing/evaluation.md`. Do not modify `eval/`, the truth files, or the fixed reference artifact. The scoring rules and the constraint are part of the benchmark.

## Resource Scale
Keep the given primitives, frames, and views. Do not re-render or re-segment the scenes, or derive new observations from hidden test views.

## Supervision
Methods may use only the public observations: inconsistent mask ids, blend weights, primitive geometry and motion, and the semantic descriptors. Video-tracker-style labels produced by external models are out of scope.

## Compliance Checklist
- [ ] Predictions cover every development and test scene with exactly one integer label per primitive
- [ ] No development or test answers used for training or as hard-coded values
- [ ] `eval/`, truth files, and `fixed_baseline.json` unchanged
- [ ] Scene inputs in `data/scenes/` unchanged
- [ ] Interaction-experiment F1 and Recalldyn beat the fixed reference artifact (checked by the evaluators)
