# Evaluation Targets

## Primary: scaled_dynamic_scene_segmentation
Recover temporally consistent instance labels for every scene primitive in kitchen-like dynamic scenes from inconsistent per-image masks.
- **Metric**: `mIoU` (higher is better), measured in hidden test views over all frames with one prediction-to-instance matching per scene.
- **Constraint metric**: `Recalldyn` (higher is better). Moving and handled objects must stay correct; gains on static furniture alone do not count for much.
- **Secondary metrics**: `F1`, `Recall`, `Recallstatic`, `mAcc` (higher is better).
- **Reference (hidden test)**: mIoU 0.791, F1 0.853, Recalldyn 0.819.
- **How to run**: write `outputs/<name>/*.json` with `pred_labels` for every scene, then `bash scripts/evaluate.sh` (development scenes).

## Primary + constraint: scaled_interaction_segmentation
Preserve identities of interacting, similar-looking people and a passed ball.
- **Metric**: `mIoU` (higher is better).
- **Constraint**: `F1` and `Recalldyn` must exceed those of the fixed reference artifact shipped with the evaluator (hidden test: F1 0.588, Recalldyn 0.399). Otherwise the whole run is scored 0 in every metric of every experiment.
- **Secondary metrics**: `F1`, `Recall`, `Recalldyn`, `Recallstatic`, `mAcc`.
- **Reference (hidden test)**: mIoU 0.637, F1 0.744, Recalldyn 0.583.
- **How to run**: same output files. `bash scripts/evaluate_train.sh` reports the development-scene version of the constraint.

## Validity
Missing scenes, wrong label counts, non-integer or out-of-range labels, or more than 512 labels in a scene fail the run (0 in every metric).

## Paper Provenance
The paper's full-scale values (Neural3DV, Multi-Human, SelfCap; Juggle ablation) are under `_paper_reported` in `reference.json`. They are not on this benchmark's scale and are not targets.
