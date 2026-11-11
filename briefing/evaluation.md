# Evaluation

This file defines how candidate outputs are scored. It describes the measurement and the output contract only.

## Scenes and Splits

There are two experiments, each with its own family of procedural scenes (60 frames, 6 training views):

- `scaled_dynamic_scene_segmentation`: kitchen-like scenes with one person handling objects, a pet, and static furniture (8 instances, 4 of them moving).
- `scaled_interaction_segmentation`: four people who meet and touch, a ball passed between them, and two static objects (7 instances, 5 moving).

Every scene's inputs are public in `data/scenes/` (see `data/split4d_scenes.py`). Development scenes (`*_dev_*`, 4 per experiment) have answers in `eval/train/truth`, and `scripts/evaluate_train.sh` / `scripts/evaluate.sh` score them. Test scenes (`*_test_*`, 10 per experiment) are scored by the hidden test evaluator. Their answers and their held-out test views are not public.

## Output Contract

Write one or more JSON files anywhere under `outputs/`:

```json
{"method": "<name>", "scenes": {"<scene_id>": {"pred_labels": [<int>, ...]}, ...}}
```

- `pred_labels` has exactly one integer per primitive, in the scene file's primitive order. `-1` means unassigned, and any other non-negative integer is an instance label. Label values are arbitrary and are matched to instances by the evaluator.
- A method must cover every development and test scene. Files with the same `method` name are merged.
- Nothing else in the file is read. Scores or flags written by a method are ignored.

## Metrics

Each scene is scored in held-out views over all frames. A pixel's predicted instance is the predicted label with the largest summed blend weight among its contributing primitives. Its true instance is the object with the largest summed blend weight. Predicted labels are matched to true instances once per scene over the whole sequence (Hungarian matching), so identity switches over time cost score. Metrics are averaged over instances, then over the scenes of an experiment:

- `mIoU` (primary): mean per-instance intersection-over-union.
- `mAcc`: mean per-instance one-vs-rest pixel accuracy.
- `Recall`: mean per-instance recall.
- `F1`: mean per-instance F1.
- `Recalldyn`: recall averaged over instances that move.
- `Recallstatic`: recall averaged over static instances.

All metrics are higher-is-better and lie in [0, 1].

## Validity and Constraint

A run is scored as failed if any required scene is missing or malformed, or if any label is not an integer ≥ −1. It also fails if the label count is wrong or a scene uses more than 512 distinct labels. A failed run receives 0 for every metric of every experiment.

A run must also beat a fixed reference artifact shipped with the evaluator, the output of a weaker learning-based solution, on interaction-experiment `F1` and `Recalldyn`. Otherwise it is scored as failed. The fixed artifact itself is reported with its real values.

## Baselines and Targets

The paper reports results on Neural3DV, Multi-Human, and SelfCap against OmniSeg3D, DGD, SA4D, and SADG, with headline mIoU 0.801, 0.893, and 0.882. Those full-scale numbers are kept in `scoring/reference.json` under `_paper_reported` for provenance only; they are not on this benchmark's scale. The reference values to beat are the scaled results in `scoring/reference.json`.
