# Design Notes

## Checked
- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. Train calls `eval.train.evaluate` and writes `scoring/scores_train.json`; test calls `eval.test.evaluate` and writes `scoring/scores_test.json`.
- Checked that both train and test evaluators discover JSON artifacts under `outputs/` by default, so future methods are scored through their emitted artifacts rather than through a hardcoded method name.
- Checked for train/test cross-imports. None were found; the only test-side reference to train is the intended `--train-scores scoring/scores_train.json` method-key mirror.
- Checked that no `scoring/scores_train.json` or `scoring/scores_test.json` stubs exist.
- Ran `bash -n` on both train/test evaluator scripts.
- Grepped the changed eval/scoring files for paper-agreement metric names or formulas; none were found in metric names or emitted keys.

## Changed
- Added `Recallstatic` to both train and test evaluators. This keeps the score from rewarding methods that tune only for moving-object recall while sacrificing slower/static instances.
- Replaced the saturated `ClusterCountError` guard with `ClusterSizeError` in both train and test evaluators. This penalizes cluster imbalance without being zero whenever a method predicts the right number of clusters.
- Matched the primary linear-motion split contract to `eval/metrics.py` by excluding the `split4d_random_order` ablation artifact from `scaled_linear_motion_segmentation`.
- Changed the train/test split from hard 50/50 primitive filtering to deterministic complementary weighted views over the full method artifact. This keeps the primary mIoU comparable to the accepted full-scene reproduction while still making train and test score different held-out point emphasis.
- Added required `description` fields to every metric in `scoring/reference.json`.
- Added the two new metrics to the reduced train/test-scored experiments in `scoring/reference.json` with modest coefficients. The full-scale paper-reference groups were left as provenance-only entries and were not given synthetic-only metrics.

## Runtime Findings
- Job `d54f6cae-3da` ran `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` successfully on CPU and wrote both split score files.
- The first run exposed two evaluator issues: `ClusterCountError` saturated at zero for every emitted method and the primary linear-motion experiment included the random-order ablation artifact. Both issues were fixed before the next submission.
- Job `e249f8fc-0f4` verified the fixed evaluators. Train and test now emit the same experiment/method key sets, with 21 numeric metric values per split and no nulls.
- The current ignored `outputs/` artifacts score below the random-order baseline on every interaction-ablation metric in both train and test. Earlier accepted job `82b7b084-80f` had the proposed streaming method ahead, but its prediction JSON artifacts are not archived in git. Because this iteration is not allowed to retrain or modify the method, the split scores are real but fail the gym reasonableness gate for the current artifacts.
- A post-design validation attempt caught that hard half-scene filtering made split mIoU more than 2x lower than the accepted reduced reference on the interaction ablation. The weighted full-artifact split is the correction; it preserves artifact-based scoring and avoids paper-agreement metrics while keeping the primary metric calibrated.
