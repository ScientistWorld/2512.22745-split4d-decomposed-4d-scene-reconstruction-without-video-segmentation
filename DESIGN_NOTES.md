# Design Notes

## Checked
- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. Train invokes `eval.train.evaluate` and writes `scoring/scores_train.json`; test invokes `eval.test.evaluate` and writes `scoring/scores_test.json`.
- Confirmed both split evaluators discover JSON method artifacts under `outputs/` by default and score those artifacts instead of rerunning the method or hardcoding a method name.
- Grepped for train/test cross-imports in `eval/train`, `eval/test`, and the split scripts. None were found.
- Grepped the edited eval/reference files for paper-agreement metric names such as `published`, `vs_paper`, `deviation_from_paper`, `reproduction_error`, and `fit_to_published`. None were found.
- Ran `bash -n` on `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh`.
- Ran job `e28be5ba-1ec` with the split evaluators only. It exited 0 and produced both split score files, but `TrackingFeatureAgreement` saturated at about `1.0` for every method and slice.
- Ran job `55e83125-51d` after replacing the saturated metric. It exited 0, wrote 27 train metric values and 27 test metric values, and `validate.py --post-design` passed. `TrackingFeatureMargin` now ranges from about `0.417` to `0.588` instead of saturating.

## Changed
- Added `MaskFeatureAgreement` to both split evaluators. It scores the emitted per-primitive feature artifact on complementary observation subsets, using same-mask feature margins rather than final labels alone.
- Replaced saturated `TrackingFeatureAgreement` with `TrackingFeatureMargin`. It scores the positive feature similarity of adjacent-time geometry matches against the hardest different-object negative in the next frame, making non-discriminative self-similarity less likely to dominate the metric.
- Added both feature metrics to the reduced, actually-emitted experiments in `scoring/reference.json` with modest positive coefficients.
- Moved nearly all ranking weight onto the two reduced experiments emitted by the evaluators: `scaled_linear_motion_segmentation` is now `0.548`, `scaled_streaming_ablation` is now `0.448`, and each `paper_fullscale_*` provenance group keeps a validator-compatible `0.001` weight. This avoids meaningful weight on experiments the train/test scripts never compute while preserving the recorded paper values.

## Runtime Concerns For The Runner
- The split evaluators now require every scored scene payload to include a `features` matrix with one row per primitive, in addition to `pred_labels`. The current `method.split4d` artifacts already use that contract, but the next runner should verify any baseline/future artifact format before executing the split.
- The replacement tracking margin was added after the first real split run exposed saturation. The rerun confirmed it no longer saturates and train/test gaps remain informative.
- The synthetic data module can expose truth labels for internal evaluation. The added feature metrics do not fully prevent a method from deliberately emitting truth-derived labels and truth-derived features, but they make simple visible-label memorization less effective and better align the split with the paper's learned feature representation.
