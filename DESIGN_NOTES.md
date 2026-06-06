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
- Added `ClusterCountError` to both train and test evaluators. This penalizes label fragmentation or instance collapse that can partially game overlap metrics.
- Added required `description` fields to every metric in `scoring/reference.json`.
- Added the two new metrics to the reduced train/test-scored experiments in `scoring/reference.json` with modest coefficients. The full-scale paper-reference groups were left as provenance-only entries and were not given synthetic-only metrics.

## Runtime Concerns For Next Iteration
- I did not run either evaluator or generate scores, per the file-only instruction. The runner should confirm that the emitted `scores_train.json` and `scores_test.json` include `Recallstatic` and `ClusterCountError` for both reduced experiments.
- `ClusterCountError` may saturate at zero for normal methods that infer the public instance count correctly. It is still useful as an anti-fragmentation guard, but should be treated as a constraint signal rather than the main benefit metric.
- The train/test split remains a deterministic 50/50 stratified primitive split over the same scenes. This is lightweight and calibrated, but it is not a strong OOD split; the added static/dynamic/count triangulation is intended to reduce shortcut risk without changing the current artifact contract.
