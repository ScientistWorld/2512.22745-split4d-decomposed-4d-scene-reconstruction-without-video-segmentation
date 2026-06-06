# Verification Report

## Initial Assessment

The workspace started at claimed milestone `core_claim`. The implemented method is a reduced-scale reproduction of the paper's fixed-primitive dynamic scene feature learning pipeline, and the current artifacts already showed the central ordered-observation benefit over the random-order ablation. The main risk was packaging quality: canonical scoring, train/test scoring, and public gym documentation needed to be checked for consistency and method leakage.

## Issues Found

- Critical: `scoring/scores.json` had been generated before the canonical evaluator reported `MaskFeatureAgreement` and `TrackingFeatureMargin`, while `reference.json`, `scores_train.json`, and `scores_test.json` already expected those metrics.
- Minor: Several method-agnostic and method-safe markdown files still contained template text or solution-specific wording that was too revealing for from-scratch scientist agents. A final pass also found wording that exposed observation ordering, feature learning, and matching details in documents visible to from-scratch agents.
- Minor: The reduced scaled rows in `scoring/reference.json` omitted newly scored metrics and described an older accepted job rather than the final verified canonical score baseline.

## Fixes Applied

- Patched the canonical evaluator and core-claim gate to require the full declared metric family, then ran CPU job `6eb37562-eec` inside the container. It regenerated `scores.json`, `scores_train.json`, and `scores_test.json`; `eval.check_core_claim` passed; `python validate.py --post-design` passed.
- Tightened `briefing/problem.md`, `briefing/evaluation.md`, `scoring/EXPERIMENTS.md`, `scoring/TARGETS.md`, `scoring/CONSTRAINTS.md`, and `scoring/DIRECTION.md` so they describe the task, scoring contract, constraints, and research scope without exposing the implementation recipe.
- Rechecked the from-scratch-facing scoring documents after the container pass and replaced remaining solution-specific phrasing with benchmark-level descriptions of candidate artifacts, temporal coherence, fixed scene elements, and the output contract.
- Updated the active scaled rows in `scoring/reference.json` to include all declared metrics and added notes explaining that these are reproducible scaled gym baselines, while the paper's full-scale reported values remain under `paper_fullscale_*`.

## Final Judgment

The workspace is ready to serve as a research gym. Improve-mode agents can run the existing method and compare against the accepted score artifacts; from-scratch agents can use the problem definition, data, evaluator, constraints, and direction documents without needing the paper method. The evaluation reads output artifacts rather than method internals, has train/test split checks, includes benefit and constraint metrics, and passed the post-design validator in the container.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: medium
- Key limitation: reduced deterministic synthetic scenes rather than the paper's full Neural3DV, Multi-Human, and SelfCap datasets.
