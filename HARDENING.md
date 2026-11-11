# Hardening Report (Iteration 11)

**Bottom line:**
- Better results now score higher and worse ones lower, by more than seed noise, on paired comparisons over the same 20 hidden test scenes.
- Every shortcut listed in `REDTEAM.md` is closed. Each one now fails (0 in every metric) or scores exactly the reference.
- The test has headroom that genuine method changes reach.
- What is left: a per-experiment hybrid that copies a simple baseline into one experiment ties the reference within noise, and the interaction experiment's absolute level is low. Both are below.

All "after" numbers come from the final CPU job `df728028-fa4`. In it, every case was written to its own `outputs/` and scored by the unchanged `bash scripts/evaluate_train.sh` and `bash scripts/evaluate_test.sh` with no arguments (`redteam/hardening/harness.py`). The evaluator fingerprint was unchanged in every case. Per-case evidence is in `redteam/hardening/cases/<case>/` (`status.json`, `scores_train.json`, `scores_test.json`). The summary is in [`redteam/hardening/analysis.json`](redteam/hardening/analysis.json) and [`redteam/hardening/score_table.md`](redteam/hardening/score_table.md).

**About the aggregate score S.** S follows the requested rank rule: Σ over experiments and metrics of weight × |coefficient| × direction × (value − reference) / σ. Here σ is the population standard deviation of the candidate rows over all cases, with failed runs included at 0. The "before" S values are the red team's, with their pool and the old reference R. The two scales are not comparable, so raw metrics and paired per-scene wins are given alongside.

## 1. What was wrong and what changed

| Problem (REDTEAM.md) | Fix |
|---|---|
| The answers, and the code that regenerates them, were in scientist-editable `data/split4d_synthetic.py`. Truth was also recoverable from row order (`np.repeat`). | The generator now lives only in `eval/test/scene_builder.py` (hidden). Public `data/scenes/*.npz` hold only inputs: primitive rows randomly permuted, and mask ids drawn as random integers per image. Answers are in `eval/train/truth` (dev) and `eval/test/truth` (test). Neither evaluator reads `data/`. |
| Scenes were trivially easy. Initial-position k-means scored 1.0 on linear motion; the mask graph and brute-force permutation scored about 1.0. | New procedural scenes, two profiles, 60 frames, 6 training views. **Dynamic** profile: a person handling a pan and a cup, a pet, 4 pieces of furniture. **Interaction** profile: 4 people with contact events, a passed ball, 2 static objects. Primitives are short-lived, so each object is a temporal chain. The simulated segmenter merges objects in contact, splits people, pets and tables into parts, flips boundary pixels, adds noise, and drops tiny masks. Mask ids vary in number and value per image. Scoring is in 3 hidden test views over all frames, with one prediction-to-instance matching per scene, so identity switches cost score. |
| Feature-proxy metrics (`MaskFeatureAgreement`, `TrackingFeatureMargin`) rewarded random features, and they were computed from mutable public observations. | Removed. The scored metrics are the paper's five (mIoU, mAcc, Recall, F1, Recalldyn) plus Recallstatic. All are recomputed from raw `pred_labels`; nothing else a method writes is read. |
| Failed or missing runs cost nothing: an empty run scored 0 = the canonical reference, a partial run skipped experiments, a malformed run aborted, and NaN was emitted. | A run with a missing or malformed scene, non-integer labels, labels < −1, a wrong label count, or more than 512 labels gets **0 in every metric of every experiment**. If `outputs/` is empty, a `missing_method_output` row of zeros is written. Output is checked to be finite and in [0, 1]. |
| The comparison baseline was a mutable `outputs/baseline` file. The constraint check was never called by the hidden test, and a magic method name (`split4d_random_order`) changed which experiments were scored. | The random-order artifact is frozen in `eval/{train,test}/fixed_baseline.json`. The evaluator enforces the constraint itself: on interaction F1 and Recalldyn, a run must beat the frozen artifact or it is scored as failed. The frozen artifact itself is reported with its real values. No method name is special. |
| The hidden test filtered methods by the mutable `scores_train.json`. | Removed. Every `*.json` under `outputs/` that has `method` or `scenes` is scored; files sharing a method name are merged. |
| Noise was large relative to the claimed effect (seed range 1.60). Seed 9 lost dynamic recall to the baseline. | 10 test scenes per experiment (20 total) plus 4 dev scenes each. The reference seed is fixed at 0. Paired per-scene comparisons are reported. |
| The reproduction departed from the paper: temperature divided by N_s twice and clamped to [0.035, 0.25]; query-class temperature in every denominator; an extra `lambda_stream` loss; 3-frame streaming batches; 3,200 steps; λ = 1/0.03; a k-means fallback using the mask-id count. | Restored: Eq. 4 PCL temperature with a per-center φ_k (numerical floor 0.01 only); no extra loss; one observation per iteration in temporal order; 10,000 iterations (33 per frame visit, matching the paper's 10k over 300 frames); exponential LR decay from 2.5e-3 to 2e-6; paper λ₁ = 100 and λ₂ = 500 (10 for interaction). Inference is first-frame HDBSCAN with the Eq. 8 filter (primitives removed, i.e. label −1); no fallback, and the instance count is not given. |

**Remaining scale choices, documented in code:**
- **Loss-weight constant.** One constant maps the paper's λ onto this implementation's loss normalization. It was chosen on dev scenes only, between 0.01 and 0.001. At 0.01 tracking dominated and merged interacting people: dev interaction mIoU was 0.496 against 0.670 at 0.001, and dynamic 0.849 against 0.864. The paper describes tracking as a mild, λ₁-insensitive regularizer.
- **HDBSCAN sample.** It uses up to 600 first-frame primitives, where the paper's 2–10% of millions corresponds to thousands.
- **Scene preset.** Merge model and primitive lifetime were chosen on dev scenes only, by the criterion pre-registered in `redteam/hardening/sweep.py`. Chosen preset: `contact_mid` (segmenter merges mainly for objects in 3D contact; dynamic primitive temporal scale 3.5–6 frames). The intermediate runs are archived: `redteam/hardening/intermediate_base_preset/` (first design) and `redteam/hardening/intermediate_contact_mid_scale0.01/` (before the λ-scale fix).

## 2. Do better results score higher? (hidden test)

| Case | Before (REDTEAM, S − R) | After: S | After: dynamic mIoU / Rdyn | After: interaction mIoU / Rdyn | Paired vs reference (mIoU wins/losses of 20) |
|---|---:|---:|---:|---:|---:|
| Reference (seed 0) | 0 | 0 | 0.791 / 0.819 | 0.637 / 0.583 | — |
| Better: ensemble of 3 seeds + k-means inference | — | **+1.088** | 0.986 / 0.989 | 0.704 / 0.929 | 18 / 2 (Rdyn 20 / 0) |
| Better: k-means inference (instance count estimated from masks) | — | **+0.735** | 0.983 / 0.986 | 0.638 / 0.859 | 15 / 5 (Rdyn 20 / 0) |
| Better: 3-seed feature ensemble, paper inference | — | +0.293 | 0.822 / 0.828 | 0.672 / 0.645 | 16 / 4 |
| Better: mask-vote refinement of the reference labels | — | +0.159 | 0.815 / 0.840 | 0.654 / 0.604 | **20 / 0** |
| Old "stronger" mask graph → now a simple baseline | +5.695 | fails constraint (0) | raw 0.730 / 0.610 | raw 0.450 / 0.357 | 3 / 17 |
| Old corrected temperature → now part of the reference | +0.869 | (is the reference) | | | |
| Longer training (30k steps) | −2.708 | −0.264 | 0.783 / 0.802 | 0.604 / 0.548 | 7 / 13 |
| Paper ablation: random order (the frozen artifact) | (baseline row) | **−1.136** | 0.725 / 0.699 | 0.484 / 0.399 | 3 / 17 |
| Paper ablation: no DINO alignment | — | −0.380 | 0.699 / 0.687 | 0.623 / 0.572 | 7 / 13 |
| Paper ablation: no 3D tracking | — | +0.282 | 0.743 / 0.688 | 0.712 / 0.734 | 11 / 9 |
| Previous λ scale (0.01) | — | fails constraint (0) | raw 0.778 / 0.806 | raw 0.469 / 0.388 | 4 / 16 |
| Weaker: zero training steps | −6.188 | fails constraint (0) | raw 0.060 / 0.028 | raw 0.056 / 0.108 | 0 / 20 |

**Noise.** Reference seeds 0, 1 and 2 score S = 0, −0.418 and −0.188, a range of **0.418** (it was 1.596 before). On paired raw metrics, seeds 1 and 2 lose 0.067 and 0.023 mIoU on average and split wins about evenly.

**What exceeds the noise:**
- Both k-means-inference variants beat the reference by more than the seed range, winning dynamic recall in 20/20 scenes. These are honest changes: they use the same learned features and only an instance count estimated from the public masks.
- The paper's streaming effect is −1.136, against a seed range of 0.418, with random order losing 17/20 paired scenes. That is the paper's direction, with dynamic recall 0.819 vs 0.699 and 0.583 vs 0.399.

**What stays inside the noise:**
- The refinement variant improves every one of the 20 scenes but by only +0.02 mIoU, so its S is inside the seed range.
- The paper's tracking-regularizer gain does not reproduce on aggregate. Tracking helps the dynamic profile (0.791 vs 0.743) and hurts interaction (0.637 vs 0.712). That cancels to +0.28 (within noise) for the no-tracking ablation. The paper reports only a small gain (0.866 → 0.895), and this is stated plainly here rather than tuned away.

**Dev vs test agree:** dev = `scores_train.json`, test = `scores_test.json`. Reference dynamic mIoU is 0.864 on dev and 0.791 on test; interaction is 0.670 on dev and 0.637 on test. For random order the figures are 0.776 / 0.725 and 0.461 / 0.484.

## 3. Major shortcuts: fix and score before → after

| Shortcut | Before | Fix | After |
|---|---|---|---|
| Read truth from public generator (`include_truth=True`) | +5.862, all metrics 1.0 | No truth or generator in public code; rows permuted | Not possible. Closest test, `shortcut_row_blocks` (labels from row order): constraint fails → **0**; raw mIoU 0.09 / 0.10 |
| Move truth (edit data generator) | +5.616 | Answers only in `eval/*/truth`; evaluator never imports `data/` | Not possible; editing `data/` changes only the method's own inputs |
| Replace fixed observations (edit public generator) | +0.557 | Feature metrics removed; evaluator uses its own hidden test views | Not possible |
| Replace the baseline artifact with the candidate | +1.656 | Constraint uses the frozen `eval/*/fixed_baseline.json` | `shortcut_replace_baseline`: candidate row **ties the reference (S 0)**; the comparison is unchanged |
| Rename the candidate to the baseline's name | +0.963 | No magic names | `shortcut_baseline_identity`: **ties (S 0)**, constraint still enforced |
| Hide a failed method via `scores_train.json` filter | +0.008 | Filter removed; every method file scored | `shortcut_hide_method`: candidate ties; the hidden failed method appears with **0**s |
| Label-coded features for proxy metrics | +0.027 | Proxy metrics removed | Not applicable |
| Empty run | 0 (= canonical reference) | Worst values | **0 in all metrics** |
| Partial experiment | +0.139 vs canonical | Worst values | **0 in all metrics** (also `shortcut_dev_only`: **0**) |
| Malformed / non-finite / non-integer labels | abort (no score) / NaN | Validation → worst values | **0 in all metrics** |
| Forged scores and pre-written score file | fails | Unchanged: only `pred_labels` read; score file overwritten | **ties (S 0)**, no gain |
| Copy the frozen baseline's predictions | — | Exempt from the constraint, real values reported | S **−1.136** (its real values) |
| Over-segmentation (one label per primitive) | — | At most 512 labels per scene | **0 in all metrics** |
| Brute-force local mask matching (k! permutations) / video-tracker label propagation | +5.695 | Harder scenes: variable mask counts, merges and splits; k! enumeration no longer applies, so it is replaced by Hungarian tracking (`mask_tracker`) | Tracker fails the constraint → **0** (raw dynamic 0.839 / interaction 0.447); raw paired mIoU loses 15/20 |
| Initial-position geometry k-means | +4.292 | Nonlinear motion, chains of short-lived primitives | Fails constraint → **0**; raw 0.193 / 0.223 |
| Semantic-descriptor k-means (new check) | — | Shared super-categories, noisy descriptors | Fails constraint → **0**; raw 0.477 / 0.254 |

## 4. Shortcuts left open

- **Per-experiment hybrid** (`shortcut_hybrid_tracker_dynamic`: mask tracker on dynamic scenes, Split4D on interaction): **S +0.068**, well inside the 0.418 seed range. On the 10 dynamic scenes it wins mIoU on 5 and loses on 5, and it loses Recalldyn (0.755 vs 0.819). It works because the tracker gets near-perfect static recall where the paper's first-frame HDBSCAN over-segments furniture. I left it open:
  - Closing it would need either a different inference than the paper's, or making the dynamic profile harder only to hurt one baseline.
  - The interaction weight (0.6, see below) limits its payoff.
  - The separate grader can flag per-experiment method switching.
- **Dev answers are visible** in `eval/train/truth` by design, for model selection. Training a grouping model on them is ruled out in `CONSTRAINTS.md` but cannot be detected by the evaluator.
- The scene builder in `eval/test/` triggers the validator's soft "detached eval" warning. It is a build-time tool that regenerates the hidden answers, so it must live there; the evaluators only read stored truth.

## 5. Reference score before → after

| | Before (old gym, hidden test) | After (this gym, hidden test) |
|---|---|---|
| Experiment 1 | linear motion: mIoU 0.639, F1 0.773, Rdyn 0.833 | dynamic scene: mIoU **0.791**, F1 0.853, Rdyn 0.819, Rstatic 0.883 |
| Experiment 2 | streaming ablation: mIoU 0.522, F1 0.682, Rdyn 0.746 | interaction: mIoU **0.637**, F1 0.744, Rdyn 0.583, Rstatic 0.994 |
| Random-order baseline | 0.274 / 0.410 / 0.587 | dynamic 0.725 / 0.793 / 0.699; interaction 0.484 / 0.588 / 0.399 |
| Old R vs canonical | R = 0.685 (canonical values ≠ hidden slice) | `reference.json` results **equal** the hidden-test rows of the reference runs (S = 0 exactly) |

The scores are not comparable across gyms (different scenes and metrics).

**Score files and weights:**
- `scoring/scores_test.json`, `scores_train.json` and `scores.json` (= dev) were regenerated from the reference runs alone (`split4d_streaming`, `split4d_random_order`).
- The paper's reported numbers are unchanged, under `_paper_reported` in `reference.json`.
- **Weights changed** from 0.548/0.448 (+4 × 0.001 for the never-emitted paper tables) to **0.4 dynamic / 0.6 interaction**. The paper's central claim (streaming propagation through interactions, Table 2 Juggle) and its biggest margins are on Multi-Human. This experiment also holds the constraint, and it is where simple baselines fail.
- Coefficients keep the earlier values: mIoU 1.0, F1 0.8, Recalldyn 0.8, Recall 0.5, Recallstatic 0.35, mAcc 0.2.

## 6. Headroom check

**Bounds.** No reference metric sits at its bound (`analysis.json: reference_metrics_at_bound = []`). The closest is interaction Recallstatic at 0.994, which has a low coefficient. mAcc is compressed (0.94–0.97) but still varies across methods.

**Headroom is real and reachable:**
- K-means inference reaches 0.98 dynamic mIoU.
- The ensemble variant raises interaction dynamic recall to 0.93.
- Interaction mIoU tops out around 0.70–0.71 in every variant tried, so more remains there for better identity handling through contacts.

**Simple baselines stay below the reference** (geometry, semantic, mask graph, tracker). All fail the constraint, and their raw values are below the reference in the interaction experiment (0.22–0.45 vs 0.637).

**Brute force** in the old form (enumerating label permutations) no longer applies, because mask counts vary per image. Its polynomial replacement (Hungarian tracking) loses on aggregate.

**No new headroom experiment was needed,** because the reference is far from the bound in both experiments.

## 7. What I could not fix

- **Interaction level is low.** The reproduced interaction level (0.637 mIoU) is far below the paper's Multi-Human 0.893. Per-object diagnostics (`redteam/hardening/diagnostics.json`, from the dev-scene sweep) show people's identities get confused at contacts where the simulated segmenter merges them. This is a property of the substitute data, not of the evaluator. The paper's numbers are kept only as provenance.
- **Tracking regularizer.** Its small benefit (paper Table 2) is not reproduced on aggregate (see §2).
- **Inference over-segmentation.** The paper's first-frame HDBSCAN at this reduced primitive count over-segments large static objects; k-means with an estimated count does much better. That is left as headroom rather than changing the paper's inference.
- **Substitute data.** The data is procedural, not Neural3DV/Multi-Human/SelfCap. The semantic descriptors stand in for DINOv2, and primitives stand in for a FreeTimeGS reconstruction.
- **Infrastructure.** One job lost its worker node after finishing every stage. It was cancelled and its results used; the final job ran to completion.

## 8. Reproduce

- `bash scripts/reproduce.sh` reruns the baseline and reference and scores both splits.
- `bash scripts/run.sh` reruns the whole hardening pipeline (resumable).
- `python3 -m redteam.hardening.analyze` recomputes the tables from the saved score files (stdlib only).
- `python validate.py --post-design` passes.
