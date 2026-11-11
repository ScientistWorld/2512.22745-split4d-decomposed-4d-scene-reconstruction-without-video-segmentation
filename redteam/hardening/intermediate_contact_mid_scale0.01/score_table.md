| Case | Kind | Valid | S (test) | Dyn mIoU / F1 / Rdyn (test) | Inter mIoU / F1 / Rdyn (test) | Inter mIoU (train) |
|---|---|---|---:|---:|---:|---:|
| ablation_no_tracking | trained variant | True | +1.574 | 0.769 / 0.815 / 0.708 | 0.734 / 0.807 / 0.732 | 0.742 |
| variant_weak_regularizers | trained variant | True | +1.100 | 0.791 / 0.853 / 0.819 | 0.637 / 0.744 / 0.583 | 0.670 |
| variant_ensemble_kmeans | trained variant | True | +0.977 | 0.938 / 0.967 / 0.958 | 0.490 / 0.593 / 0.600 | 0.486 |
| variant_kmeans_inference | trained variant | True | +0.738 | 0.932 / 0.964 / 0.954 | 0.446 / 0.555 / 0.581 | 0.467 |
| variant_mask_vote_refine | trained variant | True | +0.292 | 0.830 / 0.887 / 0.858 | 0.492 / 0.600 / 0.403 | 0.526 |
| variant_seed_ensemble | trained variant | True | +0.239 | 0.783 / 0.852 / 0.841 | 0.500 / 0.613 / 0.435 | 0.500 |
| reference_seed0 | reference run | True | +0.000 | 0.785 / 0.860 / 0.831 | 0.468 / 0.582 / 0.388 | 0.496 |
| shortcut_baseline_identity | shortcut | True | +0.000 | 0.785 / 0.860 / 0.831 | 0.468 / 0.582 / 0.388 | 0.496 |
| shortcut_forged_scores | shortcut | True | +0.000 | 0.785 / 0.860 / 0.831 | 0.468 / 0.582 / 0.388 | 0.496 |
| shortcut_hide_method | shortcut | True | +0.000 | 0.785 / 0.860 / 0.831 | 0.468 / 0.582 / 0.388 | 0.496 |
| shortcut_replace_baseline | shortcut | True | +0.000 | 0.785 / 0.860 / 0.831 | 0.468 / 0.582 / 0.388 | 0.496 |
| mask_tracker | simple baseline | True | -0.020 | 0.839 / 0.881 / 0.755 | 0.447 / 0.516 / 0.379 | 0.000 |
| reference_seed2 | trained variant | True | -0.249 | 0.727 / 0.800 / 0.773 | 0.460 / 0.578 / 0.399 | 0.489 |
| reference_seed1 | trained variant | True | -0.339 | 0.707 / 0.780 / 0.764 | 0.456 / 0.569 / 0.398 | 0.481 |
| mask_graph_spectral | simple baseline | True | -0.479 | 0.730 / 0.797 / 0.610 | 0.450 / 0.534 / 0.357 | 0.484 |
| random_order_seed0 | reference run | True | -0.551 | 0.778 / 0.859 / 0.814 | 0.399 / 0.506 / 0.294 | 0.424 |
| shortcut_copy_fixed_baseline | shortcut | True | -0.551 | 0.778 / 0.859 / 0.814 | 0.399 / 0.506 / 0.294 | 0.424 |
| variant_long_training | trained variant | True | -0.571 | 0.713 / 0.819 / 0.791 | 0.425 / 0.549 / 0.352 | 0.429 |
| ablation_no_dino | trained variant | True | -0.718 | 0.631 / 0.753 / 0.662 | 0.453 / 0.565 / 0.363 | 0.489 |
| geometry_kmeans | simple baseline | False | -7.258 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| random_order_seed1 | trained variant | False | -7.258 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| semantic_kmeans | simple baseline | False | -7.258 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| shortcut_constant | shortcut | False | -7.258 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| shortcut_dev_only | shortcut | False | -7.258 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.496 |
| shortcut_empty_run | shortcut | False | -7.258 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| shortcut_malformed | shortcut | False | -7.258 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.496 |
| shortcut_nonfinite | shortcut | False | -7.258 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.496 |
| shortcut_partial_experiment | shortcut | False | -7.258 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| shortcut_row_blocks | shortcut | False | -7.258 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| shortcut_singletons | shortcut | False | -7.258 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| weak_untrained | trained variant | False | -7.258 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |

Raw metrics before validity/constraint gating (from the evaluator's per-scene diagnostics):

| Case | Dyn mIoU / F1 / Rdyn / Rstatic | Inter mIoU / F1 / Rdyn / Rstatic |
|---|---:|---:|
| ablation_no_dino | 0.631 / 0.753 / 0.662 / 0.764 | 0.453 / 0.565 / 0.363 / 0.913 |
| ablation_no_tracking | 0.769 / 0.815 / 0.708 / 0.929 | 0.734 / 0.807 / 0.732 / 0.995 |
| geometry_kmeans | 0.193 / 0.268 / 0.096 / 0.566 | 0.223 / 0.319 / 0.173 / 0.949 |
| mask_graph_spectral | 0.730 / 0.797 / 0.610 / 0.973 | 0.450 / 0.534 / 0.357 / 0.996 |
| mask_tracker | 0.839 / 0.881 / 0.755 / 0.998 | 0.447 / 0.516 / 0.379 / 0.996 |
| random_order_seed0 | 0.778 / 0.859 / 0.814 / 0.832 | 0.399 / 0.506 / 0.294 / 0.889 |
| random_order_seed1 | 0.753 / 0.836 / 0.778 / 0.832 | 0.390 / 0.497 / 0.302 / 0.858 |
| reference_seed0 | 0.785 / 0.860 / 0.831 / 0.856 | 0.468 / 0.582 / 0.388 / 0.928 |
| reference_seed1 | 0.707 / 0.780 / 0.764 / 0.823 | 0.456 / 0.569 / 0.398 / 0.887 |
| reference_seed2 | 0.727 / 0.800 / 0.773 / 0.851 | 0.460 / 0.578 / 0.399 / 0.870 |
| semantic_kmeans | 0.477 / 0.606 / 0.733 / 0.537 | 0.254 / 0.371 / 0.438 / 0.427 |
| shortcut_baseline_identity | 0.785 / 0.860 / 0.831 / 0.856 | 0.468 / 0.582 / 0.388 / 0.928 |
| shortcut_constant | 0.030 / 0.048 / 0.000 / 0.250 | 0.026 / 0.043 / 0.040 / 0.400 |
| shortcut_copy_fixed_baseline | 0.778 / 0.859 / 0.814 / 0.832 | 0.399 / 0.506 / 0.294 / 0.889 |
| shortcut_dev_only | no valid predictions | |
| shortcut_empty_run | no valid predictions | |
| shortcut_forged_scores | 0.785 / 0.860 / 0.831 / 0.856 | 0.468 / 0.582 / 0.388 / 0.928 |
| shortcut_hide_method | 0.785 / 0.860 / 0.831 / 0.856 | 0.468 / 0.582 / 0.388 / 0.928 |
| shortcut_malformed | no valid predictions | |
| shortcut_nonfinite | no valid predictions | |
| shortcut_partial_experiment | no valid predictions | |
| shortcut_replace_baseline | 0.785 / 0.860 / 0.831 / 0.856 | 0.468 / 0.582 / 0.388 / 0.928 |
| shortcut_row_blocks | 0.092 / 0.168 / 0.176 / 0.186 | 0.096 / 0.175 / 0.161 / 0.191 |
| shortcut_singletons | no valid predictions | |
| variant_ensemble_kmeans | 0.938 / 0.967 / 0.958 / 0.980 | 0.490 / 0.593 / 0.600 / 0.684 |
| variant_kmeans_inference | 0.932 / 0.964 / 0.954 / 0.978 | 0.446 / 0.555 / 0.581 / 0.636 |
| variant_long_training | 0.713 / 0.819 / 0.791 / 0.729 | 0.425 / 0.549 / 0.352 / 0.819 |
| variant_mask_vote_refine | 0.830 / 0.887 / 0.858 / 0.886 | 0.492 / 0.600 / 0.403 / 0.984 |
| variant_seed_ensemble | 0.783 / 0.852 / 0.841 / 0.863 | 0.500 / 0.613 / 0.435 / 0.938 |
| variant_weak_regularizers | 0.791 / 0.853 / 0.819 / 0.883 | 0.637 / 0.744 / 0.583 / 0.994 |
| weak_untrained | 0.060 / 0.102 / 0.028 / 0.261 | 0.056 / 0.097 / 0.108 / 0.315 |
