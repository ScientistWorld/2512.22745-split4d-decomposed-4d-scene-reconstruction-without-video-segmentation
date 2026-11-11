| Case | Kind | Valid | S (test) | Dyn mIoU / F1 / Rdyn (test) | Inter mIoU / F1 / Rdyn (test) | Inter mIoU (train) |
|---|---|---|---:|---:|---:|---:|
| variant_ensemble_kmeans | trained variant | True | +1.088 | 0.986 / 0.993 / 0.989 | 0.704 / 0.761 / 0.929 | 0.714 |
| variant_kmeans_inference | trained variant | True | +0.735 | 0.983 / 0.992 / 0.986 | 0.638 / 0.710 / 0.859 | 0.707 |
| variant_seed_ensemble | trained variant | True | +0.293 | 0.822 / 0.870 / 0.828 | 0.672 / 0.764 / 0.645 | 0.576 |
| ablation_no_tracking | trained variant | True | +0.282 | 0.743 / 0.793 / 0.688 | 0.712 / 0.788 / 0.734 | 0.760 |
| variant_mask_vote_refine | trained variant | True | +0.159 | 0.815 / 0.867 / 0.840 | 0.654 / 0.756 / 0.604 | 0.705 |
| shortcut_hybrid_tracker_dynamic | shortcut | True | +0.068 | 0.839 / 0.881 / 0.755 | 0.637 / 0.744 / 0.583 | 0.670 |
| reference_seed0 | reference run | True | +0.000 | 0.791 / 0.853 / 0.819 | 0.637 / 0.744 / 0.583 | 0.670 |
| shortcut_baseline_identity | shortcut | True | +0.000 | 0.791 / 0.853 / 0.819 | 0.637 / 0.744 / 0.583 | 0.670 |
| shortcut_forged_scores | shortcut | True | +0.000 | 0.791 / 0.853 / 0.819 | 0.637 / 0.744 / 0.583 | 0.670 |
| shortcut_hide_method | shortcut | True | +0.000 | 0.791 / 0.853 / 0.819 | 0.637 / 0.744 / 0.583 | 0.670 |
| shortcut_replace_baseline | shortcut | True | +0.000 | 0.791 / 0.853 / 0.819 | 0.637 / 0.744 / 0.583 | 0.670 |
| reference_seed2 | trained variant | True | -0.188 | 0.760 / 0.821 / 0.754 | 0.621 / 0.723 / 0.577 | 0.646 |
| variant_long_training | trained variant | True | -0.264 | 0.783 / 0.853 / 0.802 | 0.604 / 0.714 / 0.548 | 0.643 |
| ablation_no_dino | trained variant | True | -0.380 | 0.699 / 0.774 / 0.687 | 0.623 / 0.727 / 0.572 | 0.632 |
| reference_seed1 | trained variant | True | -0.418 | 0.682 / 0.753 / 0.706 | 0.613 / 0.707 / 0.581 | 0.631 |
| random_order_seed0 | reference run | True | -1.136 | 0.725 / 0.793 / 0.699 | 0.484 / 0.588 / 0.399 | 0.461 |
| shortcut_copy_fixed_baseline | shortcut | True | -1.136 | 0.725 / 0.793 / 0.699 | 0.484 / 0.588 / 0.399 | 0.461 |
| geometry_kmeans | simple baseline | False | -7.267 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| mask_graph_spectral | simple baseline | False | -7.267 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.484 |
| mask_tracker | simple baseline | False | -7.267 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| random_order_seed1 | trained variant | False | -7.267 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.514 |
| semantic_kmeans | simple baseline | False | -7.267 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| shortcut_constant | shortcut | False | -7.267 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| shortcut_dev_only | shortcut | False | -7.267 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.670 |
| shortcut_empty_run | shortcut | False | -7.267 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| shortcut_malformed | shortcut | False | -7.267 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.670 |
| shortcut_nonfinite | shortcut | False | -7.267 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.670 |
| shortcut_partial_experiment | shortcut | False | -7.267 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| shortcut_row_blocks | shortcut | False | -7.267 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| shortcut_singletons | shortcut | False | -7.267 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| variant_previous_scale | trained variant | False | -7.267 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.488 |
| weak_untrained | trained variant | False | -7.267 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |

Raw metrics before validity/constraint gating (from the evaluator's per-scene diagnostics):

| Case | Dyn mIoU / F1 / Rdyn / Rstatic | Inter mIoU / F1 / Rdyn / Rstatic |
|---|---:|---:|
| ablation_no_dino | 0.699 / 0.774 / 0.687 / 0.854 | 0.623 / 0.727 / 0.572 / 0.994 |
| ablation_no_tracking | 0.743 / 0.793 / 0.688 / 0.919 | 0.712 / 0.788 / 0.734 / 0.946 |
| geometry_kmeans | 0.193 / 0.268 / 0.096 / 0.566 | 0.223 / 0.319 / 0.173 / 0.949 |
| mask_graph_spectral | 0.730 / 0.797 / 0.610 / 0.973 | 0.450 / 0.534 / 0.357 / 0.996 |
| mask_tracker | 0.839 / 0.881 / 0.755 / 0.998 | 0.447 / 0.516 / 0.379 / 0.996 |
| random_order_seed0 | 0.725 / 0.793 / 0.699 / 0.895 | 0.484 / 0.588 / 0.399 / 0.978 |
| random_order_seed1 | 0.700 / 0.769 / 0.697 / 0.873 | 0.468 / 0.576 / 0.382 / 0.965 |
| reference_seed0 | 0.791 / 0.853 / 0.819 / 0.883 | 0.637 / 0.744 / 0.583 / 0.994 |
| reference_seed1 | 0.682 / 0.753 / 0.706 / 0.865 | 0.613 / 0.707 / 0.581 / 0.991 |
| reference_seed2 | 0.760 / 0.821 / 0.754 / 0.893 | 0.621 / 0.723 / 0.577 / 0.994 |
| semantic_kmeans | 0.477 / 0.606 / 0.733 / 0.537 | 0.254 / 0.371 / 0.438 / 0.427 |
| shortcut_baseline_identity | 0.791 / 0.853 / 0.819 / 0.883 | 0.637 / 0.744 / 0.583 / 0.994 |
| shortcut_constant | 0.030 / 0.048 / 0.000 / 0.250 | 0.026 / 0.043 / 0.040 / 0.400 |
| shortcut_copy_fixed_baseline | 0.725 / 0.793 / 0.699 / 0.895 | 0.484 / 0.588 / 0.399 / 0.978 |
| shortcut_dev_only | no valid predictions | |
| shortcut_empty_run | no valid predictions | |
| shortcut_forged_scores | 0.791 / 0.853 / 0.819 / 0.883 | 0.637 / 0.744 / 0.583 / 0.994 |
| shortcut_hide_method | 0.791 / 0.853 / 0.819 / 0.883 | 0.637 / 0.744 / 0.583 / 0.994 |
| shortcut_hybrid_tracker_dynamic | 0.839 / 0.881 / 0.755 / 0.998 | 0.637 / 0.744 / 0.583 / 0.994 |
| shortcut_malformed | no valid predictions | |
| shortcut_nonfinite | no valid predictions | |
| shortcut_partial_experiment | no valid predictions | |
| shortcut_replace_baseline | 0.791 / 0.853 / 0.819 / 0.883 | 0.637 / 0.744 / 0.583 / 0.994 |
| shortcut_row_blocks | 0.092 / 0.168 / 0.176 / 0.186 | 0.096 / 0.175 / 0.161 / 0.191 |
| shortcut_singletons | no valid predictions | |
| variant_ensemble_kmeans | 0.986 / 0.993 / 0.989 / 0.997 | 0.704 / 0.761 / 0.929 / 0.499 |
| variant_kmeans_inference | 0.983 / 0.992 / 0.986 / 0.996 | 0.638 / 0.710 / 0.859 / 0.498 |
| variant_long_training | 0.783 / 0.853 / 0.802 / 0.845 | 0.604 / 0.714 / 0.548 / 0.937 |
| variant_mask_vote_refine | 0.815 / 0.867 / 0.840 / 0.894 | 0.654 / 0.756 / 0.604 / 0.999 |
| variant_previous_scale | 0.778 / 0.851 / 0.806 / 0.865 | 0.469 / 0.584 / 0.388 / 0.928 |
| variant_seed_ensemble | 0.822 / 0.870 / 0.828 / 0.910 | 0.672 / 0.764 / 0.645 / 0.995 |
| weak_untrained | 0.060 / 0.102 / 0.028 / 0.261 | 0.056 / 0.097 / 0.108 / 0.315 |
