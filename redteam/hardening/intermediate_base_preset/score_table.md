| Case | Kind | Valid | S (test) | Dyn mIoU / F1 / Rdyn (test) | Inter mIoU / F1 / Rdyn (test) | Inter mIoU (train) |
|---|---|---|---:|---:|---:|---:|
| variant_long_training | trained variant | True | +0.326 | 0.754 / 0.832 / 0.736 | 0.456 / 0.567 / 0.386 | 0.440 |
| reference_seed1 | trained variant | True | +0.323 | 0.702 / 0.763 / 0.622 | 0.498 / 0.605 / 0.440 | 0.428 |
| variant_seed_ensemble | trained variant | True | +0.277 | 0.684 / 0.742 / 0.582 | 0.510 / 0.613 / 0.442 | 0.477 |
| variant_weak_regularizers | trained variant | True | +0.147 | 0.664 / 0.720 / 0.559 | 0.500 / 0.593 / 0.444 | 0.512 |
| reference_seed0 | reference run | True | +0.000 | 0.675 / 0.740 / 0.587 | 0.475 / 0.576 / 0.391 | 0.454 |
| shortcut_baseline_identity | shortcut | True | +0.000 | 0.675 / 0.740 / 0.587 | 0.475 / 0.576 / 0.391 | 0.454 |
| shortcut_forged_scores | shortcut | True | +0.000 | 0.675 / 0.740 / 0.587 | 0.475 / 0.576 / 0.391 | 0.454 |
| shortcut_hide_method | shortcut | True | +0.000 | 0.675 / 0.740 / 0.587 | 0.475 / 0.576 / 0.391 | 0.454 |
| shortcut_replace_baseline | shortcut | True | +0.000 | 0.675 / 0.740 / 0.587 | 0.475 / 0.576 / 0.391 | 0.454 |
| random_order_seed0 | reference run | True | -0.178 | 0.688 / 0.755 / 0.639 | 0.437 / 0.540 / 0.344 | 0.395 |
| shortcut_copy_fixed_baseline | shortcut | True | -0.178 | 0.688 / 0.755 / 0.639 | 0.437 / 0.540 / 0.344 | 0.395 |
| variant_strong_regularizers | trained variant | True | -0.332 | 0.638 / 0.709 / 0.521 | 0.452 / 0.551 / 0.382 | 0.000 |
| reference_seed2 | trained variant | True | -0.359 | 0.616 / 0.682 / 0.500 | 0.465 / 0.560 / 0.382 | 0.474 |
| ablation_no_dino | trained variant | True | -0.361 | 0.576 / 0.666 / 0.433 | 0.494 / 0.597 / 0.416 | 0.461 |
| ablation_no_tracking | trained variant | False | -7.303 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.440 |
| geometry_kmeans | simple baseline | False | -7.303 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| mask_graph_spectral | simple baseline | False | -7.303 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| mask_tracker | simple baseline | False | -7.303 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| random_order_seed1 | trained variant | False | -7.303 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| semantic_kmeans | simple baseline | False | -7.303 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.466 |
| shortcut_constant | shortcut | False | -7.303 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| shortcut_dev_only | shortcut | False | -7.303 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.454 |
| shortcut_empty_run | shortcut | False | -7.303 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| shortcut_malformed | shortcut | False | -7.303 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.454 |
| shortcut_nonfinite | shortcut | False | -7.303 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.454 |
| shortcut_partial_experiment | shortcut | False | -7.303 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| shortcut_row_blocks | shortcut | False | -7.303 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| shortcut_singletons | shortcut | False | -7.303 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
| weak_untrained | trained variant | False | -7.303 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 |
