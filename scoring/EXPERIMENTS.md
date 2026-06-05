# Experiment Overview

## linear_motion_segmentation
This experiment checks basic dynamic-scene instance decomposition under moderate motion. It asks whether a system can recover object identities from inconsistent image-level supervision while maintaining good all-instance segmentation and dynamic-object recall.

## interaction_motion_segmentation
This experiment checks the harder interaction setting where motion and semantic ambiguity make identity propagation more difficult. It is intended to reveal failures that are hidden by easier static or single-object scenes.

## streaming_ablation
This experiment compares the reproduced system against a randomized temporal-order variant on the same evaluation task. It tests whether the central temporal consistency claim is reflected in the measured segmentation quality rather than only in qualitative outputs.
