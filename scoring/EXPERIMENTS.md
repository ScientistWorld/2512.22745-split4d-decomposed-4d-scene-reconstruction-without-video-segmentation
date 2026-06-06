# Experiment Overview

## scaled_linear_motion_segmentation
This experiment checks basic dynamic-scene instance decomposition under moderate motion. It asks whether a system can recover object identities from inconsistent image-level supervision while maintaining good all-instance segmentation and dynamic-object recall.

## scaled_streaming_ablation
This experiment checks whether temporal ordering helps on a harder interaction scene. It compares the same learning setup with temporally ordered observations versus random temporal sampling, using the same data scale and evaluation metrics for both runs.

## paper_fullscale_* Reference Groups
These entries preserve the paper's reported full-scale numbers for context. They are not emitted by `scripts/evaluate.sh` because the reproduction uses reduced deterministic scenes rather than the original large video datasets and reconstructed 4D Gaussian fields.
