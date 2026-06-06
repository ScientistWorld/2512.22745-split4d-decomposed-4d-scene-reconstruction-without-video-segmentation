# Experiment Overview

## scaled_linear_motion_segmentation
This experiment checks basic dynamic-scene instance decomposition under moderate motion. It asks whether a system can recover object identities from inconsistent image-level supervision while maintaining good all-instance segmentation and dynamic-object recall.

## scaled_streaming_ablation
This experiment checks temporal-coherence robustness on a harder interaction scene. It compares the candidate decomposition with an included reference artifact under the same data scale and output format, emphasizing balanced instance quality and moving-object recall.

The full-scale paper tables are retained in `scoring/reference.json` for provenance only. They are not emitted by `scripts/evaluate.sh`, so they are not active gym experiments.
