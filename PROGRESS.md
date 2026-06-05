# Progress

## What Works

- Paper analysis and benchmark documentation are filled in.
- `data/split4d_synthetic.py` defines deterministic fixed 4D Gaussian-style scenes with linear primitive motion, temporal opacity, inconsistent per-image mask ids, and semantic vectors.
- `method/split4d.py` implements the paper's feature-stage algorithm at reduced scale: learnable per-primitive features, contrastive mask clustering, streaming temporal sampling, 3D tracking alignment, early semantic alignment, clustering, and filtering.
- `eval/metrics.py` computes mIoU, mAcc, Recall, F1, and dynamic recall from method output artifacts without importing `method/`.
- `python validate.py` passes before submission.

## Results

No reproduced scores yet. The first compute job will run `scripts/reproduce.sh`, write `scoring/scores.json`, and compare against `scoring/reference.json`.

## Remaining

- Run the full pipeline inside the container after dependencies are installed.
- Inspect whether streaming Split4D beats the random-order ablation on the dynamic scenes.
- If the gap is weak, tune only scale-preserving hyperparameters such as step count and loss weights.

## Issues

- The login Python environment lacks NumPy, so only `py_compile` and repository validation were run locally. Dependency-backed execution is deferred to the submitted job through `environment/setup.sh`.

## Deviations from Paper

- Original datasets are Neural3DV, Multi-Human, and SelfCap with large multi-view videos and pretrained 4D Gaussian reconstructions. This reproduction uses deterministic procedural dynamic Gaussian scenes because the original data plus reconstruction pipeline are large and not released as a ready benchmark in the paper markdown.
- The algorithm is not replaced: the implemented computation still keeps fixed Gaussian geometry and optimizes per-primitive instance features using the paper's contrastive mask loss, streaming temporal sampling, 3D tracking regularization, semantic alignment, and clustering/filtering inference.
- HDBSCAN inference is represented with deterministic KMeans using the known number of evaluation instances for this small benchmark, followed by the paper's filtering cues. This is a scale/library substitution for the clustering postprocess, not a replacement of the feature-learning algorithm.
