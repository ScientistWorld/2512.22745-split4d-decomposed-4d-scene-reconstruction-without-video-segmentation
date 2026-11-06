# Split4D: Decomposed 4D Scene Reconstruction Without Video Segmentation

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2512.22745"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2512.22745-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2512.22745](https://arxiv.org/abs/2512.22745) |
| **Domain** | Computer Vision / Dynamic 3D Reconstruction and Segmentation |
| **TL;DR** | Split4D learns consistent instance features on moving 4D Gaussian primitives from independent per-image masks, avoiding the need for temporally consistent video segmentation. |

## Summary

The paper addresses instance-level decomposition of dynamic 3D scenes captured by multi-view video. Its method attaches learnable instance features to FreetimeGS-style Gaussian primitives with linear motion, then trains those features with a contrastive mask loss over temporally ordered observations. The ordering is central: nearby frames share moving primitives, so streaming feature learning propagates identities through time even when mask labels change per frame or view. Experiments on Neural3DV, Multi-Human, and SelfCap report better mIoU, F1, and dynamic recall than SA4D, SADG, DGD, and OmniSeg3D.

## Problem

How can a system recover temporally consistent instance-level segmentation of a dynamic 3D scene from multi-view videos when the available 2D masks are independent per image and do not have stable labels across views or time?

## The paper's method

Split4D learns a decomposed 4D feature field for dynamic Gaussian scenes without requiring video segmentation or globally consistent 2D instance labels.

## Key results of the paper

- Reported mIoU: 0.801 on Neural3DV, 0.893 on Multi-Human, and 0.882 on SelfCap.
- Streaming ablation on Multi-Human Juggle: mIoU drops from 0.895 to 0.552 and dynamic recall from 0.898 to 0.365 when temporal ordering is removed.
- The paper trains only the instance features after reconstruction, using 32-dimensional features, Adam, and about 10,000 iterations per scene.

## Run it

```bash
pip install sciworld
sw research 2512.22745 --mode innovate --agent claude-code
sw evaluate 2512.22745
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

