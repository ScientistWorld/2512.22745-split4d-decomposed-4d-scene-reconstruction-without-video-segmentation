# Overview

- **Paper ID:** 2512.22745
- **Title:** Split4D: Decomposed 4D Scene Reconstruction Without Video Segmentation
- **Domain:** Computer Vision / Dynamic 3D Reconstruction and Segmentation
- **TL;DR:** Split4D learns consistent instance features on moving 4D Gaussian primitives from independent per-image masks, avoiding the need for temporally consistent video segmentation.

## Short Summary

The paper addresses instance-level decomposition of dynamic 3D scenes captured by multi-view video. It attaches learnable instance features to FreeTimeGS-style Gaussian primitives with linear motion. It trains those features with a contrastive mask loss over temporally ordered observations. The ordering is central: nearby frames share moving primitives, so streaming feature learning propagates identities through time even when mask labels change per frame or view. Experiments on Neural3DV, Multi-Human, and SelfCap report better mIoU, F1, and dynamic recall than SA4D, SADG, DGD, and OmniSeg3D.

## Key Results

- Paper (full scale): mIoU 0.801 on Neural3DV, 0.893 on Multi-Human, 0.882 on SelfCap.
- Paper streaming ablation (Multi-Human Juggle): mIoU drops from 0.895 to 0.552 and dynamic recall from 0.898 to 0.365 when temporal ordering is removed.
- This gym (reduced procedural scenes, hidden test, 10 scenes per experiment):
  - Dynamic-scene experiment: Split4D mIoU 0.791 / Recalldyn 0.819; random order 0.725 / 0.699.
  - Interaction experiment: Split4D 0.637 / 0.583; random order 0.484 / 0.399.
  - The direction of the paper's claim reproduces; the absolute values are not on the paper's scale.
