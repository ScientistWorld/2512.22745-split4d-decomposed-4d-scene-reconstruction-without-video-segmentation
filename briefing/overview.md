# Overview

- **Paper ID:** 2512.22745
- **Title:** Split4D: Decomposed 4D Scene Reconstruction Without Video Segmentation
- **Domain:** Computer Vision / Dynamic 3D Reconstruction and Segmentation
- **TL;DR:** Split4D learns consistent instance features on moving 4D Gaussian primitives from independent per-image masks, avoiding the need for temporally consistent video segmentation.

## Short Summary

The paper addresses instance-level decomposition of dynamic 3D scenes captured by multi-view video. Its method attaches learnable instance features to FreetimeGS-style Gaussian primitives with linear motion, then trains those features with a contrastive mask loss over temporally ordered observations. The ordering is central: nearby frames share moving primitives, so streaming feature learning propagates identities through time even when mask labels change per frame or view. Experiments on Neural3DV, Multi-Human, and SelfCap report better mIoU, F1, and dynamic recall than SA4D, SADG, DGD, and OmniSeg3D.

## Key Results

- Reported mIoU: 0.801 on Neural3DV, 0.893 on Multi-Human, and 0.882 on SelfCap.
- Streaming ablation on Multi-Human Juggle: mIoU drops from 0.895 to 0.552 and dynamic recall from 0.898 to 0.365 when temporal ordering is removed.
- The paper trains only the instance features after reconstruction, using 32-dimensional features, Adam, and about 10,000 iterations per scene.
