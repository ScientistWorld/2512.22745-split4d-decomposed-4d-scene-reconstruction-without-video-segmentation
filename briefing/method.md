# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

Split4D learns a decomposed 4D feature field for dynamic Gaussian scenes without requiring video segmentation or globally consistent 2D instance labels.

## Approach

The scene is represented as Freetime FeatureGS: Gaussian primitives with 4D centers, temporal extent, opacity, appearance, linear 3D velocity, and a learnable instance feature vector. The pretrained geometric reconstruction is kept fixed while the instance features are optimized.

For each view and time, features are rendered by alpha blending (Eq. 2). A contrastive clustering objective (Eq. 4) samples N_s pixels per 2D mask and pulls them toward that mask's feature center while pushing them from the other masks' centers. Each center k has its own PCL-style temperature phi_k = sum ||f - fbar_k||^2 / (N_s log(N_s + 10)). Gaussian primitives move and are reused locally across neighboring times, so this transfers feature identity through time even when mask labels are inconsistent.

The key training strategy is streaming sampling: observations are processed in temporal order, so local correspondences propagate gradually instead of letting far-apart frames settle into unrelated local optima. Two regularizers add robustness:
- a 3D tracking alignment loss (Eq. 5) matches primitives that are fading out to their nearest primitive in the next frame and aligns their features with L1;
- an early DINOv2 alignment loss (Eq. 6) uses the same contrastive form with semantic feature centers.

The total loss is L_CC + lambda_1 L_align + lambda_2 L_DINO (Eq. 7), with lambda_1 = 100 and lambda_2 = 500 (10 for Multi-Human). Training uses Adam, a learning rate decayed from 2.5e-3 to 2e-6, 32-d features, and about 10,000 iterations.

Inference runs HDBSCAN on a small sample of first-frame primitives, represents each cluster by its mean feature, and assigns every primitive to its most similar center. A dynamic-perception filter (Eq. 8) then removes primitives whose velocity, position, and feature similarity all deviate from their cluster.

## Main Claims

Primary claim: Split4D produces more accurate and temporally coherent decomposed 4D segmentation than prior 4D Gaussian segmentation methods while using only per-image masks.

Secondary claims:
- temporally ordered streaming is essential for propagating identities across long sequences;
- explicit linear-motion Gaussian primitives matter for complex dynamic objects;
- 3D tracking and DINOv2 alignment give smaller but measurable robustness gains;
- decoupling feature learning from reconstruction avoids harming reconstruction quality.

## This Gym's Implementation (`method/split4d.py`)

Eqs. 2 and 4–8 and the streaming schedule are implemented as above, with:
- 10,000 iterations;
- 33 iterations per frame visit (the paper's 10k iterations over 300 frames), with views shuffled within a frame;
- the paper's learning-rate schedule.

The paper's lambda values are multiplied by a loss-normalization constant of 0.001. That constant was chosen on development scenes only, so that tracking acts as the mild regularizer the paper describes.

At this reduced scale the first-frame HDBSCAN sample uses up to 600 primitives; the paper's 2–10% of millions corresponds to thousands. The random-order ablation (`baseline/random_order.py`) changes only the sampling order.
