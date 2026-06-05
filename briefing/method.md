# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

Split4D learns a decomposed 4D feature field for dynamic Gaussian scenes without requiring video segmentation or globally consistent 2D instance labels.

## Approach

The scene is represented as Freetime FeatureGS: Gaussian primitives with 4D centers, temporal extent, opacity, appearance, linear 3D velocity, and a learnable instance feature vector. The pretrained geometric reconstruction is kept fixed while the instance features are optimized.

For each view and time, features are rendered or gathered for primitives visible in an image mask. A contrastive clustering objective pulls sampled features inside the same 2D mask toward that mask's feature center and pushes them away from centers of other masks. Since individual Gaussian primitives move and are reused locally across neighboring times, this transfers feature identity through time even when mask labels are inconsistent.

The paper's key training strategy is streaming sampling: observations are processed in temporal order so local correspondences propagate gradually instead of forcing far-apart frames into unrelated local optima. Two regularizers improve robustness: a 3D tracking alignment loss matches disappearing primitives to nearby primitives in the next frame, and an early DINOv2 feature-alignment loss supplies semantic structure. Inference clusters a small sample of first-frame primitives and then filters assignments using feature similarity, velocity, and spatial deviations.

## Main Claims

Primary claim: Split4D produces more accurate and temporally coherent decomposed 4D segmentation than prior 4D Gaussian segmentation methods while using only per-image masks.

Secondary claims: temporally ordered streaming is essential for propagating identities across long sequences; explicit linear-motion Gaussian primitives are important for complex dynamic objects; 3D tracking and DINOv2 alignment provide smaller but measurable robustness gains; decoupling feature learning from reconstruction avoids harming reconstruction quality.
