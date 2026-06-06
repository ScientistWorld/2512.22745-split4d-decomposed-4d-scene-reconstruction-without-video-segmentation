# Problem Definition

## Research Question

How can a system recover temporally consistent instance-level segmentation of a dynamic 3D scene from multi-view videos when the available 2D masks are independent per image and do not have stable labels across views or time?

## Why It Matters

Reliable decomposed 4D scene representations enable object-level editing, removal, duplication, and analysis in dynamic scenes with interacting people and objects. The hard case is preserving object identity and boundaries over time without assuming a video tracker has already solved the correspondence problem.

## Success Criteria

A successful solution should assign consistent instance identities to scene elements across time, preserve boundaries under motion and interaction, work for both static and dynamic objects, and outperform simple references that fail to maintain coherent identities through motion.
