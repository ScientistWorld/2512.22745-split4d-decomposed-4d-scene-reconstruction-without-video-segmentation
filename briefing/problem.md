# Problem Definition

## Research Question

How can a system recover temporally consistent instance-level segmentation of a dynamic 3D scene from multi-view videos when the available 2D masks come from an automatic per-image segmenter: independent per image, without stable labels across views or time, and with the segmenter's own errors?

## Setting in This Benchmark

Each scene is given as a fixed, already reconstructed dynamic 3D representation: a few hundred to about a thousand small 4D primitives. Each primitive has a position at its own time center, a temporal extent, a linear velocity, an opacity, an image footprint, and a semantic descriptor (a stand-in for frozen image-foundation-model features). Training views are given as sampled image pixels. For each pixel you get the primitives that blend into it, front to back, with their blend weights, and an instance-mask id from the automatic segmenter. Mask ids are arbitrary per image. The segmenter sometimes merges objects that touch, splits an object into parts, mislabels boundary pixels, or drops tiny masks.

The task is to assign every primitive of a scene to an instance, so that the instance masks rendered in held-out views stay correct and keep the same identity over the whole sequence. Some objects are static and some move, handle each other, or come into contact.

## Why It Matters

Reliable decomposed 4D scene representations enable object-level editing, removal, duplication, and analysis in dynamic scenes with interacting people and objects. The hard case is preserving object identity and boundaries over time and through interactions, without a video tracker having already solved correspondence.

## Success Criteria

A successful solution assigns consistent instance identities across the whole sequence. It preserves boundaries under motion and contact, works for both static and moving objects (moving objects are the harder part), and clearly outperforms simple references that do not maintain coherent identities through motion.
