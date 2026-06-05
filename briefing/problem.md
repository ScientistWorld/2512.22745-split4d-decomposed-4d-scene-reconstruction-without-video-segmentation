# Problem Definition

What problem does this paper address? Write this so that someone with no knowledge of the paper can understand what needs to be solved and how success is measured.

Do NOT describe the paper's method or solution here — only the problem itself. Think about if a future scientist agent try to solve this problem from scratch without knowing what approach the paper took. If you leak the solution, you compromise the benchmark.

## Research Question

How can a system recover temporally consistent instance-level segmentation of a dynamic 3D scene from multi-view videos when the available 2D masks are independent per image and do not have stable labels across views or time?

## Why It Matters

Reliable decomposed 4D scene representations enable object-level editing, removal, duplication, and analysis in dynamic scenes with interacting people and objects. The hard case is preserving object identity and boundaries over time without assuming a video tracker has already solved the correspondence problem.

## Success Criteria

A successful solution should assign consistent instance identities to scene elements across time, preserve boundaries under motion and interaction, work for both static and dynamic objects, and outperform methods that rely on inconsistent framewise labels or weak temporal propagation.
