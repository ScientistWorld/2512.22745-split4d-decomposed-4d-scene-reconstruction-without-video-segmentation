# Experiment Overview

## scaled_dynamic_scene_segmentation (weight 0.4)
Checks dynamic-scene instance decomposition in kitchen-like scenes: one person handling objects, a pet, and static furniture. It asks whether a method can recover every object's identity over a whole multi-view sequence from per-image automatic masks, keeping moving and handled objects correct as well as large static ones. Scored on 10 hidden test scenes in held-out views. The 4 development scenes support model selection.

## scaled_interaction_segmentation (weight 0.6)
Checks identity preservation where it is hardest: several similar-looking people who meet and touch, and a ball passed between them, where the automatic masks often merge objects in contact. It asks whether identities survive interactions over time instead of being swapped or merged. This experiment also carries the benchmark's constraint: a run must beat a fixed reference artifact on F1 and moving-object recall here, or the whole run is scored as failed. Scored on 10 hidden test scenes.

The paper's own full-scale tables are kept in `scoring/reference.json` under `_paper_reported` for provenance; they are not experiments of this benchmark.
