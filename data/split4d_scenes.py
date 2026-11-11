"""Public loader for the Split4D benchmark scenes.

Each scene file under ``data/scenes/`` holds only method inputs:

* fixed Freetime Gaussian primitives: ``mean_xyz`` (position at ``mu_t``),
  ``mu_t`` (temporal center), ``duration`` (temporal scale), ``velocity``
  (linear 3D motion), ``opacity``, ``scale`` (2D footprint), and ``semantic``
  (a 12-d pre-computed semantic descriptor per primitive, standing in for
  frozen image-foundation-model features);
* training-view observations: for every (frame, view) image a set of sampled
  pixels, each with its alpha-blended contributor primitives (front to back),
  blend weights, and an instance-mask id from an automatic per-image segmenter.
  Mask ids are arbitrary integers chosen independently for every image; ``-1``
  marks pixels without a mask. Masks contain the segmenter's own errors.

There are no instance answers in these files, and primitive rows are in random
order. Test-view renderings used for scoring are not public.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

SCENE_DIR = Path(__file__).resolve().parent / "scenes"
ACTIVE_OPACITY = 0.22


def scene_dir(root: Optional[str] = None) -> Path:
    if root:
        return Path(root)
    return Path(os.environ.get("SPLIT4D_SCENE_DIR", str(SCENE_DIR)))


def load_manifest(root: Optional[str] = None) -> Dict[str, object]:
    return json.loads((scene_dir(root) / "manifest.json").read_text(encoding="utf-8"))


def list_scenes(root: Optional[str] = None, split: Optional[str] = None, profile: Optional[str] = None) -> List[Dict[str, object]]:
    scenes = load_manifest(root)["scenes"]
    return [s for s in scenes if (split is None or s["split"] == split) and (profile is None or s["profile"] == profile)]


def load_scene(scene_id: str, root: Optional[str] = None) -> Dict[str, np.ndarray]:
    path = scene_dir(root) / f"{scene_id}.npz"
    with np.load(path, allow_pickle=False) as z:
        scene = {k: z[k] for k in z.files}
    scene["pix_weight"] = scene.pop("pix_weight_u8").astype(np.float32) / 255.0
    scene["pix_contrib"] = scene["pix_contrib"].astype(np.int64)
    scene["pix_label"] = scene["pix_label"].astype(np.int64)
    scene["id"] = np.array(str(scene_id))
    return scene


def num_primitives(scene: Dict[str, np.ndarray]) -> int:
    return int(len(scene["mu_t"]))


def position_at(scene: Dict[str, np.ndarray], t: float) -> np.ndarray:
    """Eq. 1: mu_x(t) = mu_x + v (t - mu_t)."""

    return scene["mean_xyz"] + scene["velocity"] * (float(t) - scene["mu_t"])[:, None]


def temporal_opacity(scene: Dict[str, np.ndarray], t: float) -> np.ndarray:
    return np.exp(-0.5 * ((float(t) - scene["mu_t"]) / scene["duration"]) ** 2)


def active_indices(scene: Dict[str, np.ndarray], t: float, min_opacity: float = ACTIVE_OPACITY) -> np.ndarray:
    return np.flatnonzero(temporal_opacity(scene, t) >= min_opacity)


def observation(scene: Dict[str, np.ndarray], k: int) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = int(scene["obs_offsets"][k]), int(scene["obs_offsets"][k + 1])
    return (int(scene["obs_t"][k]), int(scene["obs_view"][k]), scene["pix_contrib"][lo:hi],
            scene["pix_weight"][lo:hi], scene["pix_label"][lo:hi])


def iter_observations(scene: Dict[str, np.ndarray]) -> Iterator[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]]:
    for k in range(len(scene["obs_t"])):
        yield observation(scene, k)
