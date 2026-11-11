"""Simple grouping baselines that use the public scene inputs without feature learning.

* ``geometry_kmeans``: k-means on primitive positions extrapolated to frame 0
  with each primitive's linear motion (no masks).
* ``mask_graph_spectral``: global co-membership graph over primitives (pixels
  attributed to their highest-weight contributor; edge weight = fraction of
  shared images in which both fall in the same mask, minus 0.5), clustered by
  spectral clustering.
* ``mask_tracker``: video-tracker-style label propagation; images are visited in
  temporal order and each mask is matched (Hungarian) to existing global labels
  through the votes of its primitives; unmatched masks start new labels.
* ``semantic_kmeans``: k-means on the public semantic descriptors.

The instance count, when needed, is estimated from the median number of masks
per image (it is not given by the benchmark).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, SpectralClustering

from data.split4d_scenes import iter_observations, list_scenes, load_scene, position_at

METHODS = ("geometry_kmeans", "mask_graph_spectral", "mask_tracker", "semantic_kmeans")


def estimate_instances(scene: Dict[str, np.ndarray]) -> int:
    counts = []
    for _t, _v, _c, _w, lab in iter_observations(scene):
        ids, n = np.unique(lab[lab >= 0], return_counts=True)
        counts.append(int((n >= 3).sum()))
    return max(2, int(round(float(np.median(counts)))))


def dominant(contrib: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return contrib[np.arange(len(contrib)), weight.argmax(axis=1)]


def geometry_kmeans(scene: Dict[str, np.ndarray], k: int, seed: int) -> np.ndarray:
    return KMeans(k, n_init=20, random_state=seed).fit_predict(position_at(scene, 0))


def semantic_kmeans(scene: Dict[str, np.ndarray], k: int, seed: int) -> np.ndarray:
    return KMeans(k, n_init=20, random_state=seed).fit_predict(scene["semantic"])


def mask_graph_spectral(scene: Dict[str, np.ndarray], k: int, seed: int) -> np.ndarray:
    n = len(scene["mu_t"])
    same = np.zeros((n, n), dtype=np.float32)
    total = np.zeros((n, n), dtype=np.float32)
    for _t, _v, contrib, weight, lab in iter_observations(scene):
        ok = lab >= 0
        dom = dominant(contrib[ok], weight[ok])
        lab = lab[ok]
        prims, first = np.unique(dom, return_index=True)
        # majority mask per primitive within the image
        prim_lab = np.empty(len(prims), dtype=np.int64)
        for i, p in enumerate(prims):
            vals, cnt = np.unique(lab[dom == p], return_counts=True)
            prim_lab[i] = vals[cnt.argmax()]
        ix = np.ix_(prims, prims)
        total[ix] += 1.0
        same[ix] += (prim_lab[:, None] == prim_lab[None, :]).astype(np.float32)
    affinity = np.maximum(same / np.maximum(total, 1.0) - 0.5, 0.0)
    affinity[total < 2] = 0.0
    np.fill_diagonal(affinity, 1.0)
    return SpectralClustering(k, affinity="precomputed", random_state=seed, assign_labels="kmeans",
                              n_init=20).fit_predict(affinity)


def mask_tracker(scene: Dict[str, np.ndarray], k: int, seed: int, max_labels: int = 64) -> np.ndarray:
    n = len(scene["mu_t"])
    votes = np.zeros((n, max_labels), dtype=np.float64)
    used = 0
    order = np.lexsort((scene["obs_view"], scene["obs_t"]))
    obs = list(iter_observations(scene))
    for idx in order:
        _t, _v, contrib, weight, lab = obs[int(idx)]
        ok = lab >= 0
        dom = dominant(contrib[ok], weight[ok])
        lab = lab[ok]
        masks = np.unique(lab)
        if used == 0:
            for m in masks[:max_labels]:
                np.add.at(votes[:, used], dom[lab == m], 1.0)
                used += 1
            continue
        norm = votes[:, :used] / np.maximum(votes[:, :used].sum(axis=1, keepdims=True), 1e-9)
        overlap = np.stack([norm[dom[lab == m]].sum(axis=0) for m in masks])
        sizes = np.array([(lab == m).sum() for m in masks], dtype=float)
        rows, cols = linear_sum_assignment(-overlap)
        assign = {}
        for r, c in zip(rows, cols):
            if overlap[r, c] >= 0.25 * sizes[r]:
                assign[int(r)] = int(c)
        for r, m in enumerate(masks):
            c = assign.get(r)
            if c is None:
                if used >= max_labels:
                    c = int(overlap[r].argmax())
                else:
                    c = used
                    used += 1
            np.add.at(votes[:, c], dom[lab == m], 1.0)
    labels = votes[:, :used].argmax(axis=1)
    labels[votes[:, :used].sum(axis=1) == 0] = -1
    return labels.astype(np.int64)


def run(method: str, scene_ids: List[str], scene_dir: str | None, seed: int) -> Dict[str, object]:
    fn = {"geometry_kmeans": geometry_kmeans, "mask_graph_spectral": mask_graph_spectral,
          "mask_tracker": mask_tracker, "semantic_kmeans": semantic_kmeans}[method]
    payload: Dict[str, object] = {"method": method, "scenes": {}}
    for sid in scene_ids:
        scene = load_scene(sid, scene_dir)
        k = estimate_instances(scene)
        labels = fn(scene, k, seed)
        payload["scenes"][sid] = {"pred_labels": [int(x) for x in labels], "info": {"k_estimate": k}}
        print(f"{method} {sid}: k={k}", flush=True)
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("method", choices=METHODS)
    p.add_argument("--output", required=True)
    p.add_argument("--scene-dir", default=None)
    p.add_argument("--split", default=None, choices=["dev", "test"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    ids = [s["id"] for s in list_scenes(args.scene_dir, split=args.split)]
    payload = run(args.method, ids, args.scene_dir, args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
