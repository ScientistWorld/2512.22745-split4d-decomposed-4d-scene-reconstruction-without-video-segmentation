"""Variant: replace first-frame HDBSCAN inference by k-means over all primitives.

Uses saved Split4D features (one or several runs, concatenated) and an
instance count estimated from the public masks (median number of masks with at
least 3 pixels per image). No answers are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from baseline.simple_grouping import estimate_instances
from data.split4d_scenes import list_scenes, load_scene


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("features", nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--method-name", default="split4d_kmeans_inference")
    args = p.parse_args()
    stores = [np.load(f) for f in args.features]
    out = {"method": args.method_name, "scenes": {}}
    for s in list_scenes():
        sid = s["id"]
        feats = []
        for z in stores:
            f = z[sid].astype(np.float32)
            feats.append(f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8))
        f = np.concatenate(feats, axis=1)
        k = estimate_instances(load_scene(sid))
        labels = KMeans(k, n_init=20, random_state=0).fit_predict(f)
        out["scenes"][sid] = {"pred_labels": [int(x) for x in labels], "info": {"k_estimate": k}}
        print(f"{sid}: k={k}", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main()
