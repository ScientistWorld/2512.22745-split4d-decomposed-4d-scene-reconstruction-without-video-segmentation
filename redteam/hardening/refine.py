"""Variant: mask-vote refinement of an existing method's primitive labels.

Starting from a method output, every training image votes: each mask gives its
pixels' highest-weight primitives the mask's current label histogram (from the
existing labels of the primitives it covers). A primitive then takes the label
with the largest vote share, with a prior on its current label. Two rounds.
Uses only public scene inputs and the given labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.split4d_scenes import iter_observations, load_scene


def refine(scene, labels: np.ndarray, rounds: int = 2, prior: float = 0.35) -> np.ndarray:
    labels = labels.copy()
    obs = list(iter_observations(scene))
    for _ in range(rounds):
        n_lab = int(labels.max()) + 1
        if n_lab <= 0:
            return labels
        votes = np.zeros((len(labels), n_lab))
        for _t, _v, contrib, weight, lab in obs:
            ok = lab >= 0
            dom = contrib[ok][np.arange(int(ok.sum())), weight[ok].argmax(axis=1)]
            lab = lab[ok]
            cur = labels[dom]
            for m in np.unique(lab):
                members = dom[lab == m]
                hist = np.bincount(cur[(lab == m) & (cur >= 0)], minlength=n_lab).astype(float)
                if hist.sum() == 0:
                    continue
                np.add.at(votes, members, hist / hist.sum())
        total = votes.sum(axis=1, keepdims=True)
        voted = total[:, 0] > 0
        share = np.where(voted[:, None], votes / np.maximum(total, 1e-12), 0.0)
        own = labels >= 0
        share[np.flatnonzero(own), labels[own]] += prior
        labels = np.where(voted, share.argmax(axis=1), labels)
    return labels.astype(np.int64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("source")
    p.add_argument("--output", required=True)
    p.add_argument("--method-name", default="split4d_mask_vote_refine")
    args = p.parse_args()
    src = json.loads(Path(args.source).read_text())
    out = {"method": args.method_name, "scenes": {}}
    for sid, body in src["scenes"].items():
        scene = load_scene(sid)
        new = refine(scene, np.asarray(body["pred_labels"], dtype=np.int64))
        out["scenes"][sid] = {"pred_labels": [int(x) for x in new]}
        print(f"refined {sid}", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main()
