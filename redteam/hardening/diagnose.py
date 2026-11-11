"""Diagnostics on the preset-sweep artifacts (dev scenes only; light numpy work).

For each preset's dev scenes: per-object recall/precision of the streaming and
random-order outputs, recall of dynamic objects early/mid/late in the sequence
(feature drift), filter removals, and alternative inference settings applied
to the same saved streaming features (is the weakness in features or inference?).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from data.split4d_scenes import load_scene, temporal_opacity
from eval.train.evaluate import load_truth, pixel_predictions, scene_metrics
from method.split4d import build_parser, infer_labels

ROOT = Path(__file__).resolve().parents[2]


def per_object(labels, truth):
    pred = pixel_predictions(labels, truth["test_contrib"], truth["test_weight"])
    true = truth["test_visible"].astype(np.int64)
    tt = truth["test_t"].astype(np.int64)
    off = truth["test_offsets"]
    frame = np.repeat(tt, np.diff(off))
    k = len(truth["object_dynamic"])
    ids = np.unique(pred[pred >= 0])
    conf = np.zeros((len(ids), k), dtype=np.int64)
    if len(ids):
        np.add.at(conf, (np.searchsorted(ids, pred[pred >= 0]), true[pred >= 0]), 1)
    match = {}
    if len(ids):
        r, c = linear_sum_assignment(-conf)
        match = {int(c_): int(ids[r_]) for r_, c_ in zip(r, c)}
    out = []
    for o in range(k):
        mask = true == o
        lab = match.get(o)
        hit = (pred == lab) & mask if lab is not None else np.zeros_like(mask)
        rec = hit.sum() / max(mask.sum(), 1)
        prec = hit.sum() / max(((pred == lab).sum() if lab is not None else 0), 1)
        thirds = []
        for lo, hi in ((0, 20), (20, 40), (40, 60)):
            m = mask & (frame >= lo) & (frame < hi)
            thirds.append(round(float(((pred == lab) & m).sum() / max(m.sum(), 1)) if lab is not None else 0.0, 3))
        out.append({"category": str(truth["object_category"][o]), "dynamic": bool(truth["object_dynamic"][o]),
                    "recall": round(float(rec), 3), "precision": round(float(prec), 3), "recall_by_third": thirds,
                    "unassigned_frac": round(float((mask & (pred < 0)).sum() / max(mask.sum(), 1)), 3)})
    return out, int(len(ids))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", nargs="+", default=["contact_mid", "contact_long", "overlap_mid", "overlap_long"])
    args = ap.parse_args()
    report = {}
    base_args = build_parser().parse_args([])
    for p in args.presets:
        d = ROOT / "tmp" / "sweep" / p
        if not (d / "outputs" / "streaming.json").exists():
            continue
        exps, truth = load_truth(d / "truth")
        rep = {}
        outs = {n: json.loads((d / "outputs" / f"{n}.json").read_text()) for n in ("streaming", "random", "mask_graph_spectral")}
        feats = np.load(d / "outputs" / "streaming.features.npz")
        for exp, sids in exps.items():
            er = {"objects": {}, "alt_inference_mIoU": {}}
            alts = {"paper_default": {}, "no_filter": {"filter": False}, "eps0": {"hdbscan_epsilon": 0.0},
                    "min_cluster_10": {"hdbscan_min_cluster_size": 10, "hdbscan_min_samples": 10},
                    "all_frames_sample": "allframes", "kmeans_true_k": "kmeans"}
            for name in alts:
                er["alt_inference_mIoU"][name] = []
            for sid in sids:
                scene = load_scene(sid, str(d / "scenes"))
                for n, payload in outs.items():
                    lab = np.asarray(payload["scenes"][sid]["pred_labels"], dtype=np.int64)
                    objs, nlab = per_object(lab, truth[sid])
                    er["objects"].setdefault(n, {})[sid] = {"n_labels": nlab, "filtered_frac": round(float((lab < 0).mean()), 3),
                                                             "objects": objs}
                f = feats[sid].astype(np.float32)
                f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)
                for name, spec in alts.items():
                    if spec == "kmeans":
                        k = len(truth[sid]["object_dynamic"])
                        lab = KMeans(k, n_init=10, random_state=0).fit_predict(f)
                    elif spec == "allframes":
                        a = argparse.Namespace(**vars(base_args))
                        # cluster a sample drawn from all frames instead of the first frame
                        saved = scene["mu_t"].copy()
                        rng = np.random.default_rng(0)
                        lab, _ = infer_labels({**scene, "mu_t": np.zeros_like(saved), "duration": np.full_like(saved, 1e6)},
                                              f, a, 0)
                    else:
                        a = argparse.Namespace(**{**vars(base_args), **spec})
                        lab, _ = infer_labels(scene, f, a, 0)
                    er["alt_inference_mIoU"][name].append(scene_metrics(np.asarray(lab, dtype=np.int64), truth[sid])["mIoU"])
            er["alt_inference_mIoU"] = {k: round(statistics.mean(v), 3) for k, v in er["alt_inference_mIoU"].items()}
            rep[exp] = er
        report[p] = rep
        print(p, json.dumps({e: r["alt_inference_mIoU"] for e, r in rep.items()}), flush=True)
    out = ROOT / "redteam" / "hardening" / "diagnostics.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    # compact per-object summary
    for p, rep in report.items():
        for exp, er in rep.items():
            for n in ("streaming", "random", "mask_graph_spectral"):
                rows = er["objects"][n]
                cats = {}
                for sid, body in rows.items():
                    for o in body["objects"]:
                        cats.setdefault(o["category"], []).append((o["recall"], o["precision"], o["recall_by_third"], o["unassigned_frac"]))
                summ = {c: (round(statistics.mean(x[0] for x in v), 2), round(statistics.mean(x[1] for x in v), 2),
                            [round(statistics.mean(x[2][i] for x in v), 2) for i in range(3)], round(statistics.mean(x[3] for x in v), 3))
                        for c, v in cats.items()}
                nl = [b["n_labels"] for b in rows.values()]
                ff = [b["filtered_frac"] for b in rows.values()]
                print(p, exp[7:18], n, "labels", nl, "filtered", ff)
                for c, v in summ.items():
                    print("     ", c, "recall", v[0], "prec", v[1], "by-third", v[2], "unassigned", v[3])


if __name__ == "__main__":
    main()
