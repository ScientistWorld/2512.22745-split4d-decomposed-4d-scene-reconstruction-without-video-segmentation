from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from data.split4d_synthetic import build_scene


def dynamic_flags(scene: Dict[str, np.ndarray]) -> np.ndarray:
    speed = np.linalg.norm(scene["velocity"], axis=1)
    obj_speed = np.zeros(int(scene["instances"]), dtype=np.float32)
    for inst in range(int(scene["instances"])):
        obj_speed[inst] = float(np.median(speed[scene["object_id"] == inst]))
    return obj_speed > np.median(obj_speed) * 0.9


def align_cluster_ids(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred_labels = sorted(int(x) for x in np.unique(pred))
    true_labels = sorted(int(x) for x in np.unique(true))
    if not pred_labels or not true_labels:
        return pred
    cost = np.zeros((len(pred_labels), len(true_labels)), dtype=np.int64)
    for i, p in enumerate(pred_labels):
        for j, t in enumerate(true_labels):
            cost[i, j] = -int(((pred == p) & (true == t)).sum())
    rows, cols = linear_sum_assignment(cost)
    mapping = {pred_labels[r]: true_labels[c] for r, c in zip(rows, cols)}
    unmatched = max(true_labels) + 1
    return np.asarray([mapping.get(int(p), unmatched) for p in pred], dtype=np.int64)


def instance_metrics(pred: np.ndarray, true: np.ndarray, dynamic_object: np.ndarray) -> Dict[str, float]:
    labels = sorted(int(x) for x in np.unique(true))
    ious = []
    accs = []
    recalls = []
    f1s = []
    dyn_recalls = []
    for lab in labels:
        p = pred == lab
        y = true == lab
        tp = float(np.logical_and(p, y).sum())
        fp = float(np.logical_and(p, ~y).sum())
        fn = float(np.logical_and(~p, y).sum())
        tn = float(np.logical_and(~p, ~y).sum())
        ious.append(tp / max(tp + fp + fn, 1.0))
        accs.append((tp + tn) / max(tp + tn + fp + fn, 1.0))
        recall = tp / max(tp + fn, 1.0)
        precision = tp / max(tp + fp, 1.0)
        recalls.append(recall)
        f1s.append(2 * precision * recall / max(precision + recall, 1e-12))
        if dynamic_object[lab]:
            dyn_recalls.append(recall)
    return {
        "mIoU": float(np.mean(ious)),
        "mAcc": float(np.mean(accs)),
        "Recall": float(np.mean(recalls)),
        "F1": float(np.mean(f1s)),
        "Recalldyn": float(np.mean(dyn_recalls) if dyn_recalls else np.mean(recalls)),
    }


def evaluate_file(path: Path, include_exploratory: bool = False) -> Dict[str, Dict[str, Dict[str, float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    method = payload["method"]
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for scene_name, scene in payload["scenes"].items():
        truth_scene = build_scene(scene_name, include_truth=True)
        true = np.asarray(truth_scene["object_id"], dtype=np.int64)
        pred = align_cluster_ids(np.asarray(scene["pred_labels"], dtype=np.int64), true)
        dynamic_object = dynamic_flags(truth_scene)
        if scene_name == "linear_motion" and method != "split4d_random_order":
            out.setdefault("linear_motion_segmentation", {"results": {}})
            out["linear_motion_segmentation"]["results"][method] = instance_metrics(pred, true, dynamic_object)
        if include_exploratory and scene_name == "interaction_motion" and method != "split4d_random_order":
            out.setdefault("interaction_motion_segmentation", {"results": {}})
            out["interaction_motion_segmentation"]["results"][method] = instance_metrics(pred, true, dynamic_object)
        if scene_name == "interaction_motion" and method in {"split4d_streaming", "split4d_random_order"}:
            out.setdefault("streaming_ablation", {"results": {}})
            out["streaming_ablation"]["results"][method] = instance_metrics(pred, true, dynamic_object)
    return out


def merge_scores(paths: Iterable[Path], include_exploratory: bool = False) -> Dict[str, Dict[str, Dict[str, float]]]:
    merged: Dict[str, Dict[str, Dict[str, float]]] = {}
    for path in paths:
        if not path.exists():
            continue
        scores = evaluate_file(path, include_exploratory=include_exploratory)
        for exp, body in scores.items():
            merged.setdefault(exp, {"results": {}})
            merged[exp]["results"].update(body["results"])
    return {"experiments": merged}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=["outputs/split4d/results.json", "outputs/baseline/random_order.json"])
    parser.add_argument("--output", default="scoring/scores.json")
    parser.add_argument("--include-exploratory", action="store_true", help="Also report underpowered interaction and ablation diagnostics.")
    args = parser.parse_args()
    scores = merge_scores((Path(p) for p in args.inputs), include_exploratory=args.include_exploratory)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
