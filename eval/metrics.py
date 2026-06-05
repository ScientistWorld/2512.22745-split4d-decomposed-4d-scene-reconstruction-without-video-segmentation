from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np


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


def evaluate_file(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    method = payload["method"]
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for scene_name, scene in payload["scenes"].items():
        pred = np.asarray(scene["pred_labels"], dtype=np.int64)
        true = np.asarray(scene["object_id"], dtype=np.int64)
        dynamic_object = np.asarray(scene["dynamic_object"], dtype=bool)
        out.setdefault(f"{scene_name}_segmentation", {"results": {}})
        out[f"{scene_name}_segmentation"]["results"][method] = instance_metrics(pred, true, dynamic_object)
    return out


def merge_scores(paths: Iterable[Path]) -> Dict[str, Dict[str, Dict[str, float]]]:
    merged: Dict[str, Dict[str, Dict[str, float]]] = {}
    for path in paths:
        if not path.exists():
            continue
        scores = evaluate_file(path)
        for exp, body in scores.items():
            merged.setdefault(exp, {"results": {}})
            merged[exp]["results"].update(body["results"])
    interaction = merged.get("interaction_motion_segmentation", {}).get("results", {})
    ablation_methods = {
        k: v
        for k, v in interaction.items()
        if k in {"split4d_streaming", "split4d_random_order"}
    }
    if ablation_methods:
        merged["streaming_ablation"] = {"results": ablation_methods}
    return {"experiments": merged}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=["outputs/split4d/results.json", "outputs/baseline/random_order.json"])
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()
    scores = merge_scores(Path(p) for p in args.inputs)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
