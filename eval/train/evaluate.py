from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from data.split4d_synthetic import build_scene


SLICE = "train"
FOCUS_WEIGHT = 1.25
HOLDOUT_WEIGHT = 0.75


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
    for i, pred_label in enumerate(pred_labels):
        for j, true_label in enumerate(true_labels):
            cost[i, j] = -int(((pred == pred_label) & (true == true_label)).sum())
    rows, cols = linear_sum_assignment(cost)
    mapping = {pred_labels[row]: true_labels[col] for row, col in zip(rows, cols)}
    unmatched = max(true_labels) + 1
    return np.asarray([mapping.get(int(label), unmatched) for label in pred], dtype=np.int64)


def split_focus_mask(scene_name: str, true: np.ndarray) -> np.ndarray:
    focus = np.zeros(len(true), dtype=bool)
    for label in sorted(int(x) for x in np.unique(true)):
        indices = np.flatnonzero(true == label)
        keyed = []
        for index in indices:
            digest = hashlib.blake2b(
                f"{SLICE}:{scene_name}:{label}:{int(index)}".encode("utf-8"),
                digest_size=8,
            ).digest()
            keyed.append((int.from_bytes(digest, "little"), int(index)))
        ordered = [index for _key, index in sorted(keyed)]
        focus[ordered[: len(ordered) // 2]] = True
    return focus


def instance_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    dynamic_object: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, float]:
    labels = sorted(int(x) for x in np.unique(true))
    ious = []
    accs = []
    recalls = []
    f1s = []
    dyn_recalls = []
    static_recalls = []
    size_errors = []
    for label in labels:
        predicted = pred == label
        target = true == label
        tp = float(weights[np.logical_and(predicted, target)].sum())
        fp = float(weights[np.logical_and(predicted, ~target)].sum())
        fn = float(weights[np.logical_and(~predicted, target)].sum())
        tn = float(weights[np.logical_and(~predicted, ~target)].sum())
        ious.append(tp / max(tp + fp + fn, 1.0))
        accs.append((tp + tn) / max(tp + tn + fp + fn, 1.0))
        recall = tp / max(tp + fn, 1.0)
        precision = tp / max(tp + fp, 1.0)
        recalls.append(recall)
        f1s.append(2 * precision * recall / max(precision + recall, 1e-12))
        if dynamic_object[label]:
            dyn_recalls.append(recall)
        else:
            static_recalls.append(recall)
        pred_size = float(weights[predicted].sum())
        target_size = float(weights[target].sum())
        size_errors.append(abs(pred_size - target_size) / max(target_size, 1.0))
    return {
        "mIoU": float(np.mean(ious)),
        "mAcc": float(np.mean(accs)),
        "Recall": float(np.mean(recalls)),
        "F1": float(np.mean(f1s)),
        "Recalldyn": float(np.mean(dyn_recalls) if dyn_recalls else np.mean(recalls)),
        "Recallstatic": float(np.mean(static_recalls) if static_recalls else np.mean(recalls)),
        "ClusterSizeError": float(np.mean(size_errors)),
    }


def score_scene(scene_name: str, scene_payload: Mapping[str, object]) -> Dict[str, float]:
    truth_scene = build_scene(scene_name, include_truth=True)
    true_full = np.asarray(truth_scene["object_id"], dtype=np.int64)
    pred_full = np.asarray(scene_payload["pred_labels"], dtype=np.int64)
    if len(pred_full) != len(true_full):
        raise ValueError(
            f"{scene_name}: pred_labels has length {len(pred_full)}, expected {len(true_full)}"
        )
    pred = align_cluster_ids(pred_full, true_full)
    focus = split_focus_mask(scene_name, true_full)
    weights = np.where(focus, FOCUS_WEIGHT, HOLDOUT_WEIGHT).astype(np.float32)
    return instance_metrics(pred, true_full, dynamic_flags(truth_scene), weights)


def evaluate_payload(payload: Mapping[str, object]) -> Dict[str, Dict[str, Dict[str, float]]]:
    method = str(payload["method"])
    scenes = payload.get("scenes", {})
    if not isinstance(scenes, Mapping):
        raise ValueError("Output payload must contain a scenes object.")

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for scene_name, scene_payload in scenes.items():
        if not isinstance(scene_payload, Mapping):
            raise ValueError(f"{scene_name}: scene payload must be an object.")
        metrics = score_scene(str(scene_name), scene_payload)
        if scene_name == "linear_motion" and method != "split4d_random_order":
            out.setdefault("scaled_linear_motion_segmentation", {"results": {}})
            out["scaled_linear_motion_segmentation"]["results"][method] = metrics
        if scene_name == "interaction_motion":
            out.setdefault("scaled_streaming_ablation", {"results": {}})
            out["scaled_streaming_ablation"]["results"][method] = metrics
    return out


def discover_inputs(root: Path = Path("outputs")) -> List[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def merge_scores(paths: Iterable[Path]) -> Dict[str, Dict[str, Dict[str, float]]]:
    merged: Dict[str, Dict[str, Dict[str, float]]] = {}
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        scores = evaluate_payload(payload)
        for experiment, body in scores.items():
            merged.setdefault(experiment, {"results": {}})
            merged[experiment]["results"].update(body["results"])
    return {"experiments": merged}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the visible train slice of Split4D outputs.")
    parser.add_argument("--inputs", nargs="+", default=None, help="Output JSON files to score. Defaults to outputs/**/*.json.")
    parser.add_argument("--output", default="scoring/scores_train.json", help="Score file to write.")
    args = parser.parse_args()

    input_paths = [Path(path) for path in args.inputs] if args.inputs else discover_inputs()
    if not input_paths:
        raise SystemExit("No output JSON files found. Provide --inputs or write method artifacts under outputs/.")

    scores = merge_scores(input_paths)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
