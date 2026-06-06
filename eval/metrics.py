from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from scipy.optimize import linear_sum_assignment

from data.split4d_synthetic import build_scene, iter_observations, position_at, temporal_opacity


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


def normalize_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def render_features(features: np.ndarray, contrib: np.ndarray, weights: np.ndarray) -> np.ndarray:
    rendered = (features[contrib] * weights[..., None]).sum(axis=1)
    return normalize_rows(rendered)


def mask_feature_agreement(scene_name: str, scene: Dict[str, np.ndarray], features: np.ndarray) -> float:
    margins: List[float] = []
    for _t, _view, contrib, weights, labels in iter_observations(scene, streaming=True):
        pixel_features = render_features(features, contrib, weights)
        centers = []
        within = []
        for lab in sorted(int(x) for x in np.unique(labels)):
            local = np.flatnonzero(labels == lab)
            if len(local) < 2:
                continue
            center = normalize_rows(pixel_features[local].mean(axis=0, keepdims=True))[0]
            centers.append(center)
            within.append(float((pixel_features[local] @ center).mean()))
        if len(centers) < 2:
            continue
        centers_arr = np.stack(centers, axis=0)
        sim = centers_arr @ centers_arr.T
        np.fill_diagonal(sim, -np.inf)
        hardest_negative = np.max(sim, axis=1)
        margins.extend(float(pos - neg) for pos, neg in zip(within, hardest_negative))
    if not margins:
        raise ValueError(f"{scene_name}: no observations available for feature agreement")
    return float(np.mean(margins))


def tracking_feature_margin(scene_name: str, scene: Dict[str, np.ndarray], features: np.ndarray) -> float:
    scores: List[float] = []
    frames = int(scene["frames"])
    for t in range(frames - 1):
        active_now = np.flatnonzero(temporal_opacity(scene, t) >= 0.28)
        active_next = np.flatnonzero(temporal_opacity(scene, t + 1) >= 0.28)
        if len(active_now) == 0 or len(active_next) == 0:
            continue
        pos_now = position_at(scene, t)[active_now]
        pos_next = position_at(scene, t + 1)[active_next]
        predicted_next = pos_now + scene["velocity"][active_now]
        scale = np.median(np.linalg.norm(pos_next - np.median(pos_next, axis=0), axis=1)) + 1e-6
        dist = ((predicted_next[:, None, :] - pos_next[None, :, :]) ** 2).sum(axis=2) / (scale**2)
        nearest = dist.argmin(axis=1)
        best = dist[np.arange(len(active_now)), nearest]
        keep = best <= np.quantile(best, 0.45)
        if not np.any(keep):
            continue
        src = active_now[keep]
        dst = active_next[nearest[keep]]
        next_objects = scene["object_id"][active_next]
        for src_index, dst_index in zip(src, dst):
            positive = float(features[src_index] @ features[dst_index])
            negatives = active_next[next_objects != scene["object_id"][src_index]]
            if len(negatives) == 0:
                continue
            hardest_negative = float(np.max(features[src_index] @ features[negatives].T))
            scores.append(positive - hardest_negative)
    if not scores:
        raise ValueError(f"{scene_name}: no tracking pairs available for feature margin")
    return float(np.mean(scores))


def instance_metrics(pred: np.ndarray, true: np.ndarray, dynamic_object: np.ndarray) -> Dict[str, float]:
    labels = sorted(int(x) for x in np.unique(true))
    ious = []
    accs = []
    recalls = []
    f1s = []
    dyn_recalls = []
    static_recalls = []
    size_errors = []
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
        else:
            static_recalls.append(recall)
        size_errors.append(abs(float(p.sum()) - float(y.sum())) / max(float(y.sum()), 1.0))
    return {
        "mIoU": float(np.mean(ious)),
        "mAcc": float(np.mean(accs)),
        "Recall": float(np.mean(recalls)),
        "F1": float(np.mean(f1s)),
        "Recalldyn": float(np.mean(dyn_recalls) if dyn_recalls else np.mean(recalls)),
        "Recallstatic": float(np.mean(static_recalls) if static_recalls else np.mean(recalls)),
        "ClusterSizeError": float(np.mean(size_errors)),
    }


def score_scene(scene_name: str, scene_payload: Dict[str, object]) -> Dict[str, float]:
    truth_scene = build_scene(scene_name, include_truth=True)
    true = np.asarray(truth_scene["object_id"], dtype=np.int64)
    pred_raw = np.asarray(scene_payload["pred_labels"], dtype=np.int64)
    features = normalize_rows(np.asarray(scene_payload["features"], dtype=np.float32))
    if len(pred_raw) != len(true):
        raise ValueError(f"{scene_name}: pred_labels has length {len(pred_raw)}, expected {len(true)}")
    if features.shape[0] != len(true):
        raise ValueError(f"{scene_name}: features has length {features.shape[0]}, expected {len(true)}")
    pred = align_cluster_ids(pred_raw, true)
    metrics = instance_metrics(pred, true, dynamic_flags(truth_scene))
    metrics["MaskFeatureAgreement"] = mask_feature_agreement(scene_name, truth_scene, features)
    metrics["TrackingFeatureMargin"] = tracking_feature_margin(scene_name, truth_scene, features)
    return metrics


def evaluate_file(path: Path, include_exploratory: bool = False) -> Dict[str, Dict[str, Dict[str, float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    method = payload["method"]
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for scene_name, scene in payload["scenes"].items():
        metrics = score_scene(scene_name, scene)
        if scene_name == "linear_motion" and method != "split4d_random_order":
            out.setdefault("scaled_linear_motion_segmentation", {"results": {}})
            out["scaled_linear_motion_segmentation"]["results"][method] = metrics
        if include_exploratory and scene_name == "interaction_motion" and method != "split4d_random_order":
            out.setdefault("scaled_interaction_motion_segmentation", {"results": {}})
            out["scaled_interaction_motion_segmentation"]["results"][method] = metrics
        if scene_name == "interaction_motion" and method in {"split4d_streaming", "split4d_random_order"}:
            out.setdefault("scaled_streaming_ablation", {"results": {}})
            out["scaled_streaming_ablation"]["results"][method] = metrics
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
