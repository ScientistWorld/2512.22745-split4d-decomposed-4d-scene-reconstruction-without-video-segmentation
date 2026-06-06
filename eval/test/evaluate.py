from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set

import numpy as np
from scipy.optimize import linear_sum_assignment

from data.split4d_synthetic import build_scene, iter_observations, position_at, temporal_opacity


SLICE = "test"


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


def slice_mask(scene_name: str, true: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(true), dtype=bool)
    for label in sorted(int(x) for x in np.unique(true)):
        indices = np.flatnonzero(true == label)
        keyed = []
        for index in indices:
            digest = hashlib.blake2b(
                f"split4d-eval:{scene_name}:{label}:{int(index)}".encode("utf-8"),
                digest_size=8,
            ).digest()
            keyed.append((int.from_bytes(digest, "little"), int(index)))
        ordered = [index for _key, index in sorted(keyed)]
        half = len(ordered) // 2
        selected = ordered[:half] if SLICE == "train" else ordered[half:]
        mask[selected] = True
    return mask


def observation_in_slice(scene_name: str, t: int, view: int) -> bool:
    digest = hashlib.blake2b(
        f"split4d-observation-eval:{scene_name}:{int(t)}:{int(view)}".encode("utf-8"),
        digest_size=8,
    ).digest()
    bucket = int.from_bytes(digest, "little") % 2
    return bucket == (0 if SLICE == "train" else 1)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def render_features(features: np.ndarray, contrib: np.ndarray, weights: np.ndarray) -> np.ndarray:
    rendered = (features[contrib] * weights[..., None]).sum(axis=1)
    return normalize_rows(rendered)


def mask_feature_agreement(scene_name: str, scene: Dict[str, np.ndarray], features: np.ndarray) -> float:
    margins: List[float] = []
    for t, view, contrib, weights, labels in iter_observations(scene, streaming=True):
        if not observation_in_slice(scene_name, int(t), int(view)):
            continue
        pixel_features = render_features(features, contrib, weights)
        centers = []
        within = []
        for label in sorted(int(x) for x in np.unique(labels)):
            local = np.flatnonzero(labels == label)
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
        raise ValueError(f"{scene_name}: no observations available for {SLICE} feature agreement")
    return float(np.mean(margins))


def tracking_feature_margin(scene_name: str, scene: Dict[str, np.ndarray], features: np.ndarray) -> float:
    scores: List[float] = []
    frames = int(scene["frames"])
    for t in range(frames - 1):
        if not observation_in_slice(scene_name, t, 0):
            continue
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
        raise ValueError(f"{scene_name}: no tracking pairs available for {SLICE} feature margin")
    return float(np.mean(scores))


def instance_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    dynamic_object: np.ndarray,
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
        tp = float(np.logical_and(predicted, target).sum())
        fp = float(np.logical_and(predicted, ~target).sum())
        fn = float(np.logical_and(~predicted, target).sum())
        tn = float(np.logical_and(~predicted, ~target).sum())
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
        pred_size = float(predicted.sum())
        target_size = float(target.sum())
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
    features = normalize_rows(np.asarray(scene_payload["features"], dtype=np.float32))
    if len(pred_full) != len(true_full):
        raise ValueError(
            f"{scene_name}: pred_labels has length {len(pred_full)}, expected {len(true_full)}"
        )
    if features.shape[0] != len(true_full):
        raise ValueError(
            f"{scene_name}: features has length {features.shape[0]}, expected {len(true_full)}"
        )
    pred = align_cluster_ids(pred_full, true_full)
    keep = slice_mask(scene_name, true_full)
    if not np.any(keep):
        raise ValueError(f"{scene_name}: empty {SLICE} slice")
    metrics = instance_metrics(pred[keep], true_full[keep], dynamic_flags(truth_scene))
    metrics["MaskFeatureAgreement"] = mask_feature_agreement(scene_name, truth_scene, features)
    metrics["TrackingFeatureMargin"] = tracking_feature_margin(scene_name, truth_scene, features)
    return metrics


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


def expected_methods_from_train_scores(path: Path) -> Optional[Dict[str, Set[str]]]:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    experiments = payload.get("experiments", {})
    expected: Dict[str, Set[str]] = {}
    if not isinstance(experiments, Mapping):
        raise ValueError(f"{path}: expected top-level experiments object.")
    for experiment, body in experiments.items():
        if not isinstance(body, Mapping):
            continue
        results = body.get("results", {})
        if isinstance(results, Mapping):
            expected[str(experiment)] = {str(method) for method in results}
    return expected


def filter_to_train_methods(
    scores: Dict[str, Dict[str, Dict[str, float]]],
    expected: Optional[Dict[str, Set[str]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    if expected is None:
        return scores

    filtered: Dict[str, Dict[str, Dict[str, float]]] = {}
    missing: Dict[str, List[str]] = {}
    for experiment, methods in expected.items():
        available = scores.get(experiment, {"results": {}}).get("results", {})
        absent = sorted(method for method in methods if method not in available)
        if absent:
            missing[experiment] = absent
            continue
        filtered[experiment] = {
            "results": {
                method: available[method]
                for method in sorted(methods)
            }
        }
    if missing:
        details = "; ".join(f"{exp}: {methods}" for exp, methods in sorted(missing.items()))
        raise SystemExit(f"Missing test outputs for train-scored methods: {details}")
    return filtered


def merge_scores(
    paths: Iterable[Path],
    expected: Optional[Dict[str, Set[str]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    merged: Dict[str, Dict[str, Dict[str, float]]] = {}
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        scores = evaluate_payload(payload)
        for experiment, body in scores.items():
            merged.setdefault(experiment, {"results": {}})
            merged[experiment]["results"].update(body["results"])
    return {"experiments": filter_to_train_methods(merged, expected)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the held-out test slice of Split4D outputs.")
    parser.add_argument("--inputs", nargs="+", default=None, help="Output JSON files to score. Defaults to outputs/**/*.json.")
    parser.add_argument("--output", default="scoring/scores_test.json", help="Score file to write.")
    parser.add_argument(
        "--train-scores",
        default="scoring/scores_train.json",
        help="Train score file whose experiment/method keys must be mirrored.",
    )
    args = parser.parse_args()

    input_paths = [Path(path) for path in args.inputs] if args.inputs else discover_inputs()
    if not input_paths:
        raise SystemExit("No output JSON files found. Provide --inputs or write method artifacts under outputs/.")

    expected = expected_methods_from_train_scores(Path(args.train_scores))
    scores = merge_scores(input_paths, expected)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
