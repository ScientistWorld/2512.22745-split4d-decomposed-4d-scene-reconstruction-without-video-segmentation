from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set

import numpy as np
from scipy.optimize import linear_sum_assignment

from data.split4d_synthetic import build_scene


SLICE = "test"
SPLIT_SEED_BASE = 91017


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


def stratified_split_indices(scene_name: str, true: np.ndarray) -> np.ndarray:
    seed = SPLIT_SEED_BASE + sum((idx + 1) * ord(ch) for idx, ch in enumerate(scene_name))
    rng = np.random.default_rng(seed)
    selected: List[np.ndarray] = []
    for label in sorted(int(x) for x in np.unique(true)):
        indices = np.flatnonzero(true == label)
        shuffled = indices[rng.permutation(len(indices))]
        selected.append(shuffled[len(shuffled) // 2 :])
    return np.sort(np.concatenate(selected)).astype(np.int64)


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
        size_errors.append(abs(float(predicted.sum()) - float(target.sum())) / max(float(target.sum()), 1.0))
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
    split_indices = stratified_split_indices(scene_name, true_full)
    true = true_full[split_indices]
    pred_raw = pred_full[split_indices]
    pred = align_cluster_ids(pred_raw, true)
    return instance_metrics(pred, true, dynamic_flags(truth_scene))


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
