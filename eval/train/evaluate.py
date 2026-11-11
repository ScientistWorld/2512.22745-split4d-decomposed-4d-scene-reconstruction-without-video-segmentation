"""Visible train (dev-scene) evaluator for the Split4D gym.

Same scoring interface and rules as the hidden test evaluator, applied to the
public development scenes whose answers ship in ``eval/train/truth``. Scores
every method output found under ``outputs/`` (``*.json`` files whose top level
is an object with ``method`` and/or ``scenes``):

* each dev scene is scored in held-out views over all frames; a pixel's
  predicted instance is the predicted label with the largest summed blend
  weight among its contributors (``-1`` = unassigned), its true instance is
  the object with the largest summed blend weight;
* predicted labels are matched to true instances once per scene (Hungarian,
  over all frames and held-out views), so identity switches over time cost score;
* per-instance IoU / accuracy / recall / F1 are averaged over instances,
  dynamic-object and static-object recall separately, then over the scenes of
  each experiment.

A method run fails, and receives the worst value (0) for every metric of every
experiment, if any required dev scene is missing or malformed, if labels are
not integers >= -1 of the exact primitive count, if it uses more than
MAX_LABELS distinct labels in a scene, or if it does not beat the fixed
no-streaming reference artifact (``fixed_baseline.json``) on F1 and dynamic
recall in the interaction experiment. Nothing the method writes besides
``pred_labels`` is read.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRUTH_DIR = HERE / "truth"
FIXED_BASELINE = HERE / "fixed_baseline.json"
DEFAULT_OUTPUT = ROOT / "scoring" / "scores_train.json"
METRICS = ("mIoU", "mAcc", "Recall", "F1", "Recalldyn", "Recallstatic")
WORST = {m: 0.0 for m in METRICS}
GATE = {"scaled_interaction_segmentation": ("F1", "Recalldyn")}
MAX_LABELS = 512
MISSING_NAME = "missing_method_output"
NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,96}$")


class InvalidOutput(Exception):
    pass


def load_truth(truth_dir: Path = TRUTH_DIR) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, np.ndarray]]]:
    manifest = json.loads((truth_dir / "manifest.json").read_text(encoding="utf-8"))
    scenes: Dict[str, Dict[str, np.ndarray]] = {}
    for sid, meta in manifest["scenes"].items():
        with np.load(truth_dir / meta["file"], allow_pickle=False) as z:
            scenes[sid] = {k: z[k] for k in z.files}
        scenes[sid]["test_weight"] = scenes[sid].pop("test_weight_u8").astype(np.float64) / 255.0
        scenes[sid]["test_contrib"] = scenes[sid]["test_contrib"].astype(np.int64)
    return manifest["experiments"], scenes


def check_labels(value: object, n: int, scene_id: str) -> np.ndarray:
    if not isinstance(value, list):
        raise InvalidOutput(f"{scene_id}: pred_labels must be a list")
    if len(value) != n:
        raise InvalidOutput(f"{scene_id}: pred_labels has length {len(value)}, expected {n}")
    if not all(type(x) is int for x in value):
        raise InvalidOutput(f"{scene_id}: pred_labels must contain only integers")
    labels = np.asarray(value, dtype=np.int64)
    if labels.min() < -1 or labels.max() >= 10**9:
        raise InvalidOutput(f"{scene_id}: pred_labels must be integers in [-1, 1e9)")
    if len(np.unique(labels[labels >= 0])) > MAX_LABELS:
        raise InvalidOutput(f"{scene_id}: more than {MAX_LABELS} distinct labels")
    return labels


def pixel_predictions(labels: np.ndarray, contrib: np.ndarray, weight: np.ndarray) -> np.ndarray:
    lab = labels[contrib]
    support = ((lab[:, :, None] == lab[:, None, :]) * weight[:, None, :]).sum(axis=2)
    return lab[np.arange(len(lab)), support.argmax(axis=1)]


def scene_metrics(labels: np.ndarray, truth: Mapping[str, np.ndarray]) -> Dict[str, float]:
    pred = pixel_predictions(labels, truth["test_contrib"], truth["test_weight"])
    true = truth["test_visible"].astype(np.int64)
    dynamic = truth["object_dynamic"].astype(bool)
    k_true = len(dynamic)
    pred_ids = np.unique(pred[pred >= 0])
    conf = np.zeros((len(pred_ids), k_true), dtype=np.int64)
    if len(pred_ids):
        pi = np.searchsorted(pred_ids, pred[pred >= 0])
        np.add.at(conf, (pi, true[pred >= 0]), 1)
    pred_size = conf.sum(axis=1)
    true_size = np.bincount(true, minlength=k_true)
    total = float(len(true))
    tp = np.zeros(k_true)
    fp_plus_tp = np.zeros(k_true)
    if len(pred_ids):
        rows, cols = linear_sum_assignment(-conf)
        tp[cols] = conf[rows, cols]
        fp_plus_tp[cols] = pred_size[rows]
    fp = fp_plus_tp - tp
    fn = true_size - tp
    tn = total - tp - fp - fn
    iou = tp / np.maximum(tp + fp + fn, 1.0)
    acc = (tp + tn) / max(total, 1.0)
    recall = tp / np.maximum(tp + fn, 1.0)
    precision = tp / np.maximum(tp + fp, 1.0)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    present = true_size > 0
    dyn = present & dynamic
    sta = present & ~dynamic
    return {
        "mIoU": float(iou[present].mean()),
        "mAcc": float(acc[present].mean()),
        "Recall": float(recall[present].mean()),
        "F1": float(f1[present].mean()),
        "Recalldyn": float(recall[dyn].mean()) if dyn.any() else float(recall[present].mean()),
        "Recallstatic": float(recall[sta].mean()) if sta.any() else float(recall[present].mean()),
    }


def experiment_metrics(scene_labels: Mapping[str, np.ndarray], experiments: Mapping[str, List[str]],
                       truth: Mapping[str, Mapping[str, np.ndarray]],
                       per_scene: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Dict[str, float]]:
    out = {}
    for exp, sids in experiments.items():
        per = [scene_metrics(scene_labels[sid], truth[sid]) for sid in sids]
        if per_scene is not None:
            per_scene.update({sid: p for sid, p in zip(sids, per)})
        out[exp] = {m: float(np.mean([p[m] for p in per])) for m in METRICS}
    return out


def required_labels(scenes: object, experiments: Mapping[str, List[str]],
                    truth: Mapping[str, Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not isinstance(scenes, Mapping):
        raise InvalidOutput("scenes must be an object keyed by scene id")
    labels = {}
    for sids in experiments.values():
        for sid in sids:
            body = scenes.get(sid)
            if not isinstance(body, Mapping) or "pred_labels" not in body:
                raise InvalidOutput(f"missing predictions for scene {sid}")
            labels[sid] = check_labels(body["pred_labels"], int(len(truth[sid]["object_id"])), sid)
    return labels


def load_fixed_baseline(path: Path, experiments, truth):
    if not path.exists():
        raise SystemExit(f"{path} is missing: the fixed comparison artifact must ship with the evaluator")
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = required_labels(payload["scenes"], experiments, truth)
    return labels, experiment_metrics(labels, experiments, truth)


def discover(root: Path) -> List[Tuple[str, Optional[Mapping[str, object]], Optional[str]]]:
    found = []
    if not root.exists():
        return found
    for path in sorted(p for p in root.rglob("*.json") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            found.append((rel, None, f"unreadable output file: {exc.__class__.__name__}"))
            continue
        if isinstance(payload, Mapping) and ("scenes" in payload or "method" in payload):
            found.append((rel, payload, None))
    return found


def path_name(rel: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.\-]", "_", rel[:-5] if rel.endswith(".json") else rel)
    return stem[:96] or "unnamed_output"


def group_methods(found) -> Dict[str, Dict[str, object]]:
    """Group output files by method name; scenes from several files of one method are merged."""

    groups: Dict[str, Dict[str, object]] = {}
    for rel, payload, error in found:
        name = payload.get("method") if isinstance(payload, Mapping) else None
        if not isinstance(name, str) or not NAME_RE.match(name):
            name = path_name(rel)
        g = groups.setdefault(name, {"files": [], "scenes": {}, "error": None})
        g["files"].append(rel)
        if error:
            g["error"] = error
            continue
        scenes = payload.get("scenes")
        if not isinstance(scenes, Mapping):
            g["error"] = f"{rel}: scenes must be an object keyed by scene id"
            continue
        for sid, body in scenes.items():
            if sid in g["scenes"] and g["scenes"][sid] != body:
                g["error"] = f"conflicting predictions for scene {sid} in {g['files']}"
            g["scenes"][sid] = body
    return groups


def score_all(outputs_root: Path, truth_dir: Path, fixed_baseline: Path) -> Dict[str, object]:
    experiments, truth = load_truth(truth_dir)
    base_labels, base_metrics = load_fixed_baseline(fixed_baseline, experiments, truth)
    groups = group_methods(discover(outputs_root))
    results: Dict[str, Dict[str, Dict[str, float]]] = {exp: {} for exp in experiments}
    status: Dict[str, Dict[str, object]] = {}
    per_scene_all: Dict[str, Dict[str, Dict[str, float]]] = {}
    if not groups:
        groups = {MISSING_NAME: {"files": [], "scenes": {}, "error": "no method outputs found under outputs/"}}
    for name, g in sorted(groups.items()):
        reason = g["error"]
        metrics = None
        if reason is None:
            try:
                labels = required_labels(g["scenes"], experiments, truth)
                per_scene: Dict[str, Dict[str, float]] = {}
                metrics = experiment_metrics(labels, experiments, truth, per_scene)
                per_scene_all[name] = per_scene
            except InvalidOutput as exc:
                reason = str(exc)
        if metrics is not None:
            is_fixed_baseline = all(np.array_equal(labels[sid], base_labels[sid]) for sid in base_labels)
            for exp, gated in GATE.items():
                if is_fixed_baseline:
                    break
                for metric in gated:
                    if not metrics[exp][metric] > base_metrics[exp][metric]:
                        reason = (f"constraint failed: {exp} {metric} {metrics[exp][metric]:.6f} does not exceed the "
                                  f"fixed no-streaming reference {base_metrics[exp][metric]:.6f}")
                        break
                if reason:
                    break
        valid = reason is None and metrics is not None
        if valid and not all(math.isfinite(v) and 0.0 <= v <= 1.0 for row in metrics.values() for v in row.values()):
            valid, reason = False, "non-finite or out-of-range metric"
        for exp in experiments:
            results[exp][name] = dict(metrics[exp]) if valid else dict(WORST)
        status[name] = {"valid": valid, "files": g["files"], **({} if valid else {"reason": reason})}
    return {
        "experiments": {exp: {"results": rows} for exp, rows in results.items()},
        "_status": status,
        "_fixed_reference_artifact": base_metrics,
        "_per_scene_diagnostics": per_scene_all,
    }


def freeze_baseline(path: Path, truth_dir: Path, target: Path) -> None:
    experiments, truth = load_truth(truth_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = required_labels(payload["scenes"], experiments, truth)
    frozen = {"method": payload.get("method", "fixed_reference"),
              "note": "Fixed no-streaming (random-order) reference artifact used by the constraint gate.",
              "scenes": {sid: {"pred_labels": [int(x) for x in lab]} for sid, lab in sorted(labels.items())}}
    target.write_text(json.dumps(frozen) + "\n", encoding="utf-8")
    print(f"froze {len(labels)} scenes from {path} into {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score every method output on the visible dev scenes.")
    parser.add_argument("--outputs", default=str(ROOT / "outputs"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--truth-dir", default=str(TRUTH_DIR))
    parser.add_argument("--fixed-baseline", default=str(FIXED_BASELINE))
    parser.add_argument("--freeze-baseline", default=None, help="Build-time only: freeze this output as the fixed artifact.")
    args = parser.parse_args()
    if args.freeze_baseline:
        freeze_baseline(Path(args.freeze_baseline), Path(args.truth_dir), Path(args.fixed_baseline))
        return
    scores = score_all(Path(args.outputs), Path(args.truth_dir), Path(args.fixed_baseline))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
