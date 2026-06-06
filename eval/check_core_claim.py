from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


REQUIRED_METRICS = {
    "mIoU",
    "mAcc",
    "Recall",
    "F1",
    "Recalldyn",
    "Recallstatic",
    "ClusterSizeError",
}


def load_experiments(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    experiments = payload.get("experiments")
    if not isinstance(experiments, Mapping):
        raise SystemExit(f"{path}: missing top-level experiments object")
    return experiments


def result_metrics(path: Path, experiment: str, method: str) -> Mapping[str, float]:
    experiments = load_experiments(path)
    body = experiments.get(experiment)
    if not isinstance(body, Mapping):
        raise SystemExit(f"{path}: missing experiment {experiment!r}")
    results = body.get("results")
    if not isinstance(results, Mapping):
        raise SystemExit(f"{path}: experiment {experiment!r} has no results object")
    metrics = results.get(method)
    if not isinstance(metrics, Mapping):
        raise SystemExit(f"{path}: missing method {experiment}.{method}")
    missing = sorted(REQUIRED_METRICS.difference(metrics))
    if missing:
        raise SystemExit(f"{path}: {experiment}.{method} missing metrics {missing}")
    return metrics


def check_ablation(path: Path) -> None:
    streaming = result_metrics(path, "scaled_streaming_ablation", "split4d_streaming")
    random_order = result_metrics(path, "scaled_streaming_ablation", "split4d_random_order")
    failures = []
    for metric in ("F1", "Recalldyn"):
        if float(streaming[metric]) <= float(random_order[metric]):
            failures.append(
                f"{metric}: streaming={float(streaming[metric]):.6f} "
                f"random_order={float(random_order[metric]):.6f}"
            )
    if failures:
        joined = "; ".join(failures)
        raise SystemExit(f"{path}: streaming ablation constraint failed ({joined})")


def check_primary(path: Path) -> None:
    result_metrics(path, "scaled_linear_motion_segmentation", "split4d_streaming")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail if current Split4D scores do not support the core claim.")
    parser.add_argument(
        "--score-files",
        nargs="+",
        default=["scoring/scores.json", "scoring/scores_train.json", "scoring/scores_test.json"],
        help="Score files that must contain complete metrics and pass the streaming ablation direction.",
    )
    args = parser.parse_args()

    for raw_path in args.score_files:
        path = Path(raw_path)
        if not path.exists():
            raise SystemExit(f"{path}: score file does not exist")
        check_primary(path)
        check_ablation(path)
    print("Core-claim score gate passed.")


if __name__ == "__main__":
    main()
