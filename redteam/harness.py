"""Install one candidate, run train and the unchanged no-argument hidden test.

Each evaluation is a separate subprocess. Mutations are limited to scientist-visible
outputs, data, and train scores; protected evaluator files are never changed.
All external files are restored even when a candidate/test fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RT = ROOT / "redteam"
NAMES = ["reference_frozen", "reference_seed7", "reference_seed8", "reference_seed9",
         "weak_untrained", "strong_corrected_temperature", "strong_mask_graph_refinement",
         "simple_geometry_baseline", "bruteforce_mask_permutations", "shortcut_read_truth",
         "shortcut_label_features", "shortcut_move_truth", "shortcut_replace_fixed_observations",
         "shortcut_baseline_identity", "shortcut_empty_run", "shortcut_partial_experiment",
         "shortcut_hide_method", "shortcut_forged_scores", "shortcut_constant",
         "shortcut_empty_features", "shortcut_nonfinite", "shortcut_malformed"]
RESTORED = ["data/split4d_synthetic.py", "scoring/scores_train.json", "scoring/scores_test.json"]


def fingerprint():
    paths = sorted((ROOT / "eval").rglob("*.py")) + [ROOT / "scripts/evaluate_test.sh"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def ensure_reference():
    dest = RT / "reference_artifacts"
    dest.mkdir(exist_ok=True)
    for src, target in [("outputs/split4d/results.json", "reference.json"),
                        ("outputs/baseline/random_order.json", "baseline.json")]:
        if (dest / target).exists():
            continue
        if (ROOT / src).exists():
            shutil.copy2(ROOT / src, dest / target)
            continue
        cfg = json.loads((RT / "reference_config.json").read_text())
        command = [sys.executable, "-m", "method.split4d", "--output", str(dest / target), "--cpu"]
        for key in ["steps", "feature_dim", "samples_per_mask", "lr", "lr_final", "lambda_align",
                    "lambda_dino", "lambda_stream", "dino_fraction", "streaming_window", "seed"]:
            command.extend(["--" + key.replace("_", "-"), str(cfg[key])])
        command += ["--streaming"] if target == "reference.json" else ["--no-streaming"]
        subprocess.run(command, cwd=ROOT, check=True)


def run_command(command, log):
    start = time.monotonic()
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    return {"command": command, "returncode": result.returncode, "seconds": time.monotonic() - start}


def run(name):
    os.chdir(ROOT)
    folder = RT / name
    folder.mkdir(exist_ok=True)
    if (folder / "status.json").exists() and not os.environ.get("REDTEAM_RERUN"):
        print(f"already saved: {name}", flush=True)
        return
    ensure_reference()
    work = RT / "work" / name
    work.mkdir(parents=True, exist_ok=True)
    saved = {path: (ROOT / path).read_bytes() if (ROOT / path).exists() else None for path in RESTORED}
    initial_fingerprint = fingerprint()
    outputs = ROOT / "outputs"
    backup_outputs = work / "saved_outputs"
    if backup_outputs.exists():
        raise RuntimeError(f"Unrestored outputs from interrupted run: {backup_outputs}")
    had_outputs = outputs.exists()
    status = {"case": name, "hidden_test_no_arguments": True}
    try:
        if had_outputs:
            outputs.rename(backup_outputs)
        outputs.mkdir()
        test_score = ROOT / "scoring/scores_test.json"
        test_score.unlink(missing_ok=True)
        candidate = outputs / "candidate.json"
        print(f"running: {name}", flush=True)
        status["producer"] = run_command(
            [sys.executable, "-m", "redteam.candidates", name, "--output", str(candidate)], folder / "producer.log")
        if status["producer"]["returncode"]:
            return
        if name not in {"shortcut_empty_run", "shortcut_partial_experiment", "shortcut_baseline_identity", "shortcut_hide_method"}:
            shutil.copy2(RT / "reference_artifacts/baseline.json", outputs / "baseline.json")
        if name == "shortcut_hide_method":
            weak = json.loads(candidate.read_text())
            weak["method"] = "shortcut_hidden_failed"
            for scene in weak["scenes"].values():
                scene["pred_labels"] = [0] * len(scene["pred_labels"])
                scene["features"] = [[0.0] * 32 for _ in scene["features"]]
            (outputs / "hidden_failed.json").write_text(json.dumps(weak))
        train_cmd = ["bash", "scripts/evaluate_train.sh"]
        if name == "shortcut_hide_method":
            train_cmd += ["--inputs", str(candidate)]
        status["train"] = run_command(train_cmd, folder / "train.log")
        train_score = ROOT / "scoring/scores_train.json"
        if train_score.exists():
            shutil.copy2(train_score, folder / "scores_train.json")
        if name == "shortcut_forged_scores":
            # Test both output-field claims and hand-written score values. Hidden
            # test must overwrite them with independently measured values.
            forged = {"experiments": {"scaled_linear_motion_segmentation": {"results": {
                "split4d_streaming": {"mIoU": 1, "F1": 1, "Recalldyn": 1}}}}}
            test_score.write_text(json.dumps(forged))
            (folder / "forged_scores.json").write_text(json.dumps(forged, indent=2))
        before_test = test_score.read_bytes() if test_score.exists() else None
        # The only hidden-test invocation: original shell script with no arguments.
        status["test"] = run_command(["bash", "scripts/evaluate_test.sh"], folder / "test.log")
        status["score_file_exists"] = test_score.exists()
        status["test_wrote_scores"] = test_score.exists() and test_score.read_bytes() != before_test
        if test_score.exists():
            shutil.copy2(test_score, folder / "scores_test.json")
        shutil.copy2(candidate, folder / "artifact.json")
        status["methods_on_disk"] = {
            str(p.relative_to(outputs)): json.loads(p.read_text()).get("method") for p in outputs.rglob("*.json")}
    finally:
        status["evaluator_unchanged"] = fingerprint() == initial_fingerprint
        for path, content in saved.items():
            if content is None:
                (ROOT / path).unlink(missing_ok=True)
            else:
                (ROOT / path).write_bytes(content)
        (ROOT / "data/shortcut_override.json").unlink(missing_ok=True)
        # Avoid stale bytecode when source size/timestamp happens to match.
        shutil.rmtree(ROOT / "data/__pycache__", ignore_errors=True)
        if outputs.exists():
            shutil.rmtree(outputs)
        if had_outputs:
            backup_outputs.rename(outputs)
        (folder / "status.json").write_text(json.dumps(status, indent=2))
        print(json.dumps(status), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=NAMES + ["all"])
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    for name in NAMES if args.case == "all" else [args.case]:
        run(name)


if __name__ == "__main__":
    main()
