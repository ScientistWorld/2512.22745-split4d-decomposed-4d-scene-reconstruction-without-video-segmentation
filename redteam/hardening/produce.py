"""Produce every hardening case's method artifacts (resumable; skips finished cases).

Trained variants call the scientist-visible method code; simple baselines use
only public scene inputs; shortcuts are built from public files and the
reference outputs. Nothing here reads eval/ answers.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "redteam" / "hardening"
ART = H / "artifacts"
REF_OUT = ROOT / "outputs" / "split4d" / "results.json"
BASE_OUT = ROOT / "outputs" / "baseline" / "random_order.json"

TRAINED = {
    # case: (method name, extra args)
    "reference_seed1": ("split4d_streaming", ["--seed", "1"]),
    "reference_seed2": ("split4d_streaming", ["--seed", "2"]),
    "random_order_seed1": ("split4d_random_order", ["--no-streaming", "--seed", "1"]),
    "weak_untrained": ("split4d_untrained", ["--steps", "0"]),
    "ablation_no_tracking": ("split4d_no_tracking", ["--lambda-align", "0"]),
    "ablation_no_dino": ("split4d_no_dino", ["--lambda-dino", "0"]),
    "variant_long_training": ("split4d_long_training", ["--steps", "30000"]),
    "variant_weak_regularizers": ("split4d_weak_regularizers", ["--loss-weight-scale", "0.001"]),
}
SIMPLE = ("geometry_kmeans", "mask_graph_spectral", "mask_tracker", "semantic_kmeans")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write(case: str, rel: str, payload: object) -> None:
    path = ART / case / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def done(case: str) -> bool:
    return (ART / case / "DONE").exists()


def mark(case: str, meta: dict) -> None:
    (ART / case).mkdir(parents=True, exist_ok=True)
    (ART / case / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (ART / case / "DONE").write_text("ok\n", encoding="utf-8")


def run(cmd: list) -> float:
    t0 = time.time()
    log("run " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)
    return time.time() - t0


def main() -> None:
    ref = json.loads(REF_OUT.read_text())
    base = json.loads(BASE_OUT.read_text())

    for case, payload, name in (("reference_seed0", ref, "split4d_streaming"), ("random_order_seed0", base, "split4d_random_order")):
        if not done(case):
            write(case, "candidate/results.json", payload)
            mark(case, {"candidate": name, "kind": "reference run"})

    for case, (name, extra) in TRAINED.items():
        if done(case):
            continue
        out = ART / case / "candidate" / "results.json"
        secs = run([sys.executable, "-m", "method.split4d", "--output", str(out), "--method-name", name, *extra])
        mark(case, {"candidate": name, "kind": "trained variant", "args": extra, "seconds": secs})

    if not done("variant_seed_ensemble"):
        feats = [str(ROOT / "outputs/split4d/results.features.npz"),
                 str(ART / "reference_seed1/candidate/results.features.npz"),
                 str(ART / "reference_seed2/candidate/results.features.npz")]
        out = ART / "variant_seed_ensemble" / "candidate" / "results.json"
        secs = run([sys.executable, "-m", "method.split4d", "--output", str(out), "--method-name",
                    "split4d_seed_ensemble", "--ensemble-from", *feats])
        mark("variant_seed_ensemble", {"candidate": "split4d_seed_ensemble", "kind": "trained variant", "seconds": secs})

    if not done("variant_mask_vote_refine"):
        out = ART / "variant_mask_vote_refine" / "candidate" / "results.json"
        secs = run([sys.executable, "-m", "redteam.hardening.refine", str(REF_OUT), "--output", str(out)])
        mark("variant_mask_vote_refine", {"candidate": "split4d_mask_vote_refine", "kind": "trained variant",
                                          "source": "reference_seed0", "seconds": secs})

    procs = {}
    for method in SIMPLE:
        if done(method):
            continue
        out = ART / method / "candidate" / "results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        log(f"start simple baseline {method}")
        procs[method] = (subprocess.Popen([sys.executable, "-m", "baseline.simple_grouping", method, "--output", str(out)],
                                          cwd=ROOT), time.time())
    for method, (proc, t0) in procs.items():
        if proc.wait() == 0:
            mark(method, {"candidate": method, "kind": "simple baseline", "seconds": time.time() - t0})
        else:
            log(f"simple baseline {method} failed rc={proc.returncode}")

    # ---------------------------------------------------------------- shortcuts
    import numpy as np

    sys.path.insert(0, str(ROOT))
    from data.split4d_scenes import list_scenes

    scenes = list_scenes()
    sizes = {s["id"]: int(s["primitives"]) for s in scenes}
    test_ids = [s["id"] for s in scenes if s["split"] == "test"]

    def labelled(name: str, fn) -> dict:
        return {"method": name, "scenes": {sid: {"pred_labels": [int(x) for x in fn(sid, n)]} for sid, n in sizes.items()}}

    shortcuts = {}
    shortcuts["shortcut_row_blocks"] = {"candidate/results.json": labelled("shortcut_row_blocks", lambda sid, n: (np.arange(n) * 8) // n)}
    shortcuts["shortcut_constant"] = {"candidate/results.json": labelled("shortcut_constant", lambda sid, n: np.zeros(n, dtype=int))}
    shortcuts["shortcut_singletons"] = {"candidate/results.json": labelled("shortcut_singletons", lambda sid, n: np.arange(n))}
    shortcuts["shortcut_empty_run"] = {"candidate/results.json": {"method": "split4d_streaming", "scenes": {}}}
    partial = copy.deepcopy(ref)
    for sid in list(partial["scenes"]):
        if sid.startswith("interaction_"):
            del partial["scenes"][sid]
    shortcuts["shortcut_partial_experiment"] = {"candidate/results.json": partial}
    dev_only = copy.deepcopy(ref)
    for sid in test_ids:
        dev_only["scenes"].pop(sid, None)
    shortcuts["shortcut_dev_only"] = {"candidate/results.json": dev_only}
    malformed = copy.deepcopy(ref)
    malformed["scenes"][test_ids[0]]["pred_labels"] = malformed["scenes"][test_ids[0]]["pred_labels"][:-5]
    shortcuts["shortcut_malformed"] = {"candidate/results.json": malformed}
    nonfinite = copy.deepcopy(ref)
    labels = nonfinite["scenes"][test_ids[-1]]["pred_labels"]
    labels[0] = float("nan")
    labels[1] = 2.5
    shortcuts["shortcut_nonfinite"] = {"candidate/results.json": nonfinite}
    forged = copy.deepcopy(ref)
    forged.update(success=True, mIoU=1.0, F1=1.0, results={"mIoU": 1.0, "Recalldyn": 1.0})
    for body in forged["scenes"].values():
        body.update(mIoU=1.0, F1=1.0, Recalldyn=1.0, verdict="pass")
    shortcuts["shortcut_forged_scores"] = {"candidate/results.json": forged}
    renamed = copy.deepcopy(ref)
    renamed["method"] = "split4d_random_order"
    shortcuts["shortcut_baseline_identity"] = {"candidate/results.json": renamed}
    shortcuts["shortcut_replace_baseline"] = {"split4d/results.json": ref, "baseline/random_order.json": renamed}
    broken = labelled("shortcut_hidden_failed", lambda sid, n: np.zeros(n, dtype=int))
    shortcuts["shortcut_hide_method"] = {"split4d/results.json": ref, "zz_extra/hidden.json": broken}
    fixed = json.loads((ROOT / "eval/test/fixed_baseline.json").read_text())
    fixed_dev = json.loads((ROOT / "eval/train/fixed_baseline.json").read_text())
    copied = {"method": "shortcut_copy_fixed_baseline", "scenes": {**fixed_dev["scenes"], **fixed["scenes"]}}
    shortcuts["shortcut_copy_fixed_baseline"] = {"candidate/results.json": copied}
    for case, files in shortcuts.items():
        if done(case):
            continue
        for rel, payload in files.items():
            write(case, rel, payload)
        cand = next(iter(files.values()))
        name = cand.get("method") if isinstance(cand, dict) else None
        mark(case, {"candidate": name, "kind": "shortcut", "files": sorted(files)})
    log("all artifacts produced")


if __name__ == "__main__":
    main()
