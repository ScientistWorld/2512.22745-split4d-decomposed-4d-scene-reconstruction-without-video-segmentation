"""Choose the scene preset on development scenes only (pre-registered criterion).

For every candidate preset: build its development scenes in tmp/, run the
paper's streaming method, the random-order ablation and the simple grouping
baselines, and score them with the visible train evaluator (raw, ungated
per-scene metrics). Criterion, per experiment e, on mean mIoU:

    margin(e) = min(ref(e) - random_order(e), ref(e) - max(simple baselines)(e))

A preset is admissible if the reference stays below 0.92 mIoU in both
experiments (room above the reference). The chosen preset maximizes
min_e margin(e) over admissible presets. Test scenes are never touched.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "redteam" / "hardening"
PRESETS = ("contact_long", "contact_mid", "overlap_long", "overlap_mid")
SIMPLE = ("mask_graph_spectral", "mask_tracker", "semantic_kmeans", "geometry_kmeans")


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=True, **kw)


def run_preset(p: str) -> dict:
    d = ROOT / "tmp" / "sweep" / p
    scenes, truth, outs = d / "scenes", d / "truth", d / "outputs"
    if not (scenes / "manifest.json").exists():
        sh([sys.executable, "-m", "eval.test.scene_builder", "--preset", p, "--only-split", "dev",
            "--public-dir", str(scenes), "--train-truth-dir", str(truth), "--test-truth-dir", str(d / "unused")])
    jobs = []
    if not (outs / "streaming.json").exists():
        jobs.append([sys.executable, "-m", "method.split4d", "--scene-dir", str(scenes), "--output",
                     str(outs / "streaming.json"), "--method-name", "split4d_streaming"])
    if not (outs / "random.json").exists():
        jobs.append([sys.executable, "-m", "method.split4d", "--scene-dir", str(scenes), "--output",
                     str(outs / "random.json"), "--method-name", "split4d_random_order", "--no-streaming"])
    for j in jobs:
        sh(j)
    procs = []
    for m in SIMPLE:
        if not (outs / f"{m}.json").exists():
            print("+ simple", m, flush=True)
            procs.append(subprocess.Popen([sys.executable, "-m", "baseline.simple_grouping", m, "--scene-dir", str(scenes),
                                           "--output", str(outs / f"{m}.json")], cwd=ROOT))
    for proc in procs:
        proc.wait()
    sh([sys.executable, "-m", "eval.train.evaluate", "--truth-dir", str(truth), "--fixed-baseline", str(d / "fixed.json"),
        "--freeze-baseline", str(outs / "random.json")])
    sh([sys.executable, "-m", "eval.train.evaluate", "--truth-dir", str(truth), "--fixed-baseline", str(d / "fixed.json"),
        "--outputs", str(outs), "--output", str(d / "scores.json")], stdout=subprocess.DEVNULL)
    scores = json.loads((d / "scores.json").read_text())
    man = json.loads((truth / "manifest.json").read_text())["experiments"]
    raw = {}
    for name, ps in scores["_per_scene_diagnostics"].items():
        raw[name] = {e: {m: statistics.mean(ps[s][m] for s in sids) for m in ("mIoU", "F1", "Recalldyn", "Recallstatic")}
                     for e, sids in man.items()}
    margins = {}
    for e in man:
        ref = raw["split4d_streaming"][e]["mIoU"]
        rnd = raw["split4d_random_order"][e]["mIoU"]
        best_simple = max(raw[m][e]["mIoU"] for m in SIMPLE if m in raw)
        margins[e] = {"ref": ref, "vs_random": ref - rnd, "vs_best_simple": ref - best_simple,
                      "margin": min(ref - rnd, ref - best_simple)}
    admissible = all(v["ref"] < 0.92 for v in margins.values())
    return {"raw": raw, "margins": margins, "admissible": admissible,
            "score": min(v["margin"] for v in margins.values()) if admissible else None}


def main() -> None:
    results = {}
    for p in PRESETS:
        try:
            results[p] = run_preset(p)
        except Exception as exc:  # keep going; a failing preset is just not chosen
            results[p] = {"error": repr(exc), "score": None}
        print(p, json.dumps(results[p].get("margins", results[p].get("error")), indent=1), flush=True)
    ok = {p: r["score"] for p, r in results.items() if r.get("score") is not None}
    chosen = max(ok, key=ok.get) if ok else "contact_long"
    summary = {"criterion": __doc__.strip(), "presets": results, "chosen": chosen}
    (H / "sweep.json").write_text(json.dumps(summary, indent=2) + "\n")
    (H / "chosen_preset.txt").write_text(chosen + "\n")
    print("chosen preset:", chosen)


if __name__ == "__main__":
    main()
