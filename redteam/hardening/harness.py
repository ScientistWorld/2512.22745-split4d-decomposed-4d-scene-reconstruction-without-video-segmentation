"""Run every hardening case through the unchanged train/test entry points.

For each case the harness swaps ``outputs/`` for the case's files only, runs
``bash scripts/evaluate_train.sh`` and then ``bash scripts/evaluate_test.sh``
with no arguments in separate processes, saves both score files, and restores
``outputs/`` and the score files afterwards.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "redteam" / "hardening"
ART = H / "artifacts"
CASES = H / "cases"
OUT = ROOT / "outputs"
BACKUP = ROOT / "outputs.__hardening_backup"
SCORES = ("scores_train.json", "scores_test.json", "scores.json")


def fingerprint() -> str:
    h = hashlib.sha256()
    for path in sorted((ROOT / "eval").rglob("*.py")) + [ROOT / "scripts/evaluate_test.sh", ROOT / "scripts/evaluate_train.sh"]:
        h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    before = fingerprint()
    saved = {name: (ROOT / "scoring" / name).read_bytes() for name in SCORES if (ROOT / "scoring" / name).exists()}
    if OUT.exists():
        if BACKUP.exists():
            shutil.rmtree(BACKUP)
        OUT.rename(BACKUP)
    try:
        for case_dir in sorted(p for p in ART.iterdir() if (p / "DONE").exists()):
            case = case_dir.name
            meta = json.loads((case_dir / "meta.json").read_text())
            dest = CASES / case
            dest.mkdir(parents=True, exist_ok=True)
            if OUT.exists():
                shutil.rmtree(OUT)
            OUT.mkdir()
            for item in case_dir.iterdir():
                if item.name in {"meta.json", "DONE"}:
                    continue
                if item.is_dir():
                    shutil.copytree(item, OUT / item.name)
            for name in ("scores_train.json", "scores_test.json"):
                (ROOT / "scoring" / name).unlink(missing_ok=True)
            if case == "shortcut_forged_scores":
                forged = {"experiments": {"scaled_dynamic_scene_segmentation": {"results": {"split4d_streaming": {"mIoU": 1.0}}}}}
                (ROOT / "scoring" / "scores_test.json").write_text(json.dumps(forged))
            status = {"case": case, **meta}
            for split in ("train", "test"):
                t0 = time.time()
                proc = subprocess.run(["bash", f"scripts/evaluate_{split}.sh"], cwd=ROOT, capture_output=True, text=True)
                status[split] = {"command": ["bash", f"scripts/evaluate_{split}.sh"], "returncode": proc.returncode,
                                 "seconds": round(time.time() - t0, 2)}
                if proc.returncode != 0:
                    (dest / f"{split}_stderr.txt").write_text(proc.stderr[-4000:])
                src = ROOT / "scoring" / f"scores_{split}.json"
                if src.exists():
                    shutil.copy(src, dest / f"scores_{split}.json")
            status["methods_on_disk"] = sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*.json"))
            status["evaluator_unchanged"] = fingerprint() == before
            (dest / "status.json").write_text(json.dumps(status, indent=2) + "\n")
            print(f"{case}: train rc={status['train']['returncode']} test rc={status['test']['returncode']}", flush=True)
    finally:
        if OUT.exists():
            shutil.rmtree(OUT)
        if BACKUP.exists():
            BACKUP.rename(OUT)
        for name, data in saved.items():
            (ROOT / "scoring" / name).write_bytes(data)


if __name__ == "__main__":
    main()
