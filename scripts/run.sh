#!/bin/bash
# Diagnostic job: per-object and inference-ablation analysis of the dev-scene
# preset sweep artifacts (no training).
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m redteam.hardening.diagnose > redteam/hardening/logs/diagnose.log 2>&1
echo "diagnose rc=$?"
tail -120 redteam/hardening/logs/diagnose.log
