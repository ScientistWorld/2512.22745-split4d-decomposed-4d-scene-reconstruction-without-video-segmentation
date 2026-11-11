#!/bin/bash
# Hardening job (CPU):
#  1. choose the scene preset on development scenes only (redteam/hardening/sweep.py);
#  2. build the benchmark scenes (public inputs + dev/test answers) with that preset;
#  3. run the random-order baseline and the paper's method, freeze the fixed
#     comparison artifact inside eval/;
#  4. produce every hardening case (seeds, variants, simple baselines,
#     shortcuts) and score each through the unchanged evaluate_{train,test}.sh;
#  5. regenerate the reference score files from the reference runs and analyze.
# Resumable: finished stages are skipped.

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
H=redteam/hardening
mkdir -p "$H/logs"

step() {
  local name=$1
  shift
  local t0
  t0=$(date +%s)
  "$@" > "$H/logs/$name.log" 2>&1
  local rc=$?
  echo "$name rc=$rc seconds=$(( $(date +%s) - t0 ))" | tee -a "$H/logs/steps.txt"
  return $rc
}

python -c "import numpy, scipy, sklearn, torch; from method.split4d import available_cpus; print('usable cpus', available_cpus(), 'numpy', numpy.__version__, 'torch', torch.__version__)" | tee -a "$H/logs/steps.txt"

[ -f "$H/chosen_preset.txt" ] || step sweep python -m redteam.hardening.sweep || exit 1
PRESET=$(cat "$H/chosen_preset.txt")
echo "preset $PRESET" | tee -a "$H/logs/steps.txt"

if ! grep -q "\"preset\": \"$PRESET\"" eval/test/truth/manifest.json 2>/dev/null; then
  rm -rf data/scenes eval/train/truth eval/test/truth outputs/split4d outputs/baseline "$H/artifacts" "$H/cases"
  step build python -m eval.test.scene_builder --preset "$PRESET" || exit 1
fi

fresh() { [ -f "$1" ] && grep -q '"interaction_test_9"' "$1"; }
fresh outputs/baseline/random_order.json || step baseline bash scripts/baseline.sh || exit 1
step freeze_test python -m eval.test.evaluate --freeze-baseline outputs/baseline/random_order.json || exit 1
step freeze_train python -m eval.train.evaluate --freeze-baseline outputs/baseline/random_order.json || exit 1
fresh outputs/split4d/results.json || step method bash scripts/method.sh || exit 1
step score_reference bash -c "bash scripts/evaluate_train.sh && bash scripts/evaluate_test.sh"
step produce python -m redteam.hardening.produce
step harness python -m redteam.hardening.harness
step final bash -c "bash scripts/evaluate.sh && bash scripts/evaluate_train.sh && bash scripts/evaluate_test.sh"
step analyze python -m redteam.hardening.analyze
step validate python validate.py --post-design
exit 0
