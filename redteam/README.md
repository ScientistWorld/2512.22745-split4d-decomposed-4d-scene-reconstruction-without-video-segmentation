# Benchmark scoring audit

Each case has its own `run.sh`. Run it from the workspace root in the gym
container. It installs the candidate's permitted artifacts, invokes the visible
train evaluator, then invokes the unchanged `bash scripts/evaluate_test.sh`
with **no arguments** in a separate process. Original external files and outputs
are restored in `finally`. `eval/` and the hidden shell script are fingerprinted
and never modified. No candidate changes the evaluator process or environment.

`bash redteam/run_all.sh` bundles all cases. Actual computation must be submitted
as a job, not run on the login node. Saved `status.json`, `scores_train.json`,
`scores_test.json`, and producer diagnostics are evidence. Large candidate
artifacts and verbose logs are ignored locally. To rerun a completed case set
`REDTEAM_RERUN=1`. If the reference artifacts are unavailable, the harness first
recreates them using the scientist-visible method and the recorded configuration.

Reference seeds use 3,200 steps, matching the frozen output's configuration
rather than the launcher's 1,400-step default. The reference algorithms and
reference data are otherwise unchanged. The corrected-temperature variant repairs
the published sum and per-center temperature calculation; the mask-graph variant
adds global grouping from public mask co-membership. Neither reads hidden labels.
The untrained variant uses the original initialization with zero optimizer steps.

All deliberate shortcuts have `shortcut` in their directory and case names.
Their producer code only reads files available to a future scientist, plus
reference trained outputs whose reuse the audit explicitly permits. The hidden
test and the paper are used only to assess scoring and reproduction, never as
inputs to candidate generation. The brute-force and geometry baselines are honest
alternatives, rather than hidden-answer shortcuts.

Aggregate ranks depend on the standard deviations of the eventual attempt pool.
Audit aggregates, when present, explicitly use this audit's finite attempt pool;
the raw test vectors remain the primary evidence.
