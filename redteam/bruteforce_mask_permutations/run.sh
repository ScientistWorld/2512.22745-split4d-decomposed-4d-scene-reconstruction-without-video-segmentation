#!/bin/bash
set -euo pipefail
cd /home/user
REDTEAM_RERUN=1 python -u -m redteam.harness bruteforce_mask_permutations
