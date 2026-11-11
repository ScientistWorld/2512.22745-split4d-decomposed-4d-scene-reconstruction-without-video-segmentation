#!/bin/bash
set -euo pipefail
cd /home/user
python -u -m redteam.harness bruteforce_mask_permutations
