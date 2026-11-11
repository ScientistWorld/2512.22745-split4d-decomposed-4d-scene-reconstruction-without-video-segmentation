#!/bin/bash
set -euo pipefail
cd /home/user
REDTEAM_RERUN=1 python -u -m redteam.harness strong_more_training
