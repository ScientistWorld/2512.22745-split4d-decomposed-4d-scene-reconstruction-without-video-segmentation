#!/bin/bash
set -euo pipefail
cd /home/user
bash redteam/strong_more_training/run.sh
bash redteam/shortcut_replace_baseline/run.sh
python redteam/analyze.py
