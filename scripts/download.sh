#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: <1 MB
#
# This benchmark uses deterministic procedural data, so no external files are
# fetched. The generated files are small and reproducible from this script.

set -e

cd /home/user

mkdir -p data/generated data/downloads
python -m data.split4d_synthetic
echo "Generated deterministic Split4D synthetic scenes in data/generated/."
