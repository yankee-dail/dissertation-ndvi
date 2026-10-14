#!/usr/bin/env bash
# Waits for the grid search to finish, then runs the seed-reproducibility check.
set -euo pipefail
cd "$(dirname "$0")/.."

until grep -q "ALL GRID RUNS DONE" results/phen_grid_driver.log 2>/dev/null; do
  sleep 60
done

echo "grid search finished, starting seed check at $(date)"
bash scripts/run_phen_seed_check.sh > results/phen_seed_check_driver.log 2>&1
echo "ALL P0 WORK DONE at $(date)"
