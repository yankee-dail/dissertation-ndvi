#!/usr/bin/env bash
# P0: reproducibility check for the negative phenological-term result.
# Repeats phen_full (day-scale) and era5_phen_full (GDD-scale) configs on seeds 1,2
# (seed 0 already exists as results/phen_full and results/era5_phen_full).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv312/bin/activate

for seed in 1 2; do
  tag="phen_full_seed${seed}"
  out="results/${tag}"
  log="results/${tag}.log"
  if [ -f "${out}/report.md" ]; then
    echo "skip ${tag} (already done)"
  else
    echo "=== running ${tag} (day-scale) ==="
    python scripts/train_step2.py \
      --root data/raw \
      --train-roi s2_america_test/ROIs1158/106 \
      --train-roi s2_africa_test/ROIs2017/32 \
      --train-roi s2_africa_test/ROIs2017/140 \
      --train-roi s2_asiaEast_test/ROIs1868/73 \
      --val-roi s2_asiaEast_test/ROIs1868/119 \
      --test-roi s2_america_test/ROIs1970/132 \
      --epochs 10 --img-size 128 \
      --phen-weight 0.1 --phen-asym-ratio 5.0 \
      --seed "${seed}" \
      --out "${out}" > "${log}" 2>&1
  fi

  tag2="era5_phen_full_seed${seed}"
  out2="results/${tag2}"
  log2="results/${tag2}.log"
  if [ -f "${out2}/report.md" ]; then
    echo "skip ${tag2} (already done)"
  else
    echo "=== running ${tag2} (GDD-scale) ==="
    python scripts/train_step2.py \
      --root data/raw \
      --train-roi s2_america_test/ROIs1158/106 \
      --train-roi s2_africa_test/ROIs2017/32 \
      --train-roi s2_africa_test/ROIs2017/140 \
      --train-roi s2_asiaEast_test/ROIs1868/73 \
      --val-roi s2_asiaEast_test/ROIs1868/119 \
      --test-roi s2_america_test/ROIs1970/132 \
      --epochs 10 --img-size 128 \
      --phen-weight 0.1 --phen-asym-ratio 5.0 \
      --era5-root data/era5 \
      --seed "${seed}" \
      --out "${out2}" > "${log2}" 2>&1
  fi
done

echo "ALL SEED-CHECK RUNS DONE"
