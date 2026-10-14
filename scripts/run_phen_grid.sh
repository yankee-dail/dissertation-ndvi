#!/usr/bin/env bash
# P0: grid search --phen-weight x --phen-asym-ratio on the SAME split as step2_final/phen_full
# (train=4 ROI 986 patches, val=asiaEast/119, test=america/132), day-scaled phenology.
# Screening pass: 5 epochs (not 10) to cut CPU time ~2x; winner gets a confirmatory 10-epoch run separately.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv312/bin/activate

WEIGHTS=(0.01 0.03 0.1 0.3 1.0)
ASYMS=(2.0 5.0 10.0)

mkdir -p results/phen_grid

for w in "${WEIGHTS[@]}"; do
  for a in "${ASYMS[@]}"; do
    tag="w${w}_a${a}"
    out="results/phen_grid/${tag}"
    log="results/phen_grid/${tag}.log"
    if [ -f "${out}/report.md" ]; then
      echo "skip ${tag} (already done)"
      continue
    fi
    echo "=== running ${tag} ==="
    python scripts/train_step2.py \
      --root data/raw \
      --train-roi s2_america_test/ROIs1158/106 \
      --train-roi s2_africa_test/ROIs2017/32 \
      --train-roi s2_africa_test/ROIs2017/140 \
      --train-roi s2_asiaEast_test/ROIs1868/73 \
      --val-roi s2_asiaEast_test/ROIs1868/119 \
      --test-roi s2_america_test/ROIs1970/132 \
      --epochs 5 --img-size 128 \
      --phen-weight "${w}" --phen-asym-ratio "${a}" \
      --seed 0 \
      --out "${out}" > "${log}" 2>&1
    echo "=== done ${tag} ==="
  done
done

echo "ALL GRID RUNS DONE"
