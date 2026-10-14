#!/usr/bin/env bash
# P1: (a) 4 more seeds for the grid-search-selected config (w=0.1, asym=2.0, day-scale),
#     (b) a ramp-up variant of the same config,
#     (c) a robust-threshold variant (NDVI rate bounds estimated from a wider ROI set,
#         train ROI + 1 extra local ROI NOT used as val/test, to avoid leakage).
# Sequential (one training process at a time) to avoid CPU contention seen earlier.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv312/bin/activate

TRAIN_ROIS=(
  --train-roi s2_america_test/ROIs1158/106
  --train-roi s2_africa_test/ROIs2017/32
  --train-roi s2_africa_test/ROIs2017/140
  --train-roi s2_asiaEast_test/ROIs1868/73
)
VAL_TEST=(--val-roi s2_asiaEast_test/ROIs1868/119 --test-roi s2_america_test/ROIs1970/132)

echo "=== (a) 4 more seeds for tuned config (w=0.1, asym=2.0) ==="
for seed in 1 2 3 4; do
  out="results/phen_grid_best_seed${seed}"
  log="results/phen_grid_best_seed${seed}.log"
  if [ -f "${out}/report.md" ]; then
    echo "skip seed${seed} (already done)"
  else
    echo "=== running phen_grid_best_seed${seed} ==="
    python scripts/train_step2.py --root data/raw "${TRAIN_ROIS[@]}" "${VAL_TEST[@]}" \
      --epochs 10 --img-size 128 \
      --phen-weight 0.1 --phen-asym-ratio 2.0 \
      --seed "${seed}" \
      --out "${out}" > "${log}" 2>&1
  fi
done

echo "=== (b) ramp-up variant (3 epochs) ==="
out="results/phen_grid_best_ramp3"
log="results/phen_grid_best_ramp3.log"
if [ -f "${out}/report.md" ]; then
  echo "skip ramp3 (already done)"
else
  python scripts/train_step2.py --root data/raw "${TRAIN_ROIS[@]}" "${VAL_TEST[@]}" \
    --epochs 10 --img-size 128 \
    --phen-weight 0.1 --phen-asym-ratio 2.0 --phen-ramp-epochs 3 \
    --seed 0 \
    --out "${out}" > "${log}" 2>&1
fi

echo "=== (c) robust-threshold variant (train ROI + extra local ROI, not val/test) ==="
out="results/phen_grid_best_robustthresh"
log="results/phen_grid_best_robustthresh.log"
if [ -f "${out}/report.md" ]; then
  echo "skip robustthresh (already done)"
else
  python scripts/train_step2.py --root data/raw "${TRAIN_ROIS[@]}" "${VAL_TEST[@]}" \
    --epochs 10 --img-size 128 \
    --phen-weight 0.1 --phen-asym-ratio 2.0 \
    --phen-threshold-roi s2_america_test/ROIs1158/106 \
    --phen-threshold-roi s2_africa_test/ROIs2017/32 \
    --phen-threshold-roi s2_africa_test/ROIs2017/140 \
    --phen-threshold-roi s2_asiaEast_test/ROIs1868/73 \
    --phen-threshold-roi s2_asiaEast_test/ROIs1970/139 \
    --seed 0 \
    --out "${out}" > "${log}" 2>&1
fi

echo "ALL P1 RUNS DONE"
