#!/bin/bash
cd /Users/sbaluk/dissertation-ndvi
P=.venv312/bin/python
export REFSEL=latest_clean SUFFIX=_lc OMP_NUM_THREADS=4
# SEN12MS, причинний режим, latest_clean (порівняння з clearest)
TR="--train-roi s2_america_test/ROIs1158/106 --train-roi s2_africa_test/ROIs2017/32 --train-roi s2_africa_test/ROIs2017/140 --train-roi s2_asiaEast_test/ROIs1868/73"
(
  $P scripts/train_step2.py --root data/raw $TR --val-roi s2_asiaEast_test/ROIs1868/119 --test-roi s2_america_test/ROIs1970/132 \
    --epochs 10 --img-size 128 --seed 0 --near-mode prior --ref-select latest_clean --out results/step2_causal_lc > results/step2_causal_lc.log 2>&1 &
  $P scripts/train_step2_uncertainty.py --root data/raw $TR --calib-roi s2_asiaEast_test/ROIs1970/139 --test-roi s2_america_test/ROIs1970/132 \
    --epochs 10 --img-size 128 --coverage 0.9 --seed 0 --near-mode prior --ref-select latest_clean --out results/uncertainty_causal_lc > results/uncertainty_causal_lc.log 2>&1 &
  wait
  M="--step2-model results/step2_causal_lc/model.pt --uncertainty-model results/uncertainty_causal_lc/model.pt"
  C="--root data/raw --img-size 128 --seed 0 --near-mode prior --ref-select latest_clean"
  $P scripts/validation_extras.py $C $M --out results/validation_extras/causal_lc_cpu_seed0 > results/validation_extras/causal_lc_cpu_seed0.log 2>&1 &
  $P scripts/anomaly_detection.py $C $M --out results/anomaly_detection/causal_lc_cpu_seed0 > results/anomaly_detection/causal_lc_cpu_seed0.log 2>&1 &
  wait
  echo "$(date '+%F %T') sen12ms latest_clean done"
) &
./results/run_l2a_chain.sh exp pampa2018
./results/run_l2a_chain.sh exp poltava2021
wait
echo "$(date '+%F %T') ALL latest_clean DONE"
