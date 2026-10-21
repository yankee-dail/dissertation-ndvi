#!/bin/bash
# Послідовний ланцюг: Пампа 2018 (L2A) -> Полтава 2021 (L2A). Кожен регіон: 3 seed x
# (причинна L1-модель, причинна σ-модель, validation_extras, детектор аномалій global + Mondrian).
cd /Users/sbaluk/dissertation-ndvi
P=.venv312/bin/python
SPP=8
export OMP_NUM_THREADS=5

experiments() {
  region=$1; out=results/l2a_${region}${SUFFIX:-}
  RS="--ref-select ${REFSEL:-clearest}"
  Z=$region/zones
  mkdir -p $out
  for s in 0 1 2; do
    [ -f $out/step2_causal_seed$s/model.pt ] || \
      $P scripts/train_step2.py --root data/l2a --train-roi $Z/train --val-roi $Z/val --test-roi $Z/test \
        --epochs 10 --img-size 128 --seed $s --near-mode prior --samples-per-patch $SPP $RS \
        --out $out/step2_causal_seed$s > $out/step2_causal_seed$s.log 2>&1 &
    [ -f $out/unc_causal_seed$s/model.pt ] || \
      $P scripts/train_step2_uncertainty.py --root data/l2a --train-roi $Z/train --calib-roi $Z/calib --test-roi $Z/test \
        --epochs 10 --img-size 128 --coverage 0.9 --seed $s --near-mode prior --samples-per-patch $SPP $RS \
        --out $out/unc_causal_seed$s > $out/unc_causal_seed$s.log 2>&1 &
    wait
    M="--step2-model $out/step2_causal_seed$s/model.pt --uncertainty-model $out/unc_causal_seed$s/model.pt"
    C="--root data/l2a --calib-roi $Z/calib --test-roi $Z/test --img-size 128 --seed $s --near-mode prior --samples-per-patch $SPP $RS"
    [ -f $out/validation_seed$s/summary.json ] || \
      $P scripts/validation_extras.py $C $M --out $out/validation_seed$s > $out/validation_seed$s.log 2>&1 &
    [ -f $out/anomaly_seed$s/summary.json ] || \
      $P scripts/anomaly_detection.py $C $M --out $out/anomaly_seed$s > $out/anomaly_seed$s.log 2>&1 &
    [ -f $out/anomaly_mondrian_seed$s/summary.json ] || \
      $P scripts/anomaly_detection.py $C $M --mondrian --gap-bins 5 10 20 40 \
        --out $out/anomaly_mondrian_seed$s > $out/anomaly_mondrian_seed$s.log 2>&1 &
    wait
    echo "$(date '+%F %T') $region seed $s done"
  done
}

if [ "$1" = "exp" ]; then experiments "$2"; echo "$(date '+%F %T') $2 ALL DONE"; exit 0; fi

fetch() {
  # Повторюємо завантаження (воно продовжує з місця обриву), доки не завершиться успішно.
  name=$1; shift
  for attempt in 1 2 3 4 5 6 7 8; do
    $P scripts/fetch_s2_l2a.py "$@" >> results/fetch_$name.log 2>&1 && { echo "$(date '+%F %T') $name fetch ok"; return 0; }
    echo "$(date '+%F %T') $name fetch failed (attempt $attempt), retry in 2 min"; sleep 120
  done
  return 1
}

fetch pampa2018 --zones configs/zones_pampa2018.json --grid MGRS-21HUB --year 2018 \
  --region pampa2018 --max-cloud 50 || exit 1

fetch poltava2021 --zones configs/zones_poltava2021.json --grid MGRS-36UWA --year 2021 \
  --start 2021-03-15 --end 2021-10-31 --region poltava2021 --max-cloud 50 &
FETCH=$!

experiments pampa2018
wait $FETCH
experiments poltava2021
echo "$(date '+%F %T') ALL DONE"
