#!/bin/bash
# Повторне завантаження L2A після виправлення подвійного зсуву BOA (earthsearch:boa_offset_applied)
# і перерахунок розділу 7: причинна модель, σ-модель, калібрування, детектор (5 запусків, latest_clean),
# потім перевірка на звітах про град.
cd /Users/sbaluk/dissertation-ndvi
P=.venv312/bin/python
fetch() {
  name=$1; shift
  for attempt in 1 2 3 4 5 6 7 8; do
    $P scripts/fetch_s2_l2a.py "$@" >> results/fetch_${name}_v2.log 2>&1 && { echo "$(date '+%F %T') $name fetch ok"; return 0; }
    echo "$(date '+%F %T') $name fetch failed (attempt $attempt)"; sleep 120
  done
  return 1
}
fetch poltava2021 --zones configs/zones_poltava2021.json --grid MGRS-36UWA --year 2021 \
  --start 2021-03-15 --end 2021-10-31 --region poltava2021 --max-cloud 50 &
F1=$!
fetch pampa2018 --zones configs/zones_pampa2018.json --grid MGRS-21HUB --year 2018 --region pampa2018 --max-cloud 50 || exit 1
wait $F1 || exit 1
export REFSEL=latest_clean SUFFIX=_lc SEEDS="0 1 2 3 4" OMP_NUM_THREADS=4
bash results/run_l2a_chain.sh exp poltava2021 &
bash results/run_l2a_chain.sh exp pampa2018 &
wait
echo "$(date '+%F %T') L2A ALL DONE"
for i in 0 1 2; do
  $P scripts/hail_validation.py --events data/eswd/hail_UA.json --models results/l2a_poltava2021_lc --seeds 0 1 2 3 4 \
    --calib-root data/l2a --calib-roi poltava2021/zones/calib --shard $i --n-shards 3 --out results/hail_validation \
    > results/hail_validation/run$i.log 2>&1 &
done
wait
$P scripts/hail_subgroups.py --run results/hail_validation --out results/hail_validation/subgroups.md
echo "$(date '+%F %T') HAIL DONE"
