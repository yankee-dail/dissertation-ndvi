#!/usr/bin/env python3
"""
CSV-вибірка ESWD (формат 2.00-csv, надіслана ESSL) -> JSON подій для hail_validation.py.

Сирі дані ESWD не можна публікувати чи передавати (угода користувача ESSL): і вхідний CSV, і вихідний
JSON лежать у data/eswd/, яку .gitignore не пускає в репозиторій.

Фільтри: TYPE_EVENT = HAIL, DELETED = false, QC0+ і вище, точність часу не гірша за --max-time-acc,
точність місця не гірша за --max-place-km. Звіти однієї дати (UTC), ближчі за --dedup-km, - найчастіше
та сама гроза над сусідніми населеними пунктами (ESWD: один звіт на населений пункт); їхні зони
3 × 3 фрагменти перекриваються, тож вони об'єднуються в одну подію (лишається звіт із кращою точністю
місця, потім вищим QC), щоб бутстреп за подіями не рахував одну грозу кілька разів.

    python scripts/eswd_csv_to_events.py --csv data/eswd/eswd_UA_hail_2018_2023.csv --out data/eswd/hail_UA.json
"""
import argparse
import csv
import json
import math
from pathlib import Path

TIME_H = {"1M": 0, "5M": 0.1, "15M": 0.25, "30M": 0.5, "1H": 1, "3H": 3, "6H": 6, "12H": 12, "1D": 24, "2D": 48}
PLACE_KM = {"1KM": 1, "3KM": 3, "5KM": 5, "10KM": 10, "20KM": 20, "50KM": 50, "100KM": 100}
QC_RANK = {"QC0": 0, "QC0+": 1, "QC1": 2, "QC2": 3}


def km(a, b):
    dy = (a["lat"] - b["lat"]) * 111.2
    dx = (a["lon"] - b["lon"]) * 111.2 * math.cos(math.radians((a["lat"] + b["lat"]) / 2))
    return math.hypot(dx, dy)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--max-time-acc", default="1D", choices=list(TIME_H))
    ap.add_argument("--max-place-km", type=float, default=10)
    ap.add_argument("--dedup-km", type=float, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    ev, dropped = [], {}
    for r in rows:
        why = None
        if r["TYPE_EVENT"] != "HAIL":
            why = "тип"
        elif r["DELETED"].strip().lower() == "true":
            why = "видалено"
        elif QC_RANK.get(r["QC_LEVEL"], 0) < QC_RANK["QC0+"]:
            why = "QC0"
        elif TIME_H.get(r["TIME_ACCURACY"], 1e9) > TIME_H[args.max_time_acc]:
            why = "точність часу"
        elif PLACE_KM.get(r["PLACE_ACCURACY"], 1e9) > args.max_place_km:
            why = "точність місця"
        if why:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        ev.append({"id": f"eswd{r['ID']}", "datetime": r["TIME_EVENT"], "lat": float(r["LATITUDE"]),
                   "lon": float(r["LONGITUDE"]), "type": "HAIL", "qc": r["QC_LEVEL"], "city": r["PLACE"],
                   "place_km": PLACE_KM[r["PLACE_ACCURACY"]], "max_hail_cm": r.get("MAX_HAIL_DIAMETER") or None})

    ev.sort(key=lambda e: (e["datetime"][:10], e["place_km"], -QC_RANK[e["qc"]]))
    kept = []
    for e in ev:
        twin = next((k for k in kept if k["datetime"][:10] == e["datetime"][:10] and km(k, e) < args.dedup_km), None)
        if twin:
            twin["n_reports"] += 1
        else:
            kept.append({**e, "n_reports": 1})
    Path(args.out).write_text(json.dumps(kept, ensure_ascii=False, indent=1))
    print(f"звітів {len(rows)}, відкинуто {dropped or 0}, після фільтрів {len(ev)}, "
          f"подій після об'єднання в радіусі {args.dedup_km:g} км {len(kept)}")


if __name__ == "__main__":
    main()
