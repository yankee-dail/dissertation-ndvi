#!/usr/bin/env python3
"""
Звіти про град з European Severe Weather Database (ESWD, ESSL) через офіційний API v1.

Потрібен особистий API-токен (eswd.eu -> Settings -> API token) у змінній середовища:
    export ESWD_TOKEN=...      # не вставляти в код чи чат
    python scripts/fetch_eswd.py --country UA --years 2018 2019 2020 2021 2022 2023 \
        --out data/eswd/hail_UA.json

Запит - по одному вегетаційному сезону (--season-start/--season-end) на рік; зберігаються
нормалізовані записи {id, datetime, lat, lon, type, qc, raw}.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

API = "https://eswd.eu/api/v1/reportList"


def get(params, token):
    last = None
    for auth in (f"Bearer {token}", token):
        r = requests.get(API, params=params, headers={"Authorization": auth}, timeout=120)
        if r.status_code == 200:
            return r.json()
        last = r
        if r.status_code == 429:
            time.sleep(30)
    sys.exit(f"ESWD API: HTTP {last.status_code} - {last.text[:300]}")


def pick(d, *keys):
    for k in keys:
        for kk in (k, k.upper(), k.lower()):
            if kk in d and d[kk] not in (None, ""):
                return d[kk]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--country", default="UA")
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--season-start", default="04-01")
    ap.add_argument("--season-end", default="09-30")
    ap.add_argument("--type", default="HAIL")
    ap.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("LON0", "LAT0", "LON1", "LAT1"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    token = (os.environ.get("ESWD_TOKEN") or "").strip()
    if not token:
        sys.exit("ESWD_TOKEN відсутній: створи токен на eswd.eu (Settings -> API token) і export ESWD_TOKEN=...")

    events = []
    for y in args.years:
        params = {"sd": f"{y}-{args.season_start}T00:00:00Z", "ed": f"{y}-{args.season_end}T23:59:59Z",
                  "countries": args.country, "type": args.type}
        if args.bbox:
            lon0, lat0, lon1, lat1 = args.bbox
            params.update(x0=lat1, x1=lat0, y0=lon0, y1=lon1)
        data = get(params, token)
        items = data if isinstance(data, list) else (data.get("data") or data.get("reports") or data.get("items") or [])
        for it in items:
            events.append({
                "id": str(pick(it, "id", "ID")),
                "datetime": pick(it, "datetime", "TIME_EVENT", "time_event", "DATETIME"),
                "lat": float(pick(it, "latitude", "LATITUDE", "lat")),
                "lon": float(pick(it, "longitude", "LONGITUDE", "lon")),
                "type": pick(it, "type", "TYPE_EVENT", "TYPE"),
                "qc": pick(it, "qc_level", "QC_LEVEL", "level"),
                "city": pick(it, "city", "PLACE", "place"),
                "raw": it,
            })
        print(f"{y}: {len(items)} звітів", flush=True)
        time.sleep(2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(events, indent=1, ensure_ascii=False, default=str))
    print(f"Разом {len(events)} звітів -> {out}")


if __name__ == "__main__":
    main()
