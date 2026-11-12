#!/usr/bin/env python3
"""
Розділ 9: ряди NDVI наземних камер PhenoCam і Sentinel-2 L2A для тих самих сільськогосподарських полів.

PhenoCam (phenocam.nau.edu, CC BY 4.0) - мережа камер RGB + ближній ІЧ; для кожної області інтересу (ROI)
публікуються денні ряди NDVI і GCC. Тут камера - наземний IoT-давач, що бачить поле й під хмарами.

Для кожної відібраної ділянки:
  1. денний ряд камери (ndvi_50, gcc_90; викиди за прапорцями PhenoCam відкинуто);
  2. усі сцени Sentinel-2 L2A (earth-search, лише COG) у вегетаційні місяці тих самих років: середнє NDVI
     і частка чистих пікселів (SCL 4, 5) у квадраті 2r × 2r м навколо камери (поле зору камери в
     координатах невідоме - це обмеження), а також хмарність сцени.
Вихід: <out>/<site>.json; повторний запуск пропускає готові ділянки.

    python scripts/phenocam_s2.py --out data/phenocam --max-sites 30
"""
import argparse
import csv
import io
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio.windows import from_bounds

from fetch_s2_l2a import GDAL_ENV, STAC, scale_offset

API = "https://phenocam.nau.edu/api"
ARCHIVE = "https://phenocam.nau.edu/data/archive"
CLEAR_SCL = (4, 5)  # рослинність, ґрунт
# польові культури, близькі до українських сівозмін; сади, пасовища, ліс і рис не беруться
CROPS = ("zea", "maize", "corn", "glycine", "soy", "triticum", "wheat", "hordeum", "barley", "brassica",
         "rapeseed", "helianthus", "sunflower", "sorghum", "gossypium", "cotton", "vicia")


def api_all(path):
    url, out = f"{API}/{path}/?format=json&limit=1000", []
    while url:
        d = json.load(urllib.request.urlopen(url, timeout=120))
        out += d["results"]
        url = d["next"]
    return out


def select_sites(years, months, max_missing, max_sites):
    cams = {c["Sitename"]: c for c in api_all("cameras")}
    rois = [r for r in api_all("roilists") if r["roitype"] == "AG" and r.get("ir_flag")
            and r["first_date"] <= f"{years[1]}-12-31" and r["last_date"] >= f"{years[0]}-01-01"
            and float(r["missing_data_pct"] or 100) <= max_missing]
    by_site = {}
    for r in sorted(rois, key=lambda r: r["sequence_number"]):
        c = cams.get(r["site"])
        sp = (c or {}).get("sitemetadata", {}).get("dominant_species") or ""
        if not any(k in sp.lower() for k in CROPS) or "oryza" in sp.lower():
            continue
        if c and c["Lat"] > 0 and r["site"] not in by_site:  # північна півкуля: сезон квітень-жовтень
            by_site[r["site"]] = {"site": r["site"], "roi": r["roi_name"], "lat": c["Lat"], "lon": c["Lon"],
                                  "species": c["sitemetadata"].get("dominant_species"),
                                  "koeppen": c["sitemetadata"].get("koeppen_geiger"),
                                  "first": r["first_date"], "last": r["last_date"]}
    sites = sorted(by_site.values(), key=lambda s: s["site"])
    overlap = lambda s: min(s["last"][:4], str(years[1])) >= max(s["first"][:4], str(years[0]))
    sites = [s for s in sites if overlap(s)]
    seen, uniq = set(), []  # кілька камер над тим самим полем дають ті самі пікселі Sentinel-2
    for s in sorted(sites, key=lambda s: s["first"]):
        key = (round(s["lat"], 2), round(s["lon"], 2))
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    sites = sorted(uniq, key=lambda s: s["site"])
    return sites[:max_sites] if max_sites else sites


def camera_series(site, roi, months):
    url = f"{ARCHIVE}/{site}/ROI/{roi}_ndvi_1day.csv"
    txt = urllib.request.urlopen(url, timeout=120).read().decode()
    rows = csv.DictReader(io.StringIO("\n".join(l for l in txt.splitlines() if not l.startswith("#"))))
    out = {}
    for r in rows:
        if int(r["date"][5:7]) not in months:
            continue
        v, g = r.get("ndvi_50"), r.get("gcc_90")
        bad = r.get("outlierflag_ndvi_50") not in (None, "", "NA", "0")
        if v not in (None, "", "NA") and not bad:
            out[r["date"]] = {"ndvi": float(v), "gcc": float(g) if g not in (None, "", "NA") else None,
                              "n_img": int(r["image_count"] or 0)}
    return out


def stac_point(lon, lat, start, end):
    body = {"collections": ["sentinel-2-l2a"], "intersects": {"type": "Point", "coordinates": [lon, lat]},
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 200}
    url, items = STAC, []
    while True:
        r = requests.post(url, json=body, timeout=120).json()
        items += r.get("features", [])
        nxt = [l for l in r.get("links", []) if l["rel"] == "next"]
        if not nxt:
            return items
        body, url = nxt[0].get("body", body), nxt[0]["href"]


def s2_sample(it, lon, lat, radius):
    a = it["assets"]
    if not all(a.get(k, {}).get("href", "").endswith(".tif") for k in ("red", "nir", "scl")):
        return None
    out = {}
    try:
        with rasterio.Env(**GDAL_ENV):
            for k in ("red", "nir", "scl"):
                with rasterio.open(a[k]["href"]) as src:
                    x, y = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform(lon, lat)
                    l, b, r_, t = src.bounds
                    if not (l + radius <= x <= r_ - radius and b + radius <= y <= t - radius):
                        return None  # точка біля краю тайла - цю сцену беремо з сусіднього
                    w = from_bounds(x - radius, y - radius, x + radius, y + radius, src.transform)
                    shape = (int(round(2 * radius / 10)),) * 2  # на сітці 10 м
                    out[k] = src.read(1, window=w, out_shape=shape,
                                      resampling=rasterio.enums.Resampling.nearest).astype(np.float64)
                    if k != "scl":
                        out[k + "_so"] = scale_offset(it, k)
    except rasterio.errors.RasterioIOError:
        return None
    valid = (out["red"] > 0) & (out["nir"] > 0)
    clear = np.isin(out["scl"], CLEAR_SCL) & valid
    rec = {"date": it["properties"]["datetime"][:10], "time": it["properties"]["datetime"],
           "id": it["id"], "cloud": it["properties"].get("eo:cloud_cover"),
           "clear_frac": float(clear.mean()), "valid_frac": float(valid.mean())}
    if clear.any():
        red = out["red"] * out["red_so"][0] + out["red_so"][1]
        nir = out["nir"] * out["nir_so"][0] + out["nir_so"][1]
        nd = (nir - red) / np.maximum(nir + red, 1e-6)
        rec["ndvi_clear"] = float(np.mean(nd[clear]))
        rec["ndvi_sd"] = float(np.std(nd[clear]))
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, nargs=2, default=[2018, 2024])
    ap.add_argument("--months", type=int, nargs=2, default=[4, 10])
    ap.add_argument("--max-missing", type=float, default=20)
    ap.add_argument("--max-sites", type=int, default=None)
    ap.add_argument("--radius", type=float, default=60, help="половина сторони квадрата навколо камери, м")
    ap.add_argument("--min-cam-days", type=int, default=100, help="мінімум днів камери в сезоні року")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    months = range(args.months[0], args.months[1] + 1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sites = select_sites(args.years, months, args.max_missing, args.max_sites)
    (out / "sites.json").write_text(json.dumps(sites, indent=1, ensure_ascii=False))
    print(f"ділянок: {len(sites)}", flush=True)
    for s in sites:
        f = out / f"{s['site']}.json"
        if f.exists():
            continue
        try:
            cam = camera_series(s["site"], s["roi"], months)
        except Exception as e:
            print(f"{s['site']}: ряд камери недоступний ({e})", flush=True)
            continue
        years = [y for y in range(args.years[0], args.years[1] + 1)
                 if sum(d.startswith(str(y)) for d in cam) >= args.min_cam_days]
        recs = []
        for y in years:
            items = stac_point(s["lon"], s["lat"], f"{y}-{args.months[0]:02d}-01", f"{y}-{args.months[1]:02d}-31")
            with ThreadPoolExecutor(args.workers) as ex:
                recs += [r for r in ex.map(lambda it: s2_sample(it, s["lon"], s["lat"], args.radius), items) if r]
        f.write_text(json.dumps({**s, "years": years, "camera": cam, "s2": sorted(recs, key=lambda r: r["time"])}))
        n_clear = sum(r["clear_frac"] >= 0.9 for r in recs)
        print(f"{s['site']}: роки {years}, днів камери {len(cam)}, сцен S2 {len(recs)}, чистих {n_clear}", flush=True)


if __name__ == "__main__":
    main()
