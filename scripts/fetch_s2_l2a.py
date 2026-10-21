#!/usr/bin/env python3
"""
Власний пайплайн Sentinel-2 L2A (поверхневе відбиття + маска SCL) з відкритого STAC-каталогу
Element84 Earth Search (без реєстрації) у структуру, сумісну з sen12mscrts_dataset.py:

    <out>/<region>/zones/<zone>/S2/<tdx>/s2_l2a_<zone>_ImgNo_<tdx>_<YYYY-MM-DD>_patch_<kkkk>.tif
    <out>/<region>/zones/<zone>/CLD/<tdx>/<те саме ім'я>.tif     (0 - чисто, 100 - хмара/тінь/no-data)

ROI-специфікація для train_step2.py тощо: --root <out> --train-roi <region>/zones/<zone>.

Відмінності від SEN12MS-CR-TS (L1C, ~30 кадрів, маска - поріг яскравості):
  - L2A (поверхневе відбиття), шкала й зсув - з raster:bands кожного asset (враховує BOA offset);
  - усі дати сезону (не 30 відібраних), по одній сцені на дату (найбільше покриття зон);
  - хмарна маска з SCL: класи 0 no-data, 1 saturated, 3 тінь, 8/9 хмара, 10 цирус -> 100;
  - 13 каналів у порядку SEN12MS (B1..B12 з B8A); B10 у L2A відсутній - заповнено нулями.

Фрагменти: квадрати patch_m метрів (за замовчуванням 2560 м -> 128 пікс. при 20 м) у фіксованій
UTM-сітці тайла, випадкова вибірка n фрагментів у межах кожної зони (детерміновано за --seed).
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from shapely.geometry import box, shape

STAC = "https://earth-search.aws.element84.com/v1/search"
BAND_ASSETS = ["coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08",
               "nir09", None, "swir16", "swir22"]  # None = B10 (немає в L2A)
CLOUDY_SCL = (0, 1, 3, 8, 9, 10)
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                GDAL_HTTP_MAX_RETRY="6", GDAL_HTTP_RETRY_DELAY="3", GDAL_HTTP_MULTIPLEX="NO", GDAL_HTTP_TIMEOUT="60",
                VSI_CACHE="TRUE", GDAL_CACHEMAX=512)


def search(collection, grid, start, end):
    url, body = STAC, {"collections": [collection], "query": {"grid:code": {"eq": grid}},
                       "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 200}
    items = []
    while True:
        r = requests.post(url, json=body, timeout=120).json()
        items += r["features"]
        nxt = [l for l in r.get("links", []) if l["rel"] == "next"]
        if not nxt:
            return items
        body, url = nxt[0].get("body", body), nxt[0]["href"]


def pick_per_date(items, zones_union, max_cloud):
    best = {}
    for it in items:
        d = it["properties"]["datetime"][:10]
        cover = shape(it["geometry"]).intersection(zones_union).area / zones_union.area
        key = (cover, -it["properties"].get("eo:cloud_cover", 100))
        if d not in best or key > best[d][0]:
            best[d] = (key, it)
    out = [(d, v[1]) for d, v in sorted(best.items())
           if v[1]["properties"].get("eo:cloud_cover", 100) <= max_cloud and v[0][0] > 0.5]
    return out


def sample_cells(zones, epsg, tile_bounds, patch_m, seed):
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    rng = np.random.default_rng(seed)
    cells = {}
    for name, z in zones.items():
        xs, ys = to_utm.transform([z["bbox"][0], z["bbox"][2], z["bbox"][0], z["bbox"][2]],
                                  [z["bbox"][1], z["bbox"][1], z["bbox"][3], z["bbox"][3]])
        x0 = np.ceil(max(min(xs), tile_bounds[0] + patch_m) / patch_m) * patch_m
        y0 = np.ceil(max(min(ys), tile_bounds[1] + patch_m) / patch_m) * patch_m
        x1 = min(max(xs), tile_bounds[2] - patch_m)
        y1 = min(max(ys), tile_bounds[3] - patch_m)
        grid = [(x, y) for x in np.arange(x0, x1 - patch_m + 1, patch_m)
                for y in np.arange(y0, y1 - patch_m + 1, patch_m)]
        if len(grid) < z["n"]:
            raise SystemExit(f"зона {name}: лише {len(grid)} клітинок, потрібно {z['n']}")
        sel = rng.choice(len(grid), z["n"], replace=False)
        cells[name] = [tuple(map(float, grid[i])) for i in sorted(sel)]
    return cells


def read_asset(href, boxes, px, resampling, scale_offset, attempts=6):
    # S3 іноді обриває окремі range-запити (RasterioIOError: Read failed) - повторюємо весь asset.
    for attempt in range(1, attempts + 1):
        try:
            with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src:
                out = []
                for (x, y, x2, y2) in boxes:
                    w = from_bounds(x, y, x2, y2, src.transform)
                    out.append(src.read(1, window=w, out_shape=(px, px), resampling=resampling,
                                        boundless=True, fill_value=0))
            break
        except rasterio.errors.RasterioIOError as e:
            if attempt == attempts:
                raise
            print(f"  повтор {attempt}/{attempts - 1} для {href.rsplit('/', 1)[-1]}: {e}", flush=True)
            time.sleep(15 * attempt)
    arr = np.stack(out)
    if scale_offset is None:
        return arr
    scale, offset = scale_offset
    valid = arr > 0
    refl = np.where(valid, arr.astype(np.float32) * scale + offset, 0.0)
    return np.clip(np.round(refl * 10000), 0, 65535).astype(np.uint16), valid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zones", required=True, help="JSON: {zone: {bbox: [lon0,lat0,lon1,lat1], n: patches}}")
    ap.add_argument("--grid", required=True, help="напр. MGRS-21HUB")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--start", default=None, help="YYYY-MM-DD, за замовчуванням 1 січня --year")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD, за замовчуванням 31 грудня --year")
    ap.add_argument("--region", required=True)
    ap.add_argument("--out", default="data/l2a")
    ap.add_argument("--collection", default="sentinel-2-l2a")
    ap.add_argument("--max-cloud", type=float, default=80, help="макс. хмарність сцени (eo:cloud_cover), %%")
    ap.add_argument("--patch-m", type=int, default=2560)
    ap.add_argument("--res", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-dates", type=int, default=None, help="для перевірки швидкості")
    args = ap.parse_args()

    zones = json.loads(Path(args.zones).read_text()) if Path(args.zones).exists() else json.loads(args.zones)
    zones_union = box(*zones[next(iter(zones))]["bbox"])
    for z in zones.values():
        zones_union = zones_union.union(box(*z["bbox"]))

    start = args.start or f"{args.year}-01-01"
    end = args.end or f"{args.year}-12-31"
    items = search(args.collection, args.grid, start, end)
    dates = pick_per_date(items, zones_union, args.max_cloud)
    if args.max_dates:
        dates = dates[:args.max_dates]
    print(f"{len(items)} сцен, {len(dates)} дат після відбору (хмарність ≤ {args.max_cloud}%)", flush=True)

    ref = dates[0][1]
    epsg = int(str(ref["properties"].get("proj:code", ref["properties"].get("proj:epsg"))).split(":")[-1])
    red = ref["assets"]["red"]
    tb = red.get("proj:bbox") or ref["properties"].get("proj:bbox")
    if tb is None:
        with rasterio.Env(**GDAL_ENV), rasterio.open(red["href"]) as src:
            tb = list(src.bounds)
    cells = sample_cells(zones, epsg, tb, args.patch_m, args.seed)
    px = args.patch_m // args.res

    region_dir = Path(args.out) / args.region
    manifest = {"collection": args.collection, "grid": args.grid, "year": args.year, "start": start, "end": end,
                "epsg": epsg,
                "patch_m": args.patch_m, "res_m": args.res, "zones": zones, "cells": cells,
                "dates": [{"tdx": t, "date": d, "id": it["id"], "cloud": it["properties"].get("eo:cloud_cover")}
                          for t, (d, it) in enumerate(dates)]}
    region_dir.mkdir(parents=True, exist_ok=True)
    (region_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    all_boxes = [(zn, k, (x, y, x + args.patch_m, y + args.patch_m))
                 for zn in cells for k, (x, y) in enumerate(cells[zn])]
    boxes = [b for _, _, b in all_boxes]

    for t, (d, it) in enumerate(dates):
        def path(kind, zn, k):
            return region_dir / "zones" / zn / kind / str(t) / f"s2_l2a_{zn}_ImgNo_{t}_{d}_patch_{k:04d}.tif"
        if all(path("CLD", zn, k).exists() for zn, k, _ in all_boxes):
            continue

        def job(asset):
            if asset is None:
                return None
            a = it["assets"][asset]
            rb = (a.get("raster:bands") or [{}])[0]
            res_a = rb.get("spatial_resolution", 20)
            rs = Resampling.average if res_a < args.res else Resampling.nearest
            if asset == "scl":
                return read_asset(a["href"], boxes, px, Resampling.nearest, None)
            return read_asset(a["href"], boxes, px, rs, (rb.get("scale", 1e-4), rb.get("offset", 0.0)))

        with ThreadPoolExecutor(max_workers=6) as ex:
            res = list(ex.map(job, BAND_ASSETS + ["scl"]))
        scl = res[-1]
        bands, valid = [], np.ones_like(scl, dtype=bool)
        for r in res[:-1]:
            if r is None:
                bands.append(np.zeros_like(scl, dtype=np.uint16))
            else:
                bands.append(r[0])
                valid &= r[1]
        stack = np.stack(bands, axis=1)  # (N, 13, px, px)
        cloud = (np.isin(scl, CLOUDY_SCL) | ~valid).astype(np.uint8) * 100

        for i, (zn, k, (x, y, x2, y2)) in enumerate(all_boxes):
            prof = dict(driver="GTiff", width=px, height=px, crs=f"EPSG:{epsg}",
                        transform=from_origin(x, y2, args.res, args.res), compress="deflate")
            for kind, arr, dtype, pred in (("S2", stack[i], "uint16", 2), ("CLD", cloud[i][None], "uint8", 1)):
                p = path(kind, zn, k)
                p.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(p, "w", count=arr.shape[0], dtype=dtype, predictor=pred, **prof) as dst:
                    dst.write(arr)
        print(f"[{t + 1}/{len(dates)}] {d}  хмарність сцени {it['properties'].get('eo:cloud_cover', -1):.0f}%  "
              f"частка хмари у фрагментах {cloud.mean() / 100:.2f}", flush=True)
    print("Готово:", region_dir)


if __name__ == "__main__":
    main()
