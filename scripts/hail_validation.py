#!/usr/bin/env python3
"""
Перевірка конформного детектора стресу на РЕАЛЬНИХ пошкодженнях - звітах про град (ESWD).

Для кожної події (lat, lon, дата):
  1. Sentinel-2 L2A над точкою за [дата − pre_days, дата + post_days] (по одній сцені на дату);
  2. зона "event" - 3×3 фрагменти по 2,56 км навколо точки звіту; зона "control" - 8 фрагментів
     на відстані ~ctrl_km у восьми напрямках (ті самі дати);
  3. ціль - перший знімок ПІСЛЯ дати граду, де середня хмарність усіх фрагментів ≤ target_max_cloud
     (спільна для event і control - парне порівняння); опорне значення - останній чистий знімок
     перед ціллю (отже, до граду), причинний режим;
  4. причинна модель регіону прогнозує ціль, конформні пороги (α) - з калібрувальної зони регіону;
  5. частка позначених вегетуючих пікселів (NDVI(μ) > 0,3, чиста ціль) у event проти control.

Очікування: якщо град пошкодив посіви, частка тривог у event > control (контроль ≈ α).
Звіт - точка (часто населений пункт), смуга граду невідома, тож оцінка консервативна.

    python scripts/hail_validation.py --events data/eswd/hail_UA.json \
        --models results/l2a_poltava2021_lc --seeds 0 1 2 \
        --calib-root data/l2a --calib-roi poltava2021/zones/calib --out results/hail_validation
"""
import argparse
import json
import re
import shutil
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
import requests
import torch
from pyproj import Transformer
from shapely.geometry import box, mapping, shape

sys.path.insert(0, str(Path(__file__).parent))
from sen12mscrts_dataset import SEN12MSCRTSPatchDataset, n_bands, n_input_channels
from model import ResidualUNet, ResidualUNetUncertainty
from train_step2 import build_dataset
from eval_protocol import CLEAN_PIXEL_THRESH
from validation_extras import StreamingQuantile, predict
from anomaly_detection import scores, nd, VEG_NDVI, B8, B4, B8A, B5, DETECTORS
from rasterio.enums import Resampling
from fetch_s2_l2a import BAND_ASSETS, CLOUDY_SCL, STAC, read_asset, write_date

PATCH_M, RES = 2560, 20
PX = PATCH_M // RES
REF_MAX_CLOUD = 0.1  # як ref_max_cloud у SEN12MSCRTSPatchDataset


def stac_items(lon, lat, start, end, collection):
    url = STAC
    body = {"collections": [collection], "intersects": {"type": "Point", "coordinates": [lon, lat]},
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 200}
    items = []
    while True:
        r = requests.post(url, json=body, timeout=120).json()
        items += r.get("features", [])
        nxt = [l for l in r.get("links", []) if l["rel"] == "next"]
        if not nxt:
            return items
        body, url = nxt[0].get("body", body), nxt[0]["href"]


def load_models(model_dir, seed):
    in_ch, out_ch = n_input_channels(), n_bands()
    s2 = ResidualUNet(in_ch, out_ch)
    s2.load_state_dict(torch.load(Path(model_dir) / f"step2_causal_seed{seed}" / "model.pt", map_location="cpu"))
    un = ResidualUNetUncertainty(in_ch, out_ch)
    un.load_state_dict(torch.load(Path(model_dir) / f"unc_causal_seed{seed}" / "model.pt", map_location="cpu"))
    return s2.eval(), un.eval()


def thresholds(step2, unc, calib_ds, alpha):
    sq = {d: StreamingQuantile() for d in DETECTORS}
    for i in range(len(calib_ds)):
        _, t, tc, mu, _, sigma = predict(step2, unc, calib_ds, calib_ds, i, "cpu")
        m = (tc < CLEAN_PIXEL_THRESH) & (nd(mu, B8, B4) > VEG_NDVI)
        for d, v in scores(t, mu, sigma).items():
            sq[d].add(v[m])
    return {d: sq[d].conformal(alpha) for d in DETECTORS}


def layout(lon, lat, epsg, ctrl_km):
    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
    cx, cy = np.floor(x / PATCH_M) * PATCH_M, np.floor(y / PATCH_M) * PATCH_M
    ev = [(cx + i * PATCH_M, cy + j * PATCH_M) for j in (-1, 0, 1) for i in (-1, 0, 1)]
    off = round(ctrl_km * 1000 / PATCH_M) * PATCH_M
    ctrl = [(cx + i * off, cy + j * off) for j in (-1, 0, 1) for i in (-1, 0, 1) if (i, j) != (0, 0)]
    boxes = [("event", k, (a, b, a + PATCH_M, b + PATCH_M)) for k, (a, b) in enumerate(ev)]
    boxes += [("control", k, (a, b, a + PATCH_M, b + PATCH_M)) for k, (a, b) in enumerate(ctrl)]
    return boxes


def lonlat_union(boxes, epsg):
    tr = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    xs = [v for _, _, b in boxes for v in (b[0], b[2])]
    ys = [v for _, _, b in boxes for v in (b[1], b[3])]
    lo, la = tr.transform([min(xs), max(xs)], [min(ys), max(ys)])
    return box(min(lo), min(la), max(lo), max(la))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", required=True)
    ap.add_argument("--models", required=True, help="тека з step2_causal_seed*/unc_causal_seed*")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--calib-root", default="data/l2a")
    ap.add_argument("--calib-roi", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--months", type=int, nargs=2, default=[4, 9], help="лише події в цих місяцях (вегетація)")
    ap.add_argument("--pre-days", type=int, default=60)
    ap.add_argument("--post-days", type=int, default=30)
    ap.add_argument("--max-cloud", type=float, default=60)
    ap.add_argument("--target-max-cloud", type=float, default=0.3)
    ap.add_argument("--shard", type=int, default=0, help="паралельні процеси: цей бере події shard::n_shards")
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--min-control", type=int, default=4, help="мінімум контрольних фрагментів у сцені")
    ap.add_argument("--max-pre-dates", type=int, default=6, help="скільки дат до граду читати, шукаючи чисту")
    ap.add_argument("--ctrl-km", type=float, default=18)
    ap.add_argument("--collection", default="sentinel-2-l2a")
    ap.add_argument("--work", default="data/l2a_hail")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done_path = out / "per_event.jsonl"
    done = {json.loads(l)["id"] for l in done_path.read_text().splitlines()} if done_path.exists() else set()

    models = {s: load_models(args.models, s) for s in args.seeds}
    thr_path = out / "thresholds.json"
    if thr_path.exists():
        thr = {int(k): v for k, v in json.loads(thr_path.read_text()).items()}
    else:
        calib_ds = build_dataset(args.calib_root, [args.calib_roi], PX, "random_clean", 0.3, 0, near_mode="prior",
                                 samples_per_patch=8, ref_select="latest_clean")
        thr = {s: thresholds(*models[s], calib_ds, args.alpha) for s in args.seeds}
        thr_path.write_text(json.dumps(thr, indent=1))
    print("Пороги:", thr, flush=True)

    events = json.loads(Path(args.events).read_text())
    events = [e for e in events if e.get("datetime")]
    for e in events:
        e["date"] = datetime.fromisoformat(str(e["datetime"]).replace("Z", "+00:00")).date()
    events = [e for e in events if args.months[0] <= e["date"].month <= args.months[1]]
    events.sort(key=lambda e: e["date"])
    if args.max_events:
        events = events[:args.max_events]
    events = events[args.shard::args.n_shards]
    print(f"{len(events)} подій у вегетаційні місяці", flush=True)

    for e in events:
        if e["id"] in done:
            continue
        rec = {"id": e["id"], "date": str(e["date"]), "lat": e["lat"], "lon": e["lon"], "city": e.get("city")}
        start, end = e["date"] - timedelta(days=args.pre_days), e["date"] + timedelta(days=args.post_days)
        items = stac_items(e["lon"], e["lat"], start, end, args.collection)
        # окремі сцени earth-search досі вказують на JP2 у бакеті requester-pays - лише COG
        items = [i for i in items if all(i["assets"].get(a, {}).get("href", "").endswith(".tif")
                                         for a in BAND_ASSETS + ["scl"] if a)]
        if not items:
            rec["skip"] = "немає сцен"
            done_path.open("a").write(json.dumps(rec, ensure_ascii=False) + "\n")
            continue
        # Зона граду (3 × 3) мусить бути цілком у сцені; контрольні фрагменти рознесені на ~18 км і біля
        # краю тайла частина з них випадає - такі фрагменти відкидаються (лишаються ті, що покриті щонайменше
        # у 90 % придатних дат), а дати беруться лише ті, де покрито всі залишені фрагменти.
        epsg_of = lambda i: int(str(i["properties"].get("proj:code", i["properties"].get("proj:epsg"))).split(":")[-1])
        cands = {}
        for g in {epsg_of(i) for i in items}:
            bx = layout(e["lon"], e["lat"], g, args.ctrl_km)
            polys = [lonlat_union([b_], g) for b_ in bx]
            rows = []
            for it in (i for i in items if epsg_of(i) == g):
                fp = shape(it["geometry"])
                cov = [fp.intersection(pp).area / pp.area >= 0.99 for pp in polys]
                if all(c for (zn, _, _), c in zip(bx, cov) if zn == "event") and \
                        it["properties"].get("eo:cloud_cover", 100) <= args.max_cloud:
                    rows.append((it, cov))
            cands[g] = (bx, rows)
        epsg = max(cands, key=lambda g: len({r[0]["properties"]["datetime"][:10] for r in cands[g][1]}))
        bx, rows = cands[epsg]
        ctrl_ids = [j for j, (zn, _, _) in enumerate(bx) if zn == "control"]
        keep = [j for j in ctrl_ids if rows and sum(r[1][j] for r in rows) >= 0.9 * len(rows)]
        sel = [j for j, (zn, _, _) in enumerate(bx) if zn == "event"] + keep
        boxes = [(zn, n, b_) for n, (zn, _, b_) in enumerate(bx[j] for j in sel if bx[j][0] == "event")]
        boxes += [("control", n, bx[j][2]) for n, j in enumerate(keep)]
        n_ctrl = len(keep)
        rec["n_control"] = n_ctrl
        best = {}
        for it, cov in rows:
            if not all(cov[j] for j in sel):
                continue
            d = it["properties"]["datetime"][:10]
            key = -it["properties"].get("eo:cloud_cover", 100)
            if d not in best or key > best[d][0]:
                best[d] = (key, it)
        dates = [(d, v[1]) for d, v in sorted(best.items())]
        if n_ctrl < args.min_control:
            rec["skip"] = f"мало контрольних фрагментів у сцені: {n_ctrl}"
            done_path.open("a").write(json.dumps(rec, ensure_ascii=False) + "\n")
            continue
        pre = [d for d, _ in dates if date.fromisoformat(d) < e["date"]]
        post = [d for d, _ in dates if date.fromisoformat(d) > e["date"]]
        if len(pre) < 2 or not post:
            rec["skip"] = f"мало дат: до {len(pre)}, після {len(post)}"
            done_path.open("a").write(json.dumps(rec, ensure_ascii=False) + "\n")
            continue

        # Моделі потрібні лише ціль (перший чистий знімок після граду), найпізніший чистий знімок до неї
        # (опорне значення) і найближча дата перед ціллю, тож дати читаються ліниво: вперед від граду до
        # першої чистої, назад від граду до першої чистої. Повне завантаження 90 діб - ~30 хв на подію.
        region_dir = Path(args.work) / ("ev_" + re.sub(r"[^0-9A-Za-z_-]+", "_", e["id"])[:60])
        got, cld, bad = [], {}, 0

        def fetch(d, it):
            nonlocal bad
            t = len(got)
            try:
                write_date(t, d, it, boxes, region_dir, epsg, PX, RES)
            except Exception:  # пошкоджений чи недоступний файл сцени: відкидаємо лише цю дату
                bad += 1
                for p in (region_dir / "zones").glob(f"*/*/{t}"):
                    shutil.rmtree(p, ignore_errors=True)
                return None
            got.append((d, it))
            cld[t] = float(np.mean([np.mean(rasterio.open(p).read(1) / 100.0)
                                    for p in (region_dir / "zones").glob(f"*/CLD/{t}/*.tif")]))
            return cld[t]

        def scl_cloud(it):
            """Хмарність зон лише за маскою SCL (один файл замість 14) - для відбору дат."""
            try:
                scl = read_asset(it["assets"]["scl"]["href"], [b for _, _, b in boxes], PX, Resampling.nearest, None, attempts=2)
            except Exception:
                return None
            return float(np.isin(scl, CLOUDY_SCL).mean())

        post = [(d, it) for d, it in dates if date.fromisoformat(d) > e["date"]]
        pre = [(d, it) for d, it in dates if date.fromisoformat(d) < e["date"]]
        tgt, prev = None, None  # prev - найближча дата перед ціллю (x_near моделі)
        for j, (d, it) in enumerate(post):
            c = scl_cloud(it)
            if c is not None and c <= args.target_max_cloud:
                c = fetch(d, it)
                if c is not None and c <= args.target_max_cloud:
                    tgt = len(got) - 1
                    prev = post[j - 1] if j > 0 else None
                    break
        n_pre = 0
        if tgt is not None:
            ref = None
            for d, it in list(reversed(pre))[:args.max_pre_dates]:
                c = scl_cloud(it)
                if c is not None and c <= REF_MAX_CLOUD:
                    ref = (d, it)
                    break
            for d, it in {x[0]: x for x in (prev, ref, pre[-1] if pre else None) if x}.values():
                c = fetch(d, it)
                n_pre += c is not None and date.fromisoformat(d) < e["date"]
        dates = got
        if bad:
            rec["unreadable_dates"] = bad
        if tgt is not None and n_pre == 0:
            rec["skip"] = "немає прочитаних знімків до граду"
            done_path.open("a").write(json.dumps(rec, ensure_ascii=False) + "\n")
            shutil.rmtree(region_dir, ignore_errors=True)
            continue
        post_t = [] if tgt is None else [tgt]
        if not post_t:
            rec["skip"] = "немає чистого знімка після граду"
        else:
            tgt = post_t[0]
            rec["target_date"] = dates[tgt][0]
            rec["days_after"] = (date.fromisoformat(dates[tgt][0]) - e["date"]).days
            dsets = {}
            for zn, n in (("event", 9), ("control", n_ctrl)):
                dsets[zn] = SEN12MSCRTSPatchDataset(root=region_dir, roi_dir="zones", roi=zn, patch_indices=range(n),
                                                    img_size=PX, target_mode="random_clean", target_max_cloud=0.3,
                                                    seed=0, near_mode="prior", ref_select="latest_clean",
                                                    forced_targets={k: tgt for k in range(n)})
            p0 = dsets["event"].pick(0)
            rec["ref_date"] = str(SEN12MSCRTSPatchDataset._day_to_date(p0["t_days"][p0["ref_idx"]]))
            rec["ref_before_event"] = rec["ref_date"] < rec["date"]
            rec["seeds"] = {}
            for s, (s2, un) in models.items():
                res = {}
                for zn, ds in dsets.items():
                    flagged = {d: 0 for d in DETECTORS}
                    n_px, dndvi, dndre = 0, [], []
                    for i in range(len(ds)):
                        _, t_, tc, mu, _, sigma = predict(s2, un, ds, ds, i, "cpu")
                        m = (tc < CLEAN_PIXEL_THRESH) & (nd(mu, B8, B4) > VEG_NDVI)
                        if not m.any():
                            continue
                        sc = scores(t_, mu, sigma)
                        for d in DETECTORS:
                            flagged[d] += int((sc[d][m] > thr[s][d]).sum())
                        n_px += int(m.sum())
                        dndvi.append(float((nd(t_, B8, B4) - nd(mu, B8, B4))[m].mean()))
                        dndre.append(float((nd(t_, B8A, B5) - nd(mu, B8A, B5))[m].mean()))
                    res[zn] = {"n_veg_px": n_px,
                               "frac_flagged": {d: (flagged[d] / n_px if n_px else None) for d in DETECTORS},
                               "obs_minus_pred_ndvi": float(np.mean(dndvi)) if dndvi else None,
                               "obs_minus_pred_ndre": float(np.mean(dndre)) if dndre else None}
                rec["seeds"][str(s)] = res
            ev, ct = rec["seeds"][str(args.seeds[0])]["event"], rec["seeds"][str(args.seeds[0])]["control"]
            print(f"{rec['date']} {rec.get('city')}: ціль {rec['target_date']} (+{rec['days_after']} діб), "
                  f"тривоги event/control NDRE {ev['frac_flagged']['ndre']} / {ct['frac_flagged']['ndre']}", flush=True)
        done_path.open("a").write(json.dumps(rec, ensure_ascii=False) + "\n")
        if not args.keep_raw:
            shutil.rmtree(region_dir, ignore_errors=True)

    summarize(done_path, out, args.seeds, args.alpha)


def summarize(done_path, out, seeds, alpha):
    recs = [json.loads(l) for l in done_path.read_text().splitlines()]
    ok = [r for r in recs if "seeds" in r and r.get("ref_before_event")]
    lines = [f"# Перевірка детектора на звітах про град (ESWD)", "",
             f"Подій усього: {len(recs)}; придатних (є чистий знімок до і після, опорне значення до граду): {len(ok)}.",
             f"Пропущено: " + ", ".join(sorted({r.get('skip', '') for r in recs if 'skip' in r})), ""]
    if ok:
        rng = np.random.default_rng(0)
        lines += [f"| Детектор | Тривоги в зоні граду | Тривоги в контролі (ціль ≈ {alpha}) | Різниця [95 % ДІ, бутстреп за подіями] | Подій, де град > контролю |",
                  "|---|---|---|---|---|"]
        for d in DETECTORS:
            pairs = []
            for r in ok:
                val = lambda z, s: r["seeds"][str(s)][z]["frac_flagged"][d]
                ev = np.nanmean([np.nan if val("event", s) is None else val("event", s) for s in seeds])
                ct = np.nanmean([np.nan if val("control", s) is None else val("control", s) for s in seeds])
                if not (np.isnan(ev) or np.isnan(ct)):
                    pairs.append((ev, ct))
            if not pairs:
                continue
            a = np.array(pairs)
            diff = a[:, 0] - a[:, 1]
            bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
            lines.append(f"| {d} | {a[:, 0].mean():.3f} | {a[:, 1].mean():.3f} | {diff.mean():+.3f} "
                         f"[{np.percentile(bs, 2.5):+.3f}; {np.percentile(bs, 97.5):+.3f}] | {(diff > 0).sum()}/{len(diff)} |")
        # Таблиця по подіях (дата, місце, координати) дозволяє відтворити сирі звіти ESWD, а угода ESSL
        # (п. iv, ix) забороняє їх публікувати: вона йде в окремий файл, який .gitignore не пускає в репозиторій.
        lines.append("")
        lines.append("Джерело: European Severe Weather Database, ESSL (eswd.eu); Dotzek et al. (2009).")
        private = ["| Дата | Місце | Ціль (+діб) | NDRE: град / контроль | NDVI: град / контроль | ΔNDVI спостер.−прогноз: град / контроль |",
                  "|---|---|---|---|---|---|"]
        for r in ok:
            s0 = r["seeds"][str(seeds[0])]
            f = lambda z, d: s0[z]["frac_flagged"][d]
            g = lambda z: s0[z]["obs_minus_pred_ndvi"]
            fmt = lambda v: "-" if v is None else f"{v:.3f}"
            private.append(f"| {r['date']} | {r.get('city') or ''} ({r['lat']:.3f}, {r['lon']:.3f}) | {r['target_date']} (+{r['days_after']}) | "
                         f"{fmt(f('event', 'ndre'))} / {fmt(f('control', 'ndre'))} | {fmt(f('event', 'ndvi'))} / {fmt(f('control', 'ndvi'))} | "
                         f"{fmt(g('event'))} / {fmt(g('control'))} |")
        (out / "events_private.md").write_text("\n".join(private) + "\n")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
