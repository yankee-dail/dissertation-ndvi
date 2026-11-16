#!/usr/bin/env python3
"""
Погода як джерело даних про поле: прогноз фенологічних дат (SOS, EOS) за сумою активних температур
з конформним інтервалом; «наземна правда» - дати за щоденним рядом камери PhenoCam (розділ 9.6).

Погода - добові дані реаналізу ERA5 через архівний API Open-Meteo для координат камери (кешуються в
data/weather/<site>.json). Сума активних температур GDD = Σ max(0, T_сер − 5 °C) від 1 березня.

Для кожної ділянки з ≥ 3 роками дат камери - прогноз «з виключенням року»:
  календар - середній день року дати в інші роки;
  GDD      - день, коли GDD цього року досягає середнього GDD на дату в інші роки
             (прогноз можна видати, щойно сума накопичилася, тобто в реальному часі).
Конформний 90 % інтервал - квантиль |похибки| на половині ділянок, перевірка на іншій (два розбиття, MD5 назви).

    python scripts/weather_phenology.py --data data/phenocam --agreement results/phenocam/summary.json \
        --cache data/weather --out results/weather_phenology
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np

from phenocam_analysis import conf_q, load, split
from phenology import smooth, transitions

API = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={a}"
       "&end_date={b}&daily=temperature_2m_mean,precipitation_sum&timezone=UTC")


def weather(site, cache):
    f = cache / f"{site['site']}.json"
    if f.exists():
        return json.loads(f.read_text())
    a, b = f"{min(site['years'])}-01-01", f"{max(site['years'])}-12-31"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(API.format(lat=site["lat"], lon=site["lon"], a=a, b=b), timeout=60) as r:
                d = json.loads(r.read())["daily"]
            break
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(60 * (attempt + 1))  # ліміт безкоштовного API: чекаємо й повторюємо
    else:
        raise RuntimeError("Open-Meteo: перевищено ліміт запитів")
    f.write_text(json.dumps(d))
    time.sleep(5)
    return d


def gdd_series(w, year, base=5.0):
    """-> {ordinal: GDD від 1 березня}"""
    out, acc = {}, 0.0
    for d, t in zip(w["time"], w["temperature_2m_mean"]):
        dd = date.fromisoformat(d)
        if dd.year != year or dd < date(year, 3, 1) or t is None:
            continue
        acc += max(0.0, t - base)
        out[dd.toordinal()] = acc
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/phenocam")
    ap.add_argument("--agreement", default="results/phenocam/summary.json")
    ap.add_argument("--cache", default="data/weather")
    ap.add_argument("--index", default="ndvi")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out, cache = Path(args.out), Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    agr = json.loads(Path(args.agreement).read_text())["agreement"]
    good = {s for s, a in agr.items() if a and a["r"] >= 0.5}

    obs = []  # (site, year, kind, ordinal, gdd at date)
    for s in load(args.data, 0.9, args.index):
        if s["site"] not in good:
            continue
        w = weather(s, cache)
        for y in s["years"]:
            lo, hi = date(y, 3, 1).toordinal(), date(y, 11, 30).toordinal()
            cd = sorted(t for t in s["cam"] if lo <= t <= hi)
            if len(cd) < 150:
                continue
            tr = transitions(np.array(cd, float), smooth(cd, [s["cam"][t] for t in cd]))
            if tr is None:
                continue
            g = gdd_series(w, y)
            for k, kk in ((0, "sos"), (1, "eos")):
                if tr[k] is None:
                    continue
                t = int(round(tr[k][0]))
                if t in g:
                    obs.append({"site": s["site"], "year": y, "kind": kk, "doy": date.fromordinal(t).timetuple().tm_yday,
                                "t": t, "gdd": g[t], "gdd_series": g})
    res, L = {}, ["# Прогноз фенологічних дат поля за погодою (GDD, ERA5) і календарем", ""]
    for kk in ("sos", "eos"):
        recs = []
        R = [o for o in obs if o["kind"] == kk]
        sites = sorted({o["site"] for o in R})
        for site in sites:
            S = [o for o in R if o["site"] == site]
            if len(S) < 3:
                continue
            for o in S:
                others = [x for x in S if x is not o]
                cal = np.mean([x["doy"] for x in others])
                target = np.mean([x["gdd"] for x in others])
                reach = [t for t, v in sorted(o["gdd_series"].items()) if v >= target]
                if not reach:
                    continue
                gdd_doy = date.fromordinal(reach[0]).timetuple().tm_yday
                recs.append({"site": site, "year": o["year"], "e_cal": o["doy"] - cal, "e_gdd": o["doy"] - gdd_doy})
        if len(recs) < 10:
            continue
        r = {"n": len(recs), "sites": len({x["site"] for x in recs})}
        for m in ("cal", "gdd"):
            e = np.array([x[f"e_{m}"] for x in recs])
            hit, hw = [], []
            for fold in (0, 1):
                q = conf_q([abs(x[f"e_{m}"]) for x in recs if split(x["site"]) != fold], args.alpha)
                for x in recs:
                    if split(x["site"]) == fold:
                        hit.append(abs(x[f"e_{m}"]) <= q)
                        hw.append(q)
            r[m] = {"mae": float(np.mean(np.abs(e))), "median_abs": float(np.median(np.abs(e))),
                    "coverage": float(np.mean(hit)), "halfwidth": float(np.mean(hw))}
        better = np.mean([abs(x["e_gdd"]) < abs(x["e_cal"]) for x in recs])
        r["gdd_better_share"] = float(better)
        res[kk] = r
    L += ["| Дата | Ділянко-років (ділянок) | MAE календар, діб | MAE GDD, діб | Медіана |e| календар / GDD | "
          "Покриття календар / GDD | Півширина календар / GDD | Частка, де GDD точніша |", "|---|---|---|---|---|---|---|---|"]
    for kk, r in res.items():
        L.append(f"| {kk.upper()} | {r['n']} ({r['sites']}) | {r['cal']['mae']:.1f} | {r['gdd']['mae']:.1f} | "
                 f"{r['cal']['median_abs']:.1f} / {r['gdd']['median_abs']:.1f} | {r['cal']['coverage']:.2f} / {r['gdd']['coverage']:.2f} | "
                 f"{r['cal']['halfwidth']:.1f} / {r['gdd']['halfwidth']:.1f} | {r['gdd_better_share']:.2f} |")
    (out / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
