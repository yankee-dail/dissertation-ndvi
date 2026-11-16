#!/usr/bin/env python3
"""
Розділ 11.4: приклад інтегрованої технології на одній ділянці - щоденна оцінка NDVI поля з 90 % інтервалом
у хмарні періоди (останній чистий знімок Sentinel-2 + зміна ряду наземної камери, формула 9.1) і тривоги
детектора різких змін (розділ 10.2).

Інтервали й поріг детектора калібруються на ВСІХ ІНШИХ узгоджених ділянках (ділянка прикладу в калібрування
не входить). Для інтервалу - Мондріанівське калібрування за тривалістю пропуску (діб від останньої чистої дати).

    python scripts/field_example.py --site institutekarnobat --year 2023 --index gcc --out results/field_example
"""
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from phenocam_analysis import cam_at, conf_q, load, predictions
from phenocam_events import load as load_events
from phenocam_events import residuals

BINS = ((1, 10), (11, 20), (21, 60))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="institutekarnobat")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--index", choices=["gcc", "ndvi"], default="gcc")
    ap.add_argument("--months", type=int, nargs=2, default=[4, 7])
    ap.add_argument("--data", default="data/phenocam")
    ap.add_argument("--agreement", default="results/phenocam/summary.json")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--alpha-event", type=float, default=0.05)
    ap.add_argument("--min-fit", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    agr = json.loads(Path(args.agreement).read_text())["agreement"]
    good = {s for s, a in agr.items() if a and a["r"] >= 0.5}
    sites = {s["site"]: s for s in load(args.data, 0.9, args.index) if s["site"] in good}
    site = sites.pop(args.site)

    # калібрування інтервалів на інших ділянках (природні пропуски), окремо за тривалістю пропуску
    recs = [r for s in sites.values() for r in predictions(s, args.min_fit)
            if r["pred"]["cam_delta"] is not None]
    q = {}
    for m in ("persist", "cam_delta"):
        for lo, hi in BINS:
            res = [abs(r["pred"][m] - r["truth"]) for r in recs if lo <= r["gap"] <= hi]
            q[(m, lo)] = conf_q(res, args.alpha)
    # поріг детектора на інших ділянках
    ev_other = [load_events(Path(args.data) / f"{s}.json", args.index, 0.9) for s in sites]
    q_ev = conf_q([v for s in ev_other if s["amp"] for v in residuals(s["cam"], s["amp"]).values()],
                  args.alpha_event)
    ev_site = load_events(Path(args.data) / f"{args.site}.json", args.index, 0.9)
    alarms = {t for t, v in residuals(ev_site["cam"], ev_site["amp"]).items() if v > q_ev}

    y = args.year
    start = date(y, args.months[0], 1).toordinal()
    end = date(y, args.months[1] + 1, 1).toordinal() - 1
    clear = sorted(t for t in site["s2"] if date.fromordinal(t).year == y)
    rows = []
    for t in range(start, end + 1):
        before = [u for u in clear if u < t]
        row = {"date": date.fromordinal(t).isoformat(), "s2": site["s2"].get(t),
               "alarm": t in alarms and date.fromordinal(t).month <= args.months[1]}
        if before:
            t0 = before[-1]
            gap = t - t0
            b = next((lo for lo, hi in BINS if lo <= gap <= hi), None)
            fit = [(c, site["s2"][u]) for u in before if (c := cam_at(site["cam"], u)) is not None]
            ct, c0 = cam_at(site["cam"], t), cam_at(site["cam"], t0)
            row.update({"gap": gap, "persist": site["s2"][t0]})
            if b is not None:
                row["q_persist"] = q[("persist", b)]
            if len(fit) >= args.min_fit and ct is not None and c0 is not None and b is not None:
                x, yy = np.array(fit).T
                a = np.polyfit(x, yy, 1)[0]
                row.update({"fused": site["s2"][t0] + a * (ct - c0), "q_fused": q[("cam_delta", b)]})
        rows.append(row)
    # перевірка на чистих датах, що завершують довгі пропуски
    checks = []
    for i, t in enumerate(clear):
        if i == 0 or not (start <= t <= end) or t - clear[i - 1] < 15:
            continue
        r = next(r for r in rows if r["date"] == date.fromordinal(t).isoformat())
        checks.append({"date": r["date"], "gap": t - clear[i - 1], "truth": r["s2"], "persist": r["persist"],
                       "q_persist": r.get("q_persist"), "fused": r.get("fused"), "q_fused": r.get("q_fused")})
    summary = {"site": args.site, "year": y, "index": args.index,
               "q": {f"{m}/{lo}": v for (m, lo), v in q.items()}, "q_event": q_ev,
               "n_calib_sites": len(sites), "checks": checks,
               "alarms": sorted(r["date"] for r in rows if r["alarm"]),
               "n_clear_in_window": sum(start <= t <= end for t in clear)}
    (out / "series.json").write_text(json.dumps(rows, ensure_ascii=False))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
