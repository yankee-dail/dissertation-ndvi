#!/usr/bin/env python3
"""
Розділ 10.2 / 11: справжні (не штучні) різкі зміни стану поля і тривоги наземної камери.

Справжня подія - падіння NDVI Sentinel-2 між двома сусідніми чистими датами t_a < t_b більше ніж на --drop
за пропуску ≤ --max-gap діб (дозрівання, збирання, скошування, вилягання, ураження). Для кожної події:
    чи була тривога детектора камери (розділ 10.2, поріг калібровано на іншій половині ділянок, α) у (t_a, t_b];
    випередження = t_b − день першої тривоги (на скільки діб раніше за супутниковий знімок, що показав падіння).
Контроль: «тихі» пари сусідніх чистих дат зі зміною NDVI менше ±0,03 - частка таких проміжків із тривогою.
Детектор використовує лише минулі дні (медіана 7 попередніх діб), тож усе причинне.

    python scripts/real_events.py --data data/phenocam --agreement results/phenocam/summary.json --out results/real_events
"""
import argparse
import json
from pathlib import Path

import numpy as np

from phenocam_events import conf_q, load, residuals, split


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/phenocam")
    ap.add_argument("--agreement", default="results/phenocam/summary.json")
    ap.add_argument("--min-r", type=float, default=0.5)
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--max-gap", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    agr = json.loads(Path(args.agreement).read_text())["agreement"]
    good = sorted(s for s, a in agr.items() if a and a["r"] >= args.min_r)
    summary, L = {}, [f"# Справжні падіння NDVI Sentinel-2 (> {args.drop} за ≤ {args.max_gap} діб) і тривоги камери "
                      f"(α = {args.alpha})", ""]
    for index in ("gcc", "ndvi"):
        sites = [s for s in (load(Path(args.data) / f"{g}.json", index, 0.9) for g in good) if s["amp"]]
        res = {s["site"]: residuals(s["cam"], s["amp"]) for s in sites}
        s2 = {}
        for g in good:
            d = json.loads((Path(args.data) / f"{g}.json").read_text())
            v = {}
            for r in d["s2"]:
                if r["clear_frac"] >= 0.9 and "ndvi_clear" in r and -0.2 < r["ndvi_clear"] <= 1:
                    from datetime import date
                    v.setdefault(date.fromisoformat(r["date"]).toordinal(), []).append(r["ndvi_clear"])
            s2[g] = {k: float(np.mean(x)) for k, x in v.items()}
        ev, quiet = [], []
        for fold in (0, 1):
            q = conf_q([x for s, r in res.items() if split(s) != fold for x in r.values()], args.alpha)
            for s in sites:
                if split(s["site"]) != fold:
                    continue
                alarms = {t for t, x in res[s["site"]].items() if x > q}
                ts = sorted(s2[s["site"]])
                for a, b in zip(ts, ts[1:]):
                    if b - a > args.max_gap:
                        continue
                    covered = sum(1 for t in range(a + 1, b + 1) if t in res[s["site"]])
                    if covered < 0.8 * (b - a):
                        continue
                    hits = sorted(t for t in alarms if a < t <= b)
                    d = s2[s["site"]][a] - s2[s["site"]][b]
                    if d > args.drop:
                        ev.append({"site": s["site"], "gap": b - a, "drop": d, "hit": bool(hits),
                                   "lead": (b - hits[0]) if hits else None})
                    elif abs(d) < 0.03:
                        quiet.append(bool(hits))
        leads = [e["lead"] for e in ev if e["hit"]]
        r = {"events": len(ev), "sites": len({e["site"] for e in ev}), "detected": float(np.mean([e["hit"] for e in ev])),
             "lead_median": float(np.median(leads)) if leads else None,
             "lead_ge1_share_of_events": float(np.mean([e["hit"] and e["lead"] >= 1 for e in ev])),
             "gap_median": float(np.median([e["gap"] for e in ev])),
             "quiet_intervals": len(quiet), "quiet_alarm_share": float(np.mean(quiet)),
             "detected_by_drop": {f"{lo}-{hi}": float(np.mean([e["hit"] for e in ev if lo < e["drop"] <= hi]))
                                  for lo, hi in ((args.drop, 0.25), (0.25, 0.4), (0.4, 2)) if any(lo < e["drop"] <= hi for e in ev)}}
        summary[index] = r
        L += [f"## Камера {index.upper()}", "",
              f"Подій {r['events']} на {r['sites']} ділянках (медіана пропуску між знімками {r['gap_median']:.0f} діб). "
              f"Тривога камери в проміжку між знімками: {r['detected']:.2f} подій; раніше за знімок, що показав падіння, "
              f"щонайменше на добу: {r['lead_ge1_share_of_events']:.2f}; медіана випередження {r['lead_median']} діб. "
              f"За силою падіння: " + ", ".join(f"{k}: {v:.2f}" for k, v in r["detected_by_drop"].items()) + ". "
              f"Контроль - «тихі» проміжки (|ΔNDVI| < 0,03): {r['quiet_intervals']}, з тривогою {r['quiet_alarm_share']:.2f}.", ""]
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
