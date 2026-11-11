#!/usr/bin/env python3
"""
Агреговані підгрупи перевірки детектора на звітах про град (results/hail_validation/per_event.jsonl).

Лише зведені показники за групами подій (кількість, середні, бутстреп-ДІ за подіями): дат, місць і
координат окремих звітів ESWD у виході немає (угода ESSL). Групи за координатами - грубі прямокутники:
Крим (φ < 46,3°, λ > 32,4°), Карпати й Закарпаття (λ < 25,6°, φ < 49,0°), регіон навчання моделей
(Полтавщина, 48,7-50,9° пн. ш., 32,0-35,5° сх. д.).

    python scripts/hail_subgroups.py --run results/hail_validation --out results/hail_validation/subgroups.md
"""
import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = ("0", "1", "2", "3", "4")
DETECTORS = ("ndvi", "ndre", "ms")


def mean_over_seeds(r, zone, getter):
    v = [getter(r["seeds"][s][zone]) for s in SEEDS if s in r["seeds"]]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else np.nan


def boot_ci(x, rng, n=2000):
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def row(name, recs, rng):
    if len(recs) < 5:
        return f"| {name} | {len(recs)} | " + " | ".join(["-"] * 5) + " |"
    cells = []
    for d in DETECTORS:
        ev = np.array([mean_over_seeds(r, "event", lambda z: z["frac_flagged"][d]) for r in recs])
        ct = np.array([mean_over_seeds(r, "control", lambda z: z["frac_flagged"][d]) for r in recs])
        ok = ~(np.isnan(ev) | np.isnan(ct))
        diff = ev[ok] - ct[ok]
        lo, hi = boot_ci(diff, rng)
        cells.append(f"{100 * diff.mean():+.1f} [{100 * lo:+.1f}; {100 * hi:+.1f}]")
    ev = np.array([mean_over_seeds(r, "event", lambda z: z["obs_minus_pred_ndvi"]) for r in recs])
    ct = np.array([mean_over_seeds(r, "control", lambda z: z["obs_minus_pred_ndvi"]) for r in recs])
    ok = ~(np.isnan(ev) | np.isnan(ct))
    diff = ev[ok] - ct[ok]
    lo, hi = boot_ci(diff, rng)
    cells.append(f"{diff.mean():+.4f} [{lo:+.4f}; {hi:+.4f}]")
    veg = np.array([r["seeds"]["0"]["event"]["n_veg_px"] / max(r["seeds"]["0"]["control"]["n_veg_px"], 1)
                    * r.get("n_control", 8) / 9 for r in recs])
    cells.append(f"{np.median(veg):.2f}")
    return f"| {name} | {len(recs)} | " + " | ".join(cells) + " |"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="results/hail_validation")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    recs = [json.loads(l) for l in (Path(args.run) / "per_event.jsonl").read_text().splitlines() if l.strip()]
    ok = [r for r in recs if "seeds" in r and r.get("ref_before_event")]
    rng = np.random.default_rng(0)
    crimea = lambda r: r["lat"] < 46.3 and r["lon"] > 32.4
    carp = lambda r: r["lon"] < 25.6 and r["lat"] < 49.0
    poltava = lambda r: 48.7 <= r["lat"] <= 50.9 and 32.0 <= r["lon"] <= 35.5
    month = lambda r: int(r["date"][5:7])
    groups = [
        ("Усі придатні події", ok),
        ("Ціль ≤ 7 діб після граду", [r for r in ok if r["days_after"] <= 7]),
        ("Ціль > 7 діб після граду", [r for r in ok if r["days_after"] > 7]),
        ("Травень-липень", [r for r in ok if 5 <= month(r) <= 7]),
        ("Без Криму й Карпат", [r for r in ok if not (crimea(r) or carp(r))]),
        ("Без Криму й Карпат, ціль ≤ 7 діб, травень-липень",
         [r for r in ok if not (crimea(r) or carp(r)) and r["days_after"] <= 7 and 5 <= month(r) <= 7]),
        ("Регіон навчання (Полтавщина)", [r for r in ok if poltava(r)]),
        ("Усі 8 контрольних фрагментів у сцені", [r for r in ok if r.get("n_control", 8) == 8]),
    ]
    lines = ["# Перевірка детектора на звітах про град: підгрупи", "",
             f"Подій у прогоні: {len(recs)}, придатних: {len(ok)}. Різниця - зона граду мінус контроль, "
             "середнє за п'ятьма запусками моделі; 95 % ДІ - бутстреп за подіями (2000 повторів).", "",
             "| Група | Подій | Тривоги NDVI, в.п. | Тривоги NDRE, в.п. | Тривоги MS, в.п. | ΔNDVI спостер.−прогноз | "
             "Вегетуючих пікселів: град / контроль (медіана, на фрагмент) |",
             "|---|---|---|---|---|---|---|"]
    lines += [row(n, g, rng) for n, g in groups]
    lines += ["", "Джерело: European Severe Weather Database, ESSL (eswd.eu); Dotzek et al. (2009)."]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
