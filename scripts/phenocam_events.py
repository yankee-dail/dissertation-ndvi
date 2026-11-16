#!/usr/bin/env python3
"""
Розділ 10.2: конформний детектор різких змін за щоденним рядом наземної камери (GCC - лише RGB, або NDVI)
і порівняння часу виявлення із супутником.

Для кожної ділянки-року (квітень-жовтень) прогноз на день t - медіана ряду за попередні 7 діб (щонайменше
три значення); залишок e_t = (прогноз − спостереження) / A, де A - сезонна амплітуда ряду ділянки
(95-й мінус 5-й процентиль за всі роки), тож поріг можна переносити між ділянками. Поріг - конформний
(1 − α)-квантиль залишків на калібрувальних ділянках; тест - на інших (поділ навпіл за назвою, двічі).
Справжніх міток подій у PhenoCam немає, тому чутливість перевірено на штучному падінні: від випадкового дня
t0 зеленої фази (ряд вище сезонної медіани) ряд знижується на s·A і лишається нижчим. Подія виявлена, якщо
тривога є хоча б в один із днів t0..t0+2. Частка тривог на незмінених рядах включає справжні події
(збирання, скошування, вилягання), тож це верхня межа частки хибних тривог.
Для тих самих t0 рахується, через скільки діб буде перший чистий знімок Sentinel-2 (частка чистих
пікселів ≥ 0,9) - найраніший момент, коли зміну міг би побачити супутник.

    python scripts/phenocam_events.py --data data/phenocam --agreement results/phenocam/summary.json \
        --out results/phenocam_events
"""
import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np

SEVERITIES = (0.05, 0.1, 0.2, 0.3)


def d2i(s):
    return date.fromisoformat(s).toordinal()


def split(name):
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 2


def load(path, index, min_clear):
    d = json.loads(Path(path).read_text())
    cam = {d2i(k): v[index] for k, v in d["camera"].items() if v.get(index) is not None}
    clear = sorted({d2i(r["date"]) for r in d["s2"] if r["clear_frac"] >= min_clear and "ndvi_clear" in r
                    and -0.2 < r["ndvi_clear"] <= 1})
    vals = np.array(list(cam.values()))
    amp = float(np.percentile(vals, 95) - np.percentile(vals, 5)) if len(vals) > 50 else None
    return {"site": d["site"], "years": d["years"], "cam": cam, "clear": clear, "amp": amp}


def residuals(series, amp):
    """series: dict день -> значення (може бути змінений). Повертає dict день -> залишок."""
    out = {}
    for t in sorted(series):
        prev = [series[t - k] for k in range(1, 8) if t - k in series]
        if len(prev) >= 3:
            out[t] = (np.median(prev) - series[t]) / amp
    return out


def conf_q(res, alpha):
    res = np.sort(np.asarray(res))
    k = min(int(np.ceil((len(res) + 1) * (1 - alpha))), len(res))
    return float(res[k - 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/phenocam")
    ap.add_argument("--agreement", default="results/phenocam/summary.json",
                    help="ділянки з r ≥ --min-r (розділ 9.3); інші - камера бачить інше поле")
    ap.add_argument("--min-r", type=float, default=0.5)
    ap.add_argument("--min-clear", type=float, default=0.9)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.01])
    ap.add_argument("--per-year", type=int, default=20, help="штучних подій на ділянку-рік")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    agr = json.loads(Path(args.agreement).read_text())["agreement"]
    good = sorted(s for s, a in agr.items() if a and a["r"] >= args.min_r)
    rng = np.random.default_rng(0)
    L = ["# Розділ 10.2: конформний детектор різких змін за щоденним рядом камери", "",
         f"Ділянок: {len(good)} (r ≥ {args.min_r} у розділі 9.3). Прогноз - медіана за 7 попередніх діб; "
         "залишок у частках сезонної амплітуди; поріг калібрується на іншій половині ділянок (два розбиття). "
         "Виявлення - тривога в день t0, t0+1 або t0+2.", ""]
    summary = {}
    for index in ("gcc", "ndvi"):
        sites = [load(Path(args.data) / f"{s}.json", index, args.min_clear) for s in good]
        sites = [s for s in sites if s["amp"]]
        base = {s["site"]: residuals(s["cam"], s["amp"]) for s in sites}
        # штучні події (однакові t0 для всіх alpha)
        events = []
        for s in sites:
            med = np.median(list(s["cam"].values()))
            for y in s["years"]:
                days = [t for t in s["cam"] if date.fromordinal(t).year == y and s["cam"][t] > med
                        and date.fromordinal(t).month <= 9]
                if len(days) < 30:
                    continue
                for t0 in rng.choice(days, min(args.per_year, len(days)), replace=False):
                    nxt = [c for c in s["clear"] if c >= t0]
                    events.append({"site": s["site"], "t0": int(t0),
                                   "s2_delay": (nxt[0] - int(t0)) if nxt and nxt[0] - t0 <= 60 else None})
        res_by_site = {s["site"]: s for s in sites}
        for alpha in args.alphas:
            fpr, det = [], {sv: [] for sv in SEVERITIES}
            det1 = {sv: [] for sv in SEVERITIES}
            for fold in (0, 1):
                cal = [v for s, r in base.items() if split(s) != fold for v in r.values()]
                q = conf_q(cal, alpha)
                tst = [s for s in base if split(s) == fold]
                fpr += [v > q for s in tst for v in base[s].values()]
                for e in (e for e in events if split(e["site"]) == fold):
                    s = res_by_site[e["site"]]
                    for sv in SEVERITIES:
                        mod = {t: (v - sv * s["amp"] if t >= e["t0"] else v) for t, v in s["cam"].items()
                               if e["t0"] - 8 <= t <= e["t0"] + 2}
                        r = residuals(mod, s["amp"])
                        hits = [r[t] > q for t in range(e["t0"], e["t0"] + 3) if t in r]
                        det[sv].append(bool(hits) and any(hits))
                        det1[sv].append(e["t0"] in r and r[e["t0"]] > q)
            key = f"{index}/alpha={alpha}"
            summary[key] = {"alarm_rate": float(np.mean(fpr)), "n_days": len(fpr),
                            "detected_3d": {str(k): float(np.mean(v)) for k, v in det.items()},
                            "detected_day0": {str(k): float(np.mean(v)) for k, v in det1.items()},
                            "n_events": len(det[SEVERITIES[0]])}
        delays = [e["s2_delay"] for e in events if e["s2_delay"] is not None]
        summary[f"{index}/s2_delay"] = {"median": float(np.median(delays)), "p75": float(np.percentile(delays, 75)),
                                        "p90": float(np.percentile(delays, 90)), "n": len(delays),
                                        "frac_within_3d": float(np.mean(np.array(delays) <= 2)),
                                        "n_no_clear_60d": sum(e["s2_delay"] is None for e in events)}
    L += ["| Ряд | α | Частка тривог на незмінених рядах | Подій | " +
          " | ".join(f"Виявлено за 3 доби, падіння {int(s * 100)} %" for s in SEVERITIES) + " |",
          "|---|---|---|---|" + "---|" * len(SEVERITIES)]
    for key, v in summary.items():
        if "alpha" not in key:
            continue
        index, a = key.split("/")
        L.append(f"| {index.upper()} | {a.split('=')[1]} | {v['alarm_rate']:.3f} | {v['n_events']} | " +
                 " | ".join(f"{v['detected_3d'][str(s)]:.3f}" for s in SEVERITIES) + " |")
    v = summary["gcc/s2_delay"]
    L += ["", f"Перший чистий знімок Sentinel-2 після t0: медіана через {v['median']:.0f} діб, 75-й процентиль "
          f"{v['p75']:.0f}, 90-й {v['p90']:.0f}; у межах t0..t0+2 - для {100 * v['frac_within_3d']:.0f} % подій "
          f"({v['n']} подій; ще {v['n_no_clear_60d']} без чистого знімка 60 діб)."]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print("\n".join(L))


if __name__ == "__main__":
    main()
