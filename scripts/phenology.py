#!/usr/bin/env python3
"""
Розділ про фенологію поля: дати початку вегетації (SOS) і завершення (EOS) за рядом Sentinel-2 проти щоденного
ряду наземної камери PhenoCam, з конформним інтервалом для супутникової дати.

Для кожної ділянки-року (березень-листопад):
  камера   - щоденний ряд (GCC або NDVI камери), медіана за 7 діб; основний пік - максимум сезону;
             SOS - останній перетин рівня 50 % сезонної амплітуди знизу вгору до піку,
             EOS - перший перетин униз після піку (рівень = min + 0,5·(max − min) з урахуванням рівнів
             до і після піку окремо);
  супутник - чисті дати Sentinel-2 (частка чистих пікселів ≥ 0,9), лінійна інтерполяція між ними,
             та сама процедура. «Вікно» дати - відстань між двома чистими датами, між якими лежить перетин.
Похибка e = дата супутника − дата камери. Систематичний зсув (різні індекси, різне поле зору) оцінюється на
калібрувальних ділянках і віднімається; конформна півширина - (1 − α)-квантиль |e − зсув| на калібрувальних
ділянках; перевірка - на інших (поділ ділянок навпіл за MD5 назви, два розбиття). Mondrian-варіант - окремо
для вікна ≤ 10 і > 10 діб.

    python scripts/phenology.py --data data/phenocam --agreement results/phenocam/summary.json \
        --out results/phenology
"""
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from phenocam_analysis import conf_q, load, split


def smooth(days, vals, w=7):
    d = np.asarray(days)
    v = np.asarray(vals)
    return np.array([np.median(v[np.abs(d - t) <= w // 2]) for t in d])


def transitions(t, v):
    """t - зростаючі дні (ординали), v - значення. -> (sos, eos, t_peak) або None (лінійна інтерполяція перетину)."""
    if len(t) < 4:
        return None
    ip = int(np.argmax(v))
    if ip == 0 or ip == len(v) - 1:
        return None
    lo_l, lo_r, hi = v[:ip + 1].min(), v[ip:].min(), v[ip]
    if hi - max(lo_l, lo_r) < 0.05 * max(abs(hi), 1e-3) and hi - max(lo_l, lo_r) < 0.02:
        return None
    lv_l, lv_r = lo_l + 0.5 * (hi - lo_l), lo_r + 0.5 * (hi - lo_r)
    sos = eos = None
    for i in range(ip, 0, -1):
        if v[i - 1] < lv_l <= v[i]:
            sos = (t[i - 1] + (lv_l - v[i - 1]) / (v[i] - v[i - 1]) * (t[i] - t[i - 1]), t[i] - t[i - 1])
            break
    for i in range(ip, len(v) - 1):
        if v[i] >= lv_r > v[i + 1]:
            eos = (t[i] + (v[i] - lv_r) / (v[i] - v[i + 1]) * (t[i + 1] - t[i]), t[i + 1] - t[i])
            break
    return sos, eos, t[ip]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/phenocam")
    ap.add_argument("--agreement", default="results/phenocam/summary.json")
    ap.add_argument("--min-r", type=float, default=0.5)
    ap.add_argument("--min-clear", type=float, default=0.9)
    ap.add_argument("--min-cam-days", type=int, default=150)
    ap.add_argument("--min-s2", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    agr = json.loads(Path(args.agreement).read_text())["agreement"]
    good = {s for s, a in agr.items() if a and a["r"] >= args.min_r}

    summary, L = {}, ["# Фенологія поля: дати SOS і EOS за Sentinel-2 проти наземної камери", ""]
    for index in ("gcc", "ndvi"):
        recs = []
        for s in load(args.data, args.min_clear, index):
            if s["site"] not in good:
                continue
            for y in s["years"]:
                lo, hi = date(y, 3, 1).toordinal(), date(y, 11, 30).toordinal()
                cd = sorted(t for t in s["cam"] if lo <= t <= hi)
                sd = sorted(t for t in s["s2"] if lo <= t <= hi)
                if len(cd) < args.min_cam_days or len(sd) < args.min_s2:
                    continue
                cam = transitions(np.array(cd, float), smooth(cd, [s["cam"][t] for t in cd]))
                sat = transitions(np.array(sd, float), np.array([s["s2"][t] for t in sd]))
                if cam is None or sat is None:
                    continue
                for k, kk in ((0, "sos"), (1, "eos")):
                    if cam[k] is None or sat[k] is None:
                        continue
                    recs.append({"site": s["site"], "year": y, "kind": kk, "cam": cam[k][0], "sat": sat[k][0],
                                 "window": sat[k][1], "e": sat[k][0] - cam[k][0],
                                 "matched": abs(sat[2] - cam[2]) <= 30})
        res = {}
        for kk in ("sos", "eos", "sos/matched", "eos/matched"):
            R = [r for r in recs if r["kind"] == kk.split("/")[0] and ("matched" not in kk or r["matched"])]
            if len(R) < 10:
                continue
            e = np.array([r["e"] for r in R])
            win = np.array([r["window"] for r in R])
            hit = {"shared": [], "mondrian": []}
            hw = {"shared": [], "mondrian": []}
            bias_f = []
            for fold in (0, 1):
                cal = [r for r in R if split(r["site"]) != fold]
                tst = [r for r in R if split(r["site"]) == fold]
                if len(cal) < 5 or not tst:
                    continue
                b = float(np.median([r["e"] for r in cal]))
                bias_f.append(b)
                q = conf_q([abs(r["e"] - b) for r in cal], args.alpha)
                qm = {w: conf_q([abs(r["e"] - b) for r in cal if (r["window"] <= 10) == w], args.alpha)
                      for w in (True, False)}
                for r in tst:
                    hit["shared"].append(abs(r["e"] - b) <= q)
                    hw["shared"].append(q)
                    qq = qm[r["window"] <= 10]
                    if np.isnan(qq):
                        qq = q
                    hit["mondrian"].append(abs(r["e"] - b) <= qq)
                    hw["mondrian"].append(qq)
            short = win <= 10
            res[kk] = {"n": len(R), "matched_share": float(np.mean([r["matched"] for r in recs if r["kind"] == kk.split("/")[0]])), "sites": len({r["site"] for r in R}),
                       "bias_median": float(np.median(e)), "mae": float(np.mean(np.abs(e))),
                       "mae_debiased": float(np.mean(np.abs(e - np.median(e)))),
                       "abs_err_median_window_le10": float(np.median(np.abs(e[short] - np.median(e)))) if short.any() else None,
                       "abs_err_median_window_gt10": float(np.median(np.abs(e[~short] - np.median(e)))) if (~short).any() else None,
                       "n_window_le10": int(short.sum()), "window_median": float(np.median(win)),
                       "coverage": {k: float(np.mean(v)) for k, v in hit.items() if v},
                       "halfwidth_mean": {k: float(np.mean(v)) for k, v in hw.items() if v},
                       "n_test": len(hit["shared"])}
        summary[index] = res
        L += [f"## Камера: {index.upper()}", "",
              "Пік збігається (|пік супутника − пік камери| ≤ 30 діб): " + ", ".join(
                  f"{k}: {v['matched_share']:.2f}" for k, v in res.items() if "/" not in k), "",
              "| Дата | Ділянко-років (ділянок) | Зсув, діб | MAE, діб | MAE без зсуву | |похибка|, вікно ≤ 10 / > 10 діб | "
              "Покриття 90 %: спільне / Mondrian | Півширина, діб: спільна / Mondrian |",
              "|---|---|---|---|---|---|---|---|"]
        for kk, v in res.items():
            L.append(f"| {kk.upper()} | {v['n']} ({v['sites']}) | {v['bias_median']:+.1f} | {v['mae']:.1f} | "
                     f"{v['mae_debiased']:.1f} | {v['abs_err_median_window_le10']} / {v['abs_err_median_window_gt10']} | "
                     f"{v['coverage'].get('shared', float('nan')):.2f} / {v['coverage'].get('mondrian', float('nan')):.2f} | "
                     f"{v['halfwidth_mean'].get('shared', float('nan')):.1f} / {v['halfwidth_mean'].get('mondrian', float('nan')):.1f} |")
        L.append("")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
