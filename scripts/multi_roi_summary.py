#!/usr/bin/env python3
"""Зведення експерименту D (розд. 5.8 і 6.3 монографії): results/multi_roi/*/seed*/summary.json ->
середнє ± СКВ за запусками для кожної ділянки й разом; кількість запусків зі значущою різницею.

    python scripts/multi_roi_summary.py > results/multi_roi/summary.md
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("results/multi_roi")


def ms(x, d=4):
    x = np.asarray(x, float)
    return f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}".replace(".", ",")


def signif(ci):  # (оцінка, нижня, верхня): значуще краще, якщо верхня межа < 0
    return ci[2] < 0


rows, allr = [], []
for d in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    ss = [json.load(open(f)) for f in sorted(d.glob("seed*/summary.json"))]
    if not ss:
        continue
    allr += ss
    rm, rl = [s["refl"]["model"]["rmse"] for s in ss], [s["refl"]["linear_interp"]["rmse"] for s in ss]
    nm, nl = [s["ndvi_rmse"]["model"] for s in ss], [s["ndvi_rmse"]["linear_interp"] for s in ss]
    gain = 100 * (1 - np.mean(rm) / np.mean(rl))
    ngain = 100 * (1 - np.mean(nm) / np.mean(nl))
    rows.append([ss[0]["roi"], ss[0]["n_test"], len(ss), ms(rm), ms(rl), f"{gain:.1f}".replace(".", ","),
                 sum(signif(s["paired"]["rmse_model-linear_interp"]) for s in ss),
                 f"{ngain:.1f}".replace(".", ","), sum(signif(s["paired"]["ndvi_model-linear_interp"]) for s in ss),
                 sum(signif(s["paired"]["sam_model-linear_interp"]) for s in ss),
                 ms([100 * s["coverage"] for s in ss], 1), ms([s["mean_width"] for s in ss], 3)])

hdr = ["ROI", "фрагментів", "запусків", "RMSE модель", "RMSE інтерп.", "виграш RMSE, %", "значущо RMSE",
       "виграш RMSE NDVI, %", "значущо NDVI", "значущо SAM", "покриття, %", "ширина"]
print("| " + " | ".join(hdr) + " |")
print("|" + "---|" * len(hdr))
for r in rows:
    print("| " + " | ".join(map(str, r)) + " |")

g = [100 * (1 - s["refl"]["model"]["rmse"] / s["refl"]["linear_interp"]["rmse"]) for s in allr]
cov = [100 * s["coverage"] for s in allr]
print(f"\nУсього пар (ділянка, запуск): {len(allr)}; виграш RMSE за парою: від {min(g):.1f} до {max(g):.1f} %, "
      f"значущий у {sum(signif(s['paired']['rmse_model-linear_interp']) for s in allr)}; "
      f"покриття від {min(cov):.1f} до {max(cov):.1f} %, середнє {np.mean(cov):.1f} %")
