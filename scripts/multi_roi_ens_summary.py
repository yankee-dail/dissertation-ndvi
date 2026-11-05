#!/usr/bin/env python3
"""Зведення експерименту D2 (results/multi_roi_ens) у markdown-таблиці: ансамбль vs окрема модель vs інтерполяція
і покриття при q з asiaEast/139 проти q з інших ділянок europa.

    python scripts/multi_roi_ens_summary.py --root results/multi_roi_ens > results/multi_roi_ens/summary.md
"""
import argparse
import json
from pathlib import Path

import numpy as np

MS = ("ensemble", "single", "linear_interp")


def sig(ci):
    """-1 - значуще менше, +1 - значуще більше, 0 - ні."""
    return -1 if ci[2] < 0 else (1 if ci[1] > 0 else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/multi_roi_ens")
    root = Path(ap.parse_args().root)
    rois = sorted(p for p in root.iterdir() if p.is_dir())
    allm = []
    print("| Ділянка | RMSE анс. | RMSE окр. | RMSE інтерп. | анс.<інтерп. | анс.<окр. | NDVI анс. | NDVI окр. | NDVI інтерп. "
          "| NDVI анс. vs інтерп. (−/+) | SAM анс.−інтерп., ° |")
    print("|" + "---|" * 11)
    for r in rois + [None]:
        ms = allm if r is None else [json.load(open(f)) for f in sorted(r.glob("seed*.json"))]
        if r is not None:
            allm += ms
        g = lambda k, m: np.array([x["mean"][k][m] for x in ms])
        p = lambda key: [sig(x["paired"][key]) for x in ms]
        f = lambda k, m: f"{g(k, m).mean():.4f}±{g(k, m).std(ddof=1):.4f}"
        nd = p("ndvi_rmse_ensemble-linear_interp")
        name = "**Усі**" if r is None else r.name.replace("s2_", "").replace("_test_ROIs", "/").replace("_", "/")
        print(f"| {name} | {f('ensemble','rmse')} | {f('single','rmse')} | {f('linear_interp','rmse')} "
              f"| {p('rmse_ensemble-linear_interp').count(-1)}/{len(ms)} | {p('rmse_ensemble-single').count(-1)}/{len(ms)} "
              f"| {f('ensemble','ndvi_rmse')} | {f('single','ndvi_rmse')} | {f('linear_interp','ndvi_rmse')} "
              f"| {nd.count(-1)}/{nd.count(1)} | {(g('ensemble','sam_deg') - g('linear_interp','sam_deg')).mean():+.2f} |")
    cal = json.load(open(root / "calibration.json"))
    print("\n| Ділянка | покриття q(asiaEast) | покриття q(europa), окрема | покриття q(europa), ансамбль "
          "| ширина q(asiaEast) | ширина q(europa), окрема | ширина q(europa), анс. |")
    print("|" + "---|" * 7)
    names = sorted({c["roi"] for c in cal}) + [None]
    for n in names:
        cs = [c for c in cal if n is None or c["roi"] == n]
        v = lambda var, k: np.array([c[var][k] for c in cs])
        pc = lambda a: f"{100 * a.mean():.1f}±{100 * a.std(ddof=1):.1f}"
        w = lambda a: f"{a.mean():.3f}"
        print(f"| {'**Усі**' if n is None else n} | {pc(v('single','coverage_asiaEast'))} | {pc(v('single','coverage_region'))} "
              f"| {pc(v('ensemble','coverage_region'))} | {w(v('single','width_asiaEast'))} "
              f"| {w(v('single','width_region'))} | {w(v('ensemble','width_region'))} |")


if __name__ == "__main__":
    main()
