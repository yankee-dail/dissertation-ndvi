#!/usr/bin/env python3
"""Парна різниця NDRE − NDVI у виявленні штучного стресу (L2A, бутстреп за фрагментами, запуски разом), розд. 7.5 монографії.

    python scripts/l2a_ndre_paired.py > results/l2a_ndre_paired.txt
"""
import json

import numpy as np


def run(region, seeds):
    rows = []
    for s in seeds:
        rows += json.load(open(f"results/l2a_{region}/anomaly_seed{s}/per_patch.json"))
    rng = np.random.default_rng(0)
    n = len(rows)
    out = {}
    for key, kind in [("0.1|0.1", "tpr"), ("0.3|0.1", "tpr"), ("0.1", "fpr")]:
        if kind == "tpr":
            # частка виявлених уражених пікселів: зважуємо фрагменти кількістю пікселів осередку
            w = np.array([r["n_spot"] for r in rows], float)
            d = np.array([r["tpr"][key]["ndre"] - r["tpr"][key]["ndvi"] if r["n_spot"] else 0 for r in rows])
        else:
            w = np.ones(n)
            d = np.array([r["fpr"][key]["ndre"] - r["fpr"][key]["ndvi"] for r in rows])
        est = (w * d).sum() / w.sum()
        bs = []
        for _ in range(2000):
            i = rng.integers(0, n, n)
            bs.append((w[i] * d[i]).sum() / w[i].sum())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[kind + key] = f"{100 * est:+.1f} [{100 * lo:+.1f}; {100 * hi:+.1f}]"
    print(region, seeds, out)


for reg in ["pampa2018_lc", "poltava2021_lc"]:
    run(reg, [0, 1, 2])
    run(reg, [0, 1, 2, 3, 4])
