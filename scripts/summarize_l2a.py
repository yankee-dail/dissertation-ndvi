#!/usr/bin/env python3
"""Зведений звіт: власні L2A-регіони (кілька seed) проти SEN12MS-CR-TS на тій самій ділянці.

    python scripts/summarize_l2a.py --region pampa2018 --region poltava2021 --out results/l2a_report.md
"""
import argparse
import json
from pathlib import Path

import numpy as np

SEN12MS = {"validation": "results/validation_extras/causal_cpu_seed0",
           "anomaly": "results/anomaly_detection/causal_cpu_seed0",
           "anomaly_mondrian": "results/anomaly_detection/causal_mondrian_cpu_seed0"}
DET = ("ndvi", "ndre", "ms")


def load(path):
    p = Path(path) / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def ms(vals, f="{:.4f}"):
    v = np.array([x for x in vals if x is not None], dtype=float)
    if len(v) == 0:
        return "-"
    return f.format(v.mean()) if len(v) == 1 else f"{f.format(v.mean())} ± {f.format(v.std(ddof=1))}"


def sig(cis):
    better = sum(c[2] < 0 for c in cis)
    worse = sum(c[1] > 0 for c in cis)
    return f"краще {better}/{len(cis)}, гірше {worse}/{len(cis)}"


def gap_median(vpath):
    p = Path(vpath) / "per_patch.json"
    if not p.exists():
        return None
    return float(np.median([r["gap_days"] for r in json.loads(p.read_text())]))


def section(name, vals, anoms, anoms_m, gaps):
    L = [f"## {name} ({len(vals)} запуск(и))", ""]
    if not vals:
        return L + ["немає результатів", ""]
    L += [f"Медіанний розрив «останнє чисте спостереження → ціль»: {ms(gaps, '{:.0f}')} діб.", "",
          "### Реконструкція (причинний режим), тестова зона", "",
          "| Метод | RMSE | SAM, ° | RMSE NDVI |", "|---|---|---|---|"]
    for m, label in (("model", "Залишкова U-Net (причинна)"), ("yref", "Копіювання y_ref"),
                     ("linear_interp", "linear_interp (не причинна, довідково)"),
                     ("lc_strict", "least_cloudy_strict")):
        L.append(f"| {label} | {ms([v['refl'][m]['rmse'][0] for v in vals])} | "
                 f"{ms([v['refl'][m]['sam_deg'][0] for v in vals], '{:.2f}')} | "
                 f"{ms([v['ndvi'][m]['rmse'][0] for v in vals], '{:.3f}')} |")
    L += ["", "| Парне порівняння | Середнє | Значущість |", "|---|---|---|"]
    for k, label in (("model-yref_rmse", "ΔRMSE модель − y_ref"), ("model-yref_sam_deg", "ΔSAM модель − y_ref"),
                     ("ndvi_model-yref_rmse", "ΔRMSE NDVI модель − y_ref")):
        cis = [v["paired"][k] for v in vals]
        L.append(f"| {label} | {ms([c[0] for c in cis])} | {sig(cis)} |")
    L += ["", "### Конформне калібрування (пара μ L1-моделі + σ σ-моделі; калібрувальна ділянка - див. заголовок розділу)", "",
          f"q = {ms([v['q_l1_pair'] for v in vals], '{:.3f}')}; фактичне покриття "
          f"{ms([100 * v['coverage_l1_pair'] for v in vals], '{:.1f}')} % (номінал 90 %); середня ширина "
          f"{ms([v['mean_width_l1_pair'] for v in vals])}.", ""]
    for label, an in (("спільний поріг", anoms), ("Mondrian за розривом", anoms_m)):
        if not an:
            continue
        L += [f"### Виявлення штучного стресу, α = 0,1, {label}", "",
              "| | " + " | ".join(DET) + " |", "|---|---|---|---|",
              "| Хибні тривоги | " + " | ".join(ms([a["fpr"]["0.1"][d][0] for a in an], "{:.3f}") for d in DET) + " |"]
        for s in ("0.05", "0.1", "0.2", "0.3"):
            L.append(f"| Виявлено, стрес {s} | " + " | ".join(
                ms([a["tpr"][f"{s}|0.1"][d][0] for a in an], "{:.3f}") for d in DET) + " |")
        L.append("")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", action="append", default=[])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/l2a_report.md")
    args = ap.parse_args()

    L = ["# Власні дані Sentinel-2 L2A проти SEN12MS-CR-TS: причинна реконструкція, калібрування, виявлення стресу", ""]
    v, a, am = load(SEN12MS["validation"]), load(SEN12MS["anomaly"]), load(SEN12MS["anomaly_mondrian"])
    L += section("SEN12MS-CR-TS, america/132 (Пампа), калібрування на asiaEast/139 (Китай)",
                 [v] if v else [], [a] if a else [], [am] if am else [], [gap_median(SEN12MS["validation"])])
    for region in args.region:
        base = Path(f"results/l2a_{region}")
        vals = [load(base / f"validation_seed{s}") for s in args.seeds]
        an = [load(base / f"anomaly_seed{s}") for s in args.seeds]
        anm = [load(base / f"anomaly_mondrian_seed{s}") for s in args.seeds]
        gaps = [gap_median(base / f"validation_seed{s}") for s in args.seeds]
        L += section(f"L2A {region}, калібрування в тому ж регіоні", [x for x in vals if x],
                     [x for x in an if x], [x for x in anm if x], gaps)
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
