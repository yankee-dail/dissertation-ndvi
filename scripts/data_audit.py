#!/usr/bin/env python3
"""
Перевірка даних Sentinel-2 L2A на приховані систематичні помилки (як подвійний зсув BOA, розділ 7).

Для кожної зони й сцени на безхмарних пікселях (CLD = 0) рахуються медіани відбиття всіх каналів, медіана NDVI,
частка нульових і насичених значень. Сцена позначається, якщо:
  - медіана будь-якого каналу виходить за межі [0,005; 0,6] (зсув на 0,1 дає майже нульові видимі канали);
  - частка нулів серед безхмарних пікселів > 1 % (обрізання після віднімання зсуву);
  - медіанний NDVI сцени відрізняється від медіани сусідніх сцен (±2 у часі) більше ніж на 0,25;
  - медіана червоного каналу відрізняється від сусідніх сцен більш ніж удвічі.
Сцени з частиною безхмарних пікселів < 5 % не оцінюються.

    python scripts/data_audit.py --root data/l2a --out results/data_audit
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_patch_")
BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/l2a")
    ap.add_argument("--max-patches", type=int, default=12, help="фрагментів на сцену для оцінки")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for region in sorted(p for p in Path(args.root).iterdir() if (p / "zones").is_dir()):
        for zone in sorted((region / "zones").iterdir()):
            scenes = defaultdict(list)
            for f in sorted((zone / "S2").glob("*/*.tif")):
                scenes[(f.parent.name, DATE_RE.search(f.name).group(1))].append(f)
            for (img, d), files in sorted(scenes.items(), key=lambda x: x[0][1]):
                vals, zeros, n = [], 0, 0
                for f in files[:args.max_patches]:
                    with rasterio.open(f) as r:
                        a = r.read().astype(np.float32) / 10000.0
                    with rasterio.open(Path(str(f).replace("/S2/", "/CLD/"))) as r:
                        m = r.read(1) == 0
                    if m.sum() == 0:
                        continue
                    px = a[:, m]
                    vals.append(px)
                    zeros += int((px[[1, 2, 3, 7]] <= 0).any(axis=0).sum())
                    n += px.shape[1]
                total = sum(a.shape[1] * a.shape[2] for a in [np.zeros((1, 128, 128))] * min(len(files), args.max_patches))
                if not vals or n < 0.05 * total:
                    rows.append({"region": region.name, "zone": zone.name, "img": img, "date": d, "clear_frac": n / max(total, 1),
                                 "evaluated": False})
                    continue
                px = np.concatenate(vals, axis=1)
                med = np.median(px, axis=1)
                red, nir = px[3], px[7]
                ndvi = float(np.median((nir - red) / np.maximum(nir + red, 1e-6)))
                rows.append({"region": region.name, "zone": zone.name, "img": img, "date": d, "clear_frac": n / total,
                             "evaluated": True, "median": {b: float(v) for b, v in zip(BANDS, med)},
                             "ndvi_median": ndvi, "zero_frac": zeros / n})
    # позначки
    flags = []
    by = defaultdict(list)
    for r in rows:
        if r["evaluated"]:
            by[(r["region"], r["zone"])].append(r)
    for key, rs in by.items():
        for i, r in enumerate(rs):
            why = []
            bad = [b for b, v in r["median"].items() if b not in ("B10",) and not 0.005 <= v <= 0.6]
            if bad:
                why.append("медіана поза [0,005; 0,6]: " + ", ".join(bad))
            if r["zero_frac"] > 0.01:
                why.append(f"частка нулів {r['zero_frac']:.3f}")
            nb = [x for j, x in enumerate(rs) if j != i and abs(j - i) <= 2]
            if nb:
                dn = r["ndvi_median"] - np.median([x["ndvi_median"] for x in nb])
                if abs(dn) > 0.25:
                    why.append(f"стрибок NDVI {dn:+.2f}")
                rr = r["median"]["B4"] / np.median([x["median"]["B4"] for x in nb])
                if rr > 2 or rr < 0.5:
                    why.append(f"червоний ×{rr:.2f} до сусідніх сцен")
            if why:
                flags.append({"region": key[0], "zone": key[1], "date": r["date"], "img": r["img"], "why": why})
    ev = [r for r in rows if r["evaluated"]]
    summary = {"scenes": len(rows), "evaluated": len(ev), "flagged": len(flags), "flags": flags,
               "red_median_range": [min(r["median"]["B4"] for r in ev), max(r["median"]["B4"] for r in ev)],
               "zero_frac_max": max(r["zero_frac"] for r in ev)}
    (out / "scenes.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    L = ["# Перевірка даних L2A на систематичні помилки", "",
         f"Сцен (зона × дата): {len(rows)}, оцінено {len(ev)} (решта майже повністю хмарні). Позначено: {len(flags)}.",
         f"Медіана червоного каналу на безхмарних пікселях: {summary['red_median_range'][0]:.3f}–{summary['red_median_range'][1]:.3f}; "
         f"найбільша частка нулів: {summary['zero_frac_max']:.4f}.", ""]
    L += [f"- {f['region']}/{f['zone']} {f['date']} (ImgNo {f['img']}): " + "; ".join(f["why"]) for f in flags]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
