#!/usr/bin/env python3
"""
Експеримент E (монографія, розд. 4.2): витік цільового кадру у вхід на ВСІХ ділянках, а не на 30 семплах.

Для кожного фрагмента й двох правил вибору цілі:
    argmin       - найменш хмарний кадр ряду (офіційний завантажувач SEN12MS-CR-TS);
    random_clean - випадковий кадр із середньою хмарністю <= 0,3 (протокол монографії);
рахуються три базові методи (як у eval_protocol.py) і ознака ризику витоку:
    least_cloudy_raw    - найменш хмарний кадр ряду, ціль НЕ виключено;
    least_cloudy_strict - те саме з виключеною ціллю й кадрами того самого дня;
    linear_interp       - потрапіксельна інтерполяція між чистими пікселями до і після цілі.
risk_of_leak: найменш хмарний кадр = ціль, або його хмарність <= EPS_CLOUD і він того самого дня.
Метрики - лише за чистими пікселями цілі (CLEAN_PIXEL_THRESH).

    python scripts/leakage_full.py --root data/raw --out results/leakage_full
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sen12mscrts_dataset import SEN12MSCRTSPatchDataset, linear_interp_ref, pick_target_idx
from train_step2 import resolve_roi_spec, count_patches
from eval_protocol import all_metrics, bootstrap_diff_ci, CLEAN_PIXEL_THRESH, EPS_CLOUD, TOL_SAME_DAYS

ROIS = ["s2_america_test/ROIs1158/106", "s2_america_test/ROIs1970/132",
        "s2_africa_test/ROIs2017/32", "s2_africa_test/ROIs2017/140",
        "s2_asiaEast_test/ROIs1868/73", "s2_asiaEast_test/ROIs1868/119", "s2_asiaEast_test/ROIs1970/139"]
METHODS = ("least_cloudy_raw", "least_cloudy_strict", "linear_interp")


def evaluate(frames, clouds, t_days, tgt):
    frac = clouds.reshape(len(clouds), -1).mean(1)
    raw = int(np.argmin(frac))
    same_day = np.abs(t_days - t_days[tgt]) <= TOL_SAME_DAYS
    risk = bool(raw == tgt or (frac[raw] <= EPS_CLOUD and same_day[raw]))
    keep = ~same_day
    keep[tgt] = False
    cand = np.where(keep)[0]
    strict = int(cand[np.argmin(frac[cand])])
    target, mask = frames[tgt], clouds[tgt] < CLEAN_PIXEL_THRESH
    preds = {"least_cloudy_raw": frames[raw], "least_cloudy_strict": frames[strict],
             "linear_interp": linear_interp_ref(frames, clouds, t_days, tgt)}
    return risk, {m: all_metrics(p, target, mask=mask) for m, p in preds.items()}


def summarize(rows):
    def stats(sub):
        return {m: {k: float(np.nanmean([r["metrics"][m][k] for r in sub])) for k in ("rmse", "sam_deg")}
                for m in METHODS} if sub else None
    free = [r for r in rows if not r["risk"]]
    out = {"n": len(rows), "n_risk": len(rows) - len(free), "share_risk": 1 - len(free) / max(len(rows), 1),
           "full": stats(rows), "leak_free": stats(free)}
    if free:
        a = [r["metrics"]["linear_interp"]["rmse"] for r in free]
        b = [r["metrics"]["least_cloudy_strict"]["rmse"] for r in free]
        out["leak_free_rmse_interp-strict"] = bootstrap_diff_ci(np.array(a), np.array(b))
    a = [r["metrics"]["linear_interp"]["rmse"] for r in rows]
    b = [r["metrics"]["least_cloudy_raw"]["rmse"] for r in rows]
    out["full_rmse_interp-raw"] = bootstrap_diff_ci(np.array(a), np.array(b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--roi", action="append", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-patches", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = {"argmin": [], "random_clean": []}
    for spec in args.roi or ROIS:
        r, roi_dir, roi = resolve_roi_spec(args.root, spec)
        n = count_patches(r, roi_dir, roi)
        if args.max_patches:
            n = min(n, args.max_patches)
        ds = SEN12MSCRTSPatchDataset(root=r, roi_dir=roi_dir, roi=roi, patch_indices=range(n),
                                     target_mode="random_clean", target_max_cloud=0.3, seed=args.seed)
        for i in range(len(ds)):
            frames, clouds = ds.full_series(i)
            t_days = ds.pick(i)["t_days"]
            frac = clouds.reshape(len(clouds), -1).mean(1)
            for mode in rows:
                rng = np.random.default_rng(args.seed + ds.pdx(i))
                tgt = pick_target_idx(frac, rng, 0.3, mode)
                risk, metrics = evaluate(frames, clouds, t_days, tgt)
                rows[mode].append({"roi": spec, "patch": int(ds.pdx(i)), "risk": risk, "metrics": metrics})
        print(f"{spec}: {n} фрагментів", flush=True)

    summary = {mode: {"all": summarize(rr),
                      "per_roi": {spec: summarize([x for x in rr if x["roi"] == spec])
                                  for spec in dict.fromkeys(x["roi"] for x in rr)}}
               for mode, rr in rows.items()}
    (out / "per_patch.json").write_text(json.dumps(rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for mode in rows:
        s = summary[mode]["all"]
        print(f"\n{mode}: ризик витоку {s['n_risk']}/{s['n']} ({100 * s['share_risk']:.1f} %)")
        for part in ("full", "leak_free"):
            if s[part]:
                print(f"  {part}: " + ", ".join(f"{m} RMSE {s[part][m]['rmse']:.4f}" for m in METHODS))


if __name__ == "__main__":
    main()
