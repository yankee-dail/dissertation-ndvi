#!/usr/bin/env python3
"""
Конформне виявлення аномалій стану посіву (причинний режим) на штучно внесеному стресі.

Ідея: причинна модель прогнозує ціль лише з МИНУЛИХ спостережень (μ, σ). Коли надходить
справжній знімок, піксель позначається аномальним, якщо його відхилення від прогнозу
перевищує поріг, відкалібрований КОНФОРМНО на окремій ділянці - тоді частка хибних тривог
на нестресованих пікселях ≈ α за побудовою, а не підбирається на тесті.

Справжніх міток хвороб у SEN12MS-CR-TS немає, тому стрес вноситься штучно (перевірка
ЧУТЛИВОСТІ методу, не виявлення реальних хвороб): у квадратній ділянці вегетуючих пікселів
цілі стилізований спектральний підпис стресу (втрата хлорофілу й води):
    NIR-плато (B6, B7, B8, B8A) × (1 − s);  червоний B4 × (1 + s);
    червоний край B5 і SWIR (B11, B12) × (1 + s/2).
Ураження осередком, меншим за піксель, еквівалентне меншій силі s (лінійне змішування).

Три детектори (усі - одностороння аномалія "стан гірший за прогноз"):
    ndvi : NDVI(μ) − NDVI(y)           (класичний підхід, лише B4/B8)
    ndre : NDRE(μ) − NDRE(y)           (червоний край, B5/B8A)
    ms   : max_k sign_k · (y_k − μ_k) / σ_k  за каналами підпису стресу
           (мультиспектральний, нормований на передбачену невизначеність)

Запуск:
    python scripts/anomaly_detection.py --root data/raw \
        --step2-model results/step2_causal/model.pt \
        --uncertainty-model results/uncertainty_causal/model.pt \
        --img-size 128 --seed 0 --out results/anomaly_detection/causal_cpu_seed0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from sen12mscrts_dataset import n_bands, n_input_channels
from model import ResidualUNet, ResidualUNetUncertainty
from train_step2 import build_dataset
from eval_protocol import CLEAN_PIXEL_THRESH
from validation_extras import StreamingQuantile, predict, CALIB_ROI, TEST_ROI

B4, B5, B6, B7, B8, B8A, B11, B12 = 3, 4, 5, 6, 7, 8, 11, 12
NIR_BANDS, UP_BANDS = (B6, B7, B8, B8A), (B4, B5, B11, B12)
VEG_NDVI = 0.3
DETECTORS = ("ndvi", "ndre", "ms")


def nd(img, a, b):
    return (img[a] - img[b]) / (img[a] + img[b] + 1e-6)


def apply_stress(target, s, rows, cols):
    out = target.copy()
    for b in NIR_BANDS:
        out[b, rows, cols] *= (1 - s)
    out[B4, rows, cols] *= (1 + s)
    for b in (B5, B11, B12):
        out[b, rows, cols] *= (1 + s / 2)
    return np.clip(out, 0, 1)


def scores(y, mu, sigma):
    s = np.maximum(sigma, 1e-4)
    signed = [-(y[b] - mu[b]) / s[b] for b in NIR_BANDS] + [(y[b] - mu[b]) / s[b] for b in UP_BANDS]
    return {
        "ndvi": nd(mu, B8, B4) - nd(y, B8, B4),
        "ndre": nd(mu, B8A, B5) - nd(y, B8A, B5),
        "ms": np.max(np.stack(signed), axis=0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--step2-model", required=True)
    ap.add_argument("--uncertainty-model", required=True)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--near-mode", default="prior", choices=["nearest", "prior"])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.05])
    ap.add_argument("--severities", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--spot", type=int, default=8, help="сторона квадрата стресу, пікселів")
    ap.add_argument("--mondrian", action="store_true",
                     help="окремий конформний поріг для кожного біна довжини розриву (--gap-bins)")
    ap.add_argument("--gap-bins", type=float, nargs="+", default=[30, 60, 120],
                     help="межі бінів довжини розриву ref->ціль, діб")
    ap.add_argument("--min-calib-px", type=int, default=20000,
                     help="мінімум калібрувальних пікселів у біні, інакше - глобальний поріг")
    ap.add_argument("--calib-roi", default=CALIB_ROI)
    ap.add_argument("--test-roi", default=TEST_ROI)
    ap.add_argument("--samples-per-patch", type=int, default=1)
    ap.add_argument("--ref-select", default="clearest", choices=["clearest", "latest_clean"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cpu")
    in_ch, out_ch = n_input_channels(), n_bands()
    step2 = ResidualUNet(in_ch, out_ch)
    step2.load_state_dict(torch.load(args.step2_model, map_location=device))
    unc = ResidualUNetUncertainty(in_ch, out_ch)
    unc.load_state_dict(torch.load(args.uncertainty_model, map_location=device))
    step2.eval(), unc.eval()

    kw = dict(near_mode=args.near_mode, samples_per_patch=args.samples_per_patch, ref_select=args.ref_select)
    calib_ds = build_dataset(args.root, [args.calib_roi], args.img_size, "random_clean", 0.3, args.seed, **kw)
    test_ds = build_dataset(args.root, [args.test_roi], args.img_size, "random_clean", 0.3, args.seed, **kw)

    # Конформні пороги: (1−α)-квантиль кожного score на чистих вегетуючих пікселях калібрувальної ділянки.
    # Mondrian-конформ: окремий поріг для кожного біна довжини розриву ref->ціль (днів), бо
    # невизначеність прогнозу з минулого різко зростає з розривом - один спільний поріг
    # "роздувається" довгими розривами й позбавляє чутливості короткі.
    edges = np.array(args.gap_bins, dtype=float)

    def gap_bin(ds, i):
        pk = ds.pick(i)
        g = abs(int(pk["t_days"][pk["target_idx"]] - pk["t_days"][pk["ref_idx"]]))
        return g, int(np.digitize(g, edges))

    n_bins = len(edges) + 1
    sq = {(d, b): StreamingQuantile() for d in DETECTORS for b in range(n_bins)}
    sq_all = {d: StreamingQuantile() for d in DETECTORS}
    for i in range(len(calib_ds)):
        _, t, tc, mu, _, sigma = predict(step2, unc, calib_ds, calib_ds, i, device)
        _, b = gap_bin(calib_ds, i)
        m = (tc < CLEAN_PIXEL_THRESH) & (nd(mu, B8, B4) > VEG_NDVI)
        for d, v in scores(t, mu, sigma).items():
            sq[(d, b)].add(v[m])
            sq_all[d].add(v[m])
    calib_counts = [sq[(DETECTORS[0], b)].n for b in range(n_bins)]
    thr_global = {a: {d: sq_all[d].conformal(a) for d in DETECTORS} for a in args.alphas}
    # Бін із замалою калібрувальною вибіркою - глобальний поріг (консервативний відступ).
    thr_bins = {a: {d: [sq[(d, b)].conformal(a) if sq[(d, b)].n >= args.min_calib_px else thr_global[a][d]
                        for b in range(n_bins)] for d in DETECTORS} for a in args.alphas}
    thr = None

    rng = np.random.default_rng(args.seed)
    rows_out = []
    for i in range(len(test_ds)):
        _, t, tc, mu, _, sigma = predict(step2, unc, test_ds, test_ds, i, device)
        pick = test_ds.pick(i)
        gap, gb = gap_bin(test_ds, i)
        thr = {a: {d: (thr_bins[a][d][gb] if args.mondrian else thr_global[a][d]) for d in DETECTORS}
               for a in args.alphas}
        # Вегетація - за ПРОГНОЗОМ μ, не за спостереженням: відбір за y зсуває розподіл y − μ.
        veg = (tc < CLEAN_PIXEL_THRESH) & (nd(mu, B8, B4) > VEG_NDVI)
        H, W = veg.shape
        k = args.spot
        density = np.lib.stride_tricks.sliding_window_view(veg, (k, k)).mean(axis=(-1, -2))
        cand = np.argwhere(density >= 0.75)
        if len(cand) == 0:
            continue
        r0, c0 = cand[rng.integers(len(cand))]
        spot = np.zeros_like(veg)
        spot[r0:r0 + k, c0:c0 + k] = True
        spot &= veg
        clean_ref = veg & ~spot

        sc0 = scores(t, mu, sigma)
        rec = {"patch": i, "gap_days": gap, "n_spot": int(spot.sum()), "fpr": {}, "tpr": {}}
        for a in args.alphas:
            rec["fpr"][str(a)] = {d: float((sc0[d][clean_ref] > thr[a][d]).mean()) for d in DETECTORS}
        for s in args.severities:
            ss = scores(apply_stress(t, s, *np.where(spot)), mu, sigma)
            for a in args.alphas:
                rec["tpr"][f"{s}|{a}"] = {d: float((ss[d][spot] > thr[a][d]).mean()) for d in DETECTORS}
        rows_out.append(rec)

    gaps = np.array([r["gap_days"] for r in rows_out])
    edges = np.percentile(gaps, [0, 100 / 3, 200 / 3, 100])
    strata = np.clip(np.digitize(gaps, edges[1:-1]), 0, 2)

    def boot_ci(vals, n=2000):
        vals = np.asarray(vals)
        r = np.random.default_rng(0)
        bs = [vals[r.integers(0, len(vals), len(vals))].mean() for _ in range(n)]
        return [float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    summary = {"n_patches": len(rows_out), "spot_px": args.spot, "img_size": args.img_size,
               "mondrian": args.mondrian, "gap_bins_days": args.gap_bins, "calib_px_per_bin": calib_counts,
               "thresholds_global": {str(a): v for a, v in thr_global.items()},
               "thresholds_by_gap_bin": {str(a): v for a, v in thr_bins.items()},
               "gap_tercile_edges_days": [float(e) for e in edges],
               "fpr": {}, "tpr": {}, "tpr_by_gap": {}}
    for a in args.alphas:
        summary["fpr"][str(a)] = {d: boot_ci([r["fpr"][str(a)][d] for r in rows_out]) for d in DETECTORS}
        summary.setdefault("fpr_by_gap", {})[str(a)] = {
            d: [float(np.mean([r["fpr"][str(a)][d] for r, st in zip(rows_out, strata) if st == b])) for b in range(3)]
            for d in DETECTORS}
        for s in args.severities:
            key = f"{s}|{a}"
            summary["tpr"][key] = {d: boot_ci([r["tpr"][key][d] for r in rows_out]) for d in DETECTORS}
            summary["tpr_by_gap"][key] = {
                d: [float(np.mean([r["tpr"][key][d] for r, st in zip(rows_out, strata) if st == b])) for b in range(3)]
                for d in DETECTORS}
        summary[f"paired_ms_minus_ndvi_tpr|{a}"] = {
            str(s): boot_ci([r["tpr"][f"{s}|{a}"]["ms"] - r["tpr"][f"{s}|{a}"]["ndvi"] for r in rows_out])
            for s in args.severities}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out / "per_patch.json").write_text(json.dumps(rows_out))

    lines = [f"# Конформне виявлення штучного стресу (near_mode={args.near_mode}, "
             f"{'Mondrian за розривом ' + str(args.gap_bins) if args.mondrian else 'спільний поріг'}), {len(rows_out)} фрагментів",
             f"Осередок {args.spot}×{args.spot} пікс., межі терцилей розриву (діб): "
             + ", ".join(f"{e:.0f}" for e in edges), ""]
    for a in args.alphas:
        lines += [f"## α = {a} (номінальна частка хибних тривог)", "",
                  "| | " + " | ".join(DETECTORS) + " |", "|---|" + "---|" * len(DETECTORS),
                  "| FPR | " + " | ".join(f"{summary['fpr'][str(a)][d][0]:.3f}" for d in DETECTORS) + " |"]
        for s in args.severities:
            k = f"{s}|{a}"
            lines.append(f"| TPR, s={s} | " + " | ".join(
                f"{summary['tpr'][k][d][0]:.3f} [{summary['tpr'][k][d][1]:.3f}; {summary['tpr'][k][d][2]:.3f}]"
                for d in DETECTORS) + " |")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
