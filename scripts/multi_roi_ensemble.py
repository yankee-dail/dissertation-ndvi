#!/usr/bin/env python3
"""
Експеримент D2 (монографія, розд. 5.8 і 6.3): дві перевірки на нових тестових ділянках експерименту D
без навчання.

  1. Ансамбль: μ = середнє п'яти моделей з y_ref = інтерполяція (seed 0-4) на тих самих цілях, що й у D
     (ціль для «запуску» s - та, яку бачила модель seed s). Порівняння з інтерполяцією та з окремою
     моделлю seed s: RMSE, SAM, RMSE NDVI, парний бутстреп за фрагментами.
  2. Калібрування в межах регіону: для кожної ділянки europa q рахується на чотирьох інших ділянках europa
     (leave-one-ROI-out), для asiaWest/100 - на всіх п'ятьох europa; для порівняння - q з asiaEast/139,
     як у D. Пари: «μ моделі seed s + σ σ-моделі seed s» (як у D) і «μ ансамблю + середня σ п'яти σ-моделей».

Score |y - μ| / σ накопичуються гістограмами (StreamingQuantile), тож покриття для будь-якого q
рахується без повторного прогону; гістограми кешуються в --cache (не для git).

    python scripts/multi_roi_ensemble.py --root data/raw --roi s2_asiaWest_test/ROIs1868/100 \
        --roi s2_europa_test/ROIs1868/142 ... --out results/multi_roi_ens
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
from eval_protocol import all_metrics, bootstrap_diff_ci, CLEAN_PIXEL_THRESH
from validation_extras import StreamingQuantile, HIST_EDGES, baselines, ndvi_metrics

SEEDS = (0, 1, 2, 3, 4)
ALPHA = 0.1


def load(args, device):
    mu, sg = [], []
    for s in SEEDS:
        m = ResidualUNet(n_input_channels(), n_bands()).to(device)
        m.load_state_dict(torch.load(args.step2_model.format(seed=s), map_location=device))
        u = ResidualUNetUncertainty(n_input_channels(), n_bands()).to(device)
        u.load_state_dict(torch.load(args.uncertainty_model.format(seed=s), map_location=device))
        mu.append(m.eval()), sg.append(u.eval())
    return mu, sg


def run_roi_seed(spec, seed, mus, sigmas, args, device):
    """Метрики й гістограми score для однієї ділянки й одного набору цілей (seed)."""
    common = dict(img_size=args.img_size, target_mode="random_clean", target_max_cloud=0.3, seed=seed,
                  ref_cache_dir=args.ref_cache_dir, near_mode="nearest", ref_select="clearest",
                  max_patches_per_roi=args.max_patches)
    ds = build_dataset(args.root, [spec], ref_mode="linear_interp", **common)
    uds = build_dataset(args.root, [spec], ref_mode="prior_clean", **common)  # σ-моделі навчено з y_ref = копіювання
    h_single, h_ens = StreamingQuantile(), StreamingQuantile()
    sig_single = sig_ens = 0.0
    rows = []
    for i in range(len(ds)):
        x, y_ref, t, tc, _ = ds[i]
        xu, yu, *_ = uds[i]
        x, y_ref, xu = x[None].to(device), y_ref[None].to(device), xu[None].to(device)
        with torch.no_grad():
            recs = [(y_ref + m(x)).clamp(0, 1)[0].cpu().numpy() for m in mus]
            sig = [torch.exp(u(xu)[1])[0].cpu().numpy() for u in sigmas]
        t, tc = t.numpy(), tc.numpy()
        ens, sig_e = np.mean(recs, 0), np.mean(sig, 0)
        single, sig_s = recs[seed], sig[seed]
        _, lin, _ = baselines(ds, i)
        mask = tc < CLEAN_PIXEL_THRESH
        m3 = np.broadcast_to(mask, t.shape)
        h_single.add(np.abs(t - single)[m3] / np.maximum(sig_s[m3], 1e-4))
        h_ens.add(np.abs(t - ens)[m3] / np.maximum(sig_e[m3], 1e-4))
        sig_single += float(sig_s[m3].sum())
        sig_ens += float(sig_e[m3].sum())
        rows.append({k: {**{m: v for m, v in all_metrics(p, t, mask).items() if m in ("rmse", "sam_deg")},
                         "ndvi_rmse": ndvi_metrics(p, t, mask)["rmse"]}
                     for k, p in (("ensemble", ens), ("single", single), ("linear_interp", lin))})

    def col(k, m):
        return np.array([r[k][m] for r in rows], float)

    metrics = {"roi": spec, "seed": seed, "n_test": len(rows),
               "mean": {k: {m: float(np.nanmean(col(k, m))) for m in ("rmse", "sam_deg", "ndvi_rmse")}
                        for k in ("ensemble", "single", "linear_interp")},
               "paired": {f"{m}_{a}-{b}": bootstrap_diff_ci(col(a, m), col(b, m))
                          for m in ("rmse", "sam_deg", "ndvi_rmse")
                          for a, b in (("ensemble", "linear_interp"), ("single", "linear_interp"),
                                       ("ensemble", "single"))}}
    hist = {"single": h_single.counts, "ensemble": h_ens.counts,
            "sigma_sum": np.array([sig_single, sig_ens]), "n": np.array([h_single.n, h_ens.n])}
    return metrics, hist


def q_from(counts_list, n):
    """Конформний квантиль за об'єднаними гістограмами (та сама поправка, що й у StreamingQuantile)."""
    sq = StreamingQuantile()
    sq.counts, sq.n = np.sum(counts_list, 0), int(n)
    return sq.conformal(ALPHA)


def coverage(counts, n, q):
    b = int(np.searchsorted(HIST_EDGES, q))  # q - межа біна; score <= q лежать у бінах 0..b
    return float(np.cumsum(counts)[b] / n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--roi", action="append", required=True)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--step2-model", default="results/step2_interp_gpu_256_seed{seed}/model.pt")
    ap.add_argument("--uncertainty-model", default="results/uncertainty_gpu_256_seed{seed}/model.pt")
    ap.add_argument("--calib-summary", default="results/validation_extras/interp_gpu_seed{seed}/summary.json")
    ap.add_argument("--ref-cache-dir", default="data/ref_cache")
    ap.add_argument("--cache", default="data/multi_roi_ens_cache")
    ap.add_argument("--region", default="europa", help="ділянки, чиє ім'я містить це, калібрують одна одну")
    ap.add_argument("--max-patches", type=int, default=None, help="лише для швидкої перевірки")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mus, sigmas = load(args, device)
    out, cache = Path(args.out), Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    H = {}
    for spec in args.roi:
        tag = spec.replace("/", "_")
        for s in SEEDS:
            mf, hf = out / tag / f"seed{s}.json", cache / f"{tag}_seed{s}.npz"
            if mf.exists() and hf.exists():
                print(f"{spec} seed {s}: вже є", flush=True)
            else:
                m, h = run_roi_seed(spec, s, mus, sigmas, args, device)
                mf.parent.mkdir(parents=True, exist_ok=True)
                mf.write_text(json.dumps(m, indent=2, ensure_ascii=False))
                np.savez_compressed(hf, **h)
                p = m["mean"]
                print(f"{spec} seed {s}: RMSE анс. {p['ensemble']['rmse']:.4f} окрема {p['single']['rmse']:.4f} "
                      f"інтерп. {p['linear_interp']['rmse']:.4f}; NDVI анс. {p['ensemble']['ndvi_rmse']:.4f} "
                      f"окрема {p['single']['ndvi_rmse']:.4f} інтерп. {p['linear_interp']['ndvi_rmse']:.4f}", flush=True)
            H[spec, s] = dict(np.load(hf))

    # калібрування: q з asiaEast/139 (як у D) проти q з інших ділянок europa
    europa = [r for r in args.roi if args.region in r]
    cal = []
    for spec in args.roi:
        donors = [r for r in europa if r != spec]
        if not donors:
            continue
        for s in SEEDS:
            h = H[spec, s]
            row = {"roi": spec, "seed": s, "calib_rois": donors}
            for j, var in enumerate(("single", "ensemble")):
                n = h["n"][j]
                q_reg = q_from([H[r, s][var] for r in donors], sum(H[r, s]["n"][j] for r in donors))
                row[var] = {"q_region": q_reg, "coverage_region": coverage(h[var], n, q_reg),
                            "width_region": 2 * q_reg * float(h["sigma_sum"][j]) / n}
                if var == "single":
                    q_ae = json.load(open(args.calib_summary.format(seed=s)))["q_l1_pair"]
                    row[var].update(q_asiaEast=q_ae, coverage_asiaEast=coverage(h[var], n, q_ae),
                                    width_asiaEast=2 * q_ae * float(h["sigma_sum"][j]) / n)
            cal.append(row)
            print(f"{spec} seed {s}: покриття q(asiaEast) {100 * row['single']['coverage_asiaEast']:.1f} %, "
                  f"q(europa) окрема {100 * row['single']['coverage_region']:.1f} %, "
                  f"ансамбль {100 * row['ensemble']['coverage_region']:.1f} %", flush=True)
    (out / "calibration.json").write_text(json.dumps(cal, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
