#!/usr/bin/env python3
"""
Додаткові перевірки до дисертації без нового тренування (на вже збережених model.pt):

  1. Базові рівні на тестовому ROI за тим самим протоколом, що й крок 2 (та сама ціль
     random_clean, та сама маска хмар цілі): y_ref, least_cloudy_strict, linear_interp.
     Цільовий кадр і кадри того самого дня виключено зі входу базових методів.
  2. Метрики в просторі NDVI (B8, B4) для моделі й базових рівнів.
  3. Конформне калібрування пари (μ з L1-моделі кроку 2, σ з σ-моделі методу 2):
     score = |y - μ_L1| / σ на калібрувальному ROI, квантиль зі скінченновибірковою
     поправкою ceil((n+1)(1-α))/n. Для порівняння - те саме для власного μ σ-моделі.
  4. Поріг low_confidence (p90 ширини інтервалу на патч) з КАЛІБРУВАЛЬНОГО ROI, а не з тесту.

Запуск (одна пара моделей):
    python scripts/validation_extras.py --root data/raw \
        --step2-model results/step2_final/model.pt \
        --uncertainty-model results/uncertainty_final/model.pt \
        --img-size 128 --seed 0 --out results/validation_extras/cpu_seed0
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from sen12mscrts_dataset import n_bands, n_input_channels, linear_interp_ref
from model import ResidualUNet, ResidualUNetUncertainty
from train_step2 import build_dataset, ndvi
from eval_protocol import all_metrics, rmse, mae, bootstrap_diff_ci, CLEAN_PIXEL_THRESH, TOL_SAME_DAYS

CALIB_ROI = "s2_asiaEast_test/ROIs1970/139"
TEST_ROI = "s2_america_test/ROIs1970/132"
HIST_EDGES = np.logspace(-6, 4, 400_001)


class StreamingQuantile:
    """Точний до ~6e-5 відносної похибки квантиль сотень мільйонів score без їх зберігання."""

    def __init__(self):
        self.counts = np.zeros(len(HIST_EDGES) + 1, dtype=np.int64)
        self.n = 0

    def add(self, values):
        idx = np.searchsorted(HIST_EDGES, values, side="right")
        self.counts += np.bincount(idx, minlength=len(self.counts))
        self.n += values.size

    def conformal(self, alpha):
        k = math.ceil((self.n + 1) * (1 - alpha))
        cum = np.cumsum(self.counts)
        b = int(np.searchsorted(cum, k))
        return float(HIST_EDGES[min(b, len(HIST_EDGES) - 1)])


def baselines(ds, i):
    pdx = ds.patch_indices[i]
    pick = ds._picks[pdx]
    frames, clouds = ds.full_series(i)
    t_days, tgt = pick["t_days"], pick["target_idx"]
    keep = np.abs(t_days - t_days[tgt]) > TOL_SAME_DAYS
    keep[tgt] = False
    cand = np.where(keep)[0]
    lc_strict = frames[cand[int(np.argmin(clouds[cand].reshape(len(cand), -1).mean(1)))]]
    return lc_strict, linear_interp_ref(frames, clouds, t_days, tgt), pick


def ndvi_metrics(pred, target, mask):
    return {"rmse": rmse(ndvi(pred)[None], ndvi(target)[None], mask),
            "mae": mae(ndvi(pred)[None], ndvi(target)[None], mask)}


def predict(step2, unc, ds, unc_ds, i, device):
    x, y_ref, target, target_cloud, _ = ds[i]
    x, y_ref = x[None].to(device), y_ref[None].to(device)
    if unc_ds is ds:
        xu, yu = x, y_ref
    else:
        xu, yu, *_ = unc_ds[i]
        xu, yu = xu[None].to(device), yu[None].to(device)
    with torch.no_grad():
        recon = (y_ref + step2(x)).clamp(0, 1)[0].cpu().numpy()
        delta, log_sigma = unc(xu)
        mu_u = (yu + delta).clamp(0, 1)[0].cpu().numpy()
        sigma = torch.exp(log_sigma)[0].cpu().numpy()
    return y_ref[0].cpu().numpy(), target.numpy(), target_cloud.numpy(), recon, mu_u, sigma


def patch_width(mu, sigma, q):
    return float((np.clip(mu + q * sigma, 0, 1) - np.clip(mu - q * sigma, 0, 1)).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--step2-model", required=True)
    ap.add_argument("--uncertainty-model", required=True)
    ap.add_argument("--img-size", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--low-conf-percentile", type=float, default=90.0)
    ap.add_argument("--ref-mode", default="prior_clean", choices=["prior_clean", "linear_interp"])
    ap.add_argument("--ref-cache-dir", default=None)
    ap.add_argument("--unc-ref-mode", default=None, choices=["prior_clean", "linear_interp"],
                     help="режим y_ref, на якому навчено σ-модель (за замовчуванням = --ref-mode)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = torch.device("cpu")
    in_ch, out_ch = n_input_channels(), n_bands()
    step2 = ResidualUNet(in_ch, out_ch).to(device)
    step2.load_state_dict(torch.load(args.step2_model, map_location=device))
    unc = ResidualUNetUncertainty(in_ch, out_ch).to(device)
    unc.load_state_dict(torch.load(args.uncertainty_model, map_location=device))
    step2.eval(), unc.eval()

    calib_ds = build_dataset(args.root, [CALIB_ROI], args.img_size, "random_clean", 0.3, args.seed,
                             ref_mode=args.ref_mode, ref_cache_dir=args.ref_cache_dir)
    test_ds = build_dataset(args.root, [TEST_ROI], args.img_size, "random_clean", 0.3, args.seed,
                             ref_mode=args.ref_mode, ref_cache_dir=args.ref_cache_dir)

    unc_mode = args.unc_ref_mode or args.ref_mode
    if unc_mode == args.ref_mode:
        calib_unc_ds, test_unc_ds = calib_ds, test_ds
    else:
        calib_unc_ds = build_dataset(args.root, [CALIB_ROI], args.img_size, "random_clean", 0.3, args.seed,
                                     ref_mode=unc_mode, ref_cache_dir=args.ref_cache_dir)
        test_unc_ds = build_dataset(args.root, [TEST_ROI], args.img_size, "random_clean", 0.3, args.seed,
                                    ref_mode=unc_mode, ref_cache_dir=args.ref_cache_dir)

    sq_l1, sq_u = StreamingQuantile(), StreamingQuantile()
    calib_sigma_patches = []
    for i in range(len(calib_ds)):
        _, t, tc, recon, mu_u, sigma = predict(step2, unc, calib_ds, calib_unc_ds, i, device)
        m = np.broadcast_to(tc < CLEAN_PIXEL_THRESH, t.shape)
        s = np.maximum(sigma, 1e-4)
        sq_l1.add((np.abs(t - recon) / s)[m])
        sq_u.add((np.abs(t - mu_u) / s)[m])
        calib_sigma_patches.append((recon, sigma))
    q_l1, q_u = sq_l1.conformal(args.alpha), sq_u.conformal(args.alpha)
    calib_widths = np.array([patch_width(r, s, q_l1) for r, s in calib_sigma_patches])
    width_cut = float(np.percentile(calib_widths, args.low_conf_percentile))
    del calib_sigma_patches

    rows = []
    cov = {"l1": [0, 0], "u": [0, 0]}
    wsum = {"l1": 0.0, "u": 0.0}
    for i in range(len(test_ds)):
        y_ref, t, tc, recon, mu_u, sigma = predict(step2, unc, test_ds, test_unc_ds, i, device)
        lc_strict, lin, pick = baselines(test_ds, i)
        mask = tc < CLEAN_PIXEL_THRESH
        m3 = np.broadcast_to(mask, t.shape)
        for key, mu, q in (("l1", recon, q_l1), ("u", mu_u, q_u)):
            inside = (t >= mu - q * sigma) & (t <= mu + q * sigma)
            cov[key][0] += int(inside[m3].sum())
            cov[key][1] += int(m3.sum())
            wsum[key] += float((2 * q * sigma)[m3].sum())
        days = pick["t_days"]
        rows.append({
            "gap_days": int(abs(days[pick["target_idx"]] - days[pick["ref_idx"]])),
            "near_after_target": bool(days[pick["near_idx"]] > days[pick["target_idx"]]),
            "model": all_metrics(recon, t, mask), "yref": all_metrics(y_ref, t, mask),
            "lc_strict": all_metrics(lc_strict, t, mask), "linear_interp": all_metrics(lin, t, mask),
            "ndvi": {k: ndvi_metrics(v, t, mask) for k, v in
                     (("model", recon), ("yref", y_ref), ("lc_strict", lc_strict), ("linear_interp", lin))},
            "width_l1": patch_width(recon, sigma, q_l1),
        })

    def mean(path):
        vals = [r for r in rows]
        for k in path:
            vals = [v[k] for v in vals]
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    def paired(a_path, b_path):
        def get(path):
            out = []
            for r in rows:
                v = r
                for k in path:
                    v = v[k]
                out.append(v)
            return out
        return bootstrap_diff_ci(get(a_path), get(b_path))

    flagged = np.array([r["width_l1"] > width_cut for r in rows])
    rm = np.array([r["model"]["rmse"] for r in rows])
    summary = {
        "img_size": args.img_size, "seed": args.seed, "ref_mode": args.ref_mode, "unc_ref_mode": unc_mode, "n_test": len(rows), "n_calib": len(calib_ds),
        "q_l1_pair": q_l1, "q_sigma_model_own_mu": q_u,
        "coverage_l1_pair": cov["l1"][0] / cov["l1"][1], "coverage_sigma_model": cov["u"][0] / cov["u"][1],
        "mean_width_l1_pair": wsum["l1"] / cov["l1"][1], "mean_width_sigma_model": wsum["u"] / cov["u"][1],
        "low_conf_cut_from_calib": width_cut, "low_conf_frac_test": float(flagged.mean()),
        "rmse_low_conf": [float(np.nanmean(rm[flagged])), float(np.nanstd(rm[flagged])), int(flagged.sum())],
        "rmse_trust": [float(np.nanmean(rm[~flagged])), float(np.nanstd(rm[~flagged])), int((~flagged).sum())],
        "frac_near_after_target": float(np.mean([r["near_after_target"] for r in rows])),
        "refl": {m: {k: mean([m, k]) for k in ("rmse", "mae", "ssim", "sam_deg")}
                 for m in ("model", "yref", "lc_strict", "linear_interp")},
        "ndvi": {m: {k: mean(["ndvi", m, k]) for k in ("rmse", "mae")}
                 for m in ("model", "yref", "lc_strict", "linear_interp")},
        "paired": {
            f"model-{b}_{metric}": paired(["model", metric], [b, metric])
            for b in ("yref", "lc_strict", "linear_interp") for metric in ("rmse", "sam_deg")
        } | {
            f"ndvi_model-{b}_rmse": paired(["ndvi", "model", "rmse"], ["ndvi", b, "rmse"])
            for b in ("yref", "lc_strict", "linear_interp")
        } | {"linear_interp-yref_rmse": paired(["linear_interp", "rmse"], ["yref", "rmse"])},
    }
    if flagged.any() and (~flagged).any():
        summary["paired_low_conf_vs_trust_rmse_unpaired_diff"] = float(np.nanmean(rm[flagged]) - np.nanmean(rm[~flagged]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out / "per_patch.json").write_text(json.dumps(rows))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
