#!/usr/bin/env python3
"""
Діагностика конформного калібрування (монографія, розд. 6.3-6.5) без нового навчання.

Пара «μ L1-моделі з y_ref = інтерполяція, σ σ-моделі» (як у results/validation_extras/interp_gpu_seed*):
  1. діаграма надійності: q для кожного α з калібрувальної ділянки, фактичне покриття на тесті;
  2. блоковий бутстреп покриття: блок = фрагмент 256 × 256 (2,56 км), 2000 повторів; фрагмент менший
     за радіус варіограми (8-29 км), тож сусідні фрагменти корельовані й інтервал - нижня оцінка розкиду;
  3. розкид покриття між фрагментами (5-й і 95-й перцентилі);
  4. крива «ризик-покриття»: RMSE прийнятих пікселів, якщо відкидати пікселі з найбільшою σ,
     проти оракула (відкидання за фактичною помилкою) і випадкового відкидання;
  5. кореляція середньої σ фрагмента з RMSE фрагмента (рівень блоків, а не пікселів).

    python scripts/calib_diagnostics.py --root data/raw --seeds 0 1 2 3 4 --out results/calib_diagnostics
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

ALPHAS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
SIGMA_EDGES = np.logspace(-4, 1, 501)   # біни σ для кривої ризик-покриття
ERR_EDGES = np.logspace(-6, 0.5, 501)   # біни |похибки| для оракула


def spearman(a, b):
    """Рангова кореляція без scipy (у середовищі scipy несумісний із numpy); нічиїх у неперервних даних немає."""
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def run_seed(seed, args, device):
    step2 = ResidualUNet(n_input_channels(), n_bands()).to(device)
    step2.load_state_dict(torch.load(args.step2_model.format(seed=seed), map_location=device))
    unc = ResidualUNetUncertainty(n_input_channels(), n_bands()).to(device)
    unc.load_state_dict(torch.load(args.uncertainty_model.format(seed=seed), map_location=device))
    step2.eval(), unc.eval()
    common = dict(img_size=args.img_size, target_mode="random_clean", target_max_cloud=0.3, seed=seed,
                  ref_cache_dir=args.ref_cache_dir, near_mode="nearest", ref_select="clearest",
                  max_patches_per_roi=args.max_patches)

    def pair(spec):
        mu_ds = build_dataset(args.root, [spec], ref_mode="linear_interp", **common)
        if args.unc_ref_mode == "linear_interp":  # σ-голова поверх тієї самої моделі (A6): той самий вхід
            return mu_ds, mu_ds
        return mu_ds, build_dataset(args.root, [spec], ref_mode=args.unc_ref_mode, **common)

    calib, calib_u = pair(args.calib_roi)
    sq = StreamingQuantile()
    for i in range(len(calib)):
        _, t, tc, mu, _, sigma = predict(step2, unc, calib, calib_u, i, device)
        m = np.broadcast_to(tc < CLEAN_PIXEL_THRESH, t.shape)
        sq.add((np.abs(t - mu) / np.maximum(sigma, 1e-4))[m])
    qs = {a: sq.conformal(a) for a in ALPHAS}

    test, test_u = pair(args.test_roi)
    cov = np.zeros((len(test), len(ALPHAS)))   # покриті пікселі по фрагментах
    tot = np.zeros(len(test))
    p_sigma, p_rmse = np.zeros(len(test)), np.zeros(len(test))
    sig_n, sig_se = np.zeros(len(SIGMA_EDGES) + 1), np.zeros(len(SIGMA_EDGES) + 1)
    err_n, err_se = np.zeros(len(ERR_EDGES) + 1), np.zeros(len(ERR_EDGES) + 1)
    for i in range(len(test)):
        _, t, tc, mu, _, sigma = predict(step2, unc, test, test_u, i, device)
        m = np.broadcast_to(tc < CLEAN_PIXEL_THRESH, t.shape)
        e, s = (t - mu)[m], sigma[m]
        tot[i] = e.size
        for j, a in enumerate(ALPHAS):
            cov[i, j] = np.count_nonzero(np.abs(e) <= qs[a] * s)
        if e.size:
            p_sigma[i], p_rmse[i] = s.mean(), np.sqrt(np.mean(e ** 2))
        bs = np.searchsorted(SIGMA_EDGES, s)
        sig_n += np.bincount(bs, minlength=len(sig_n))
        sig_se += np.bincount(bs, weights=e ** 2, minlength=len(sig_se))
        be = np.searchsorted(ERR_EDGES, np.abs(e))
        err_n += np.bincount(be, minlength=len(err_n))
        err_se += np.bincount(be, weights=e ** 2, minlength=len(err_se))

    rng = np.random.default_rng(seed)
    keep = tot > 0
    cov, tot = cov[keep], tot[keep]
    boot = np.array([cov[idx].sum(0) / tot[idx].sum()
                     for idx in (rng.integers(0, len(tot), len(tot)) for _ in range(args.n_boot))])
    per_patch = cov / tot[:, None]

    def risk_curve(n, se):
        """Від найменших σ (або похибок) до найбільших: частка прийнятих -> RMSE прийнятих."""
        cn, cse = np.cumsum(n), np.cumsum(se)
        frac = cn / cn[-1]
        return [(float(f), float(np.sqrt(x / c))) for f, x, c in zip(frac, cse, cn) if c > 0]

    def at(curve, fracs=(0.5, 0.7, 0.8, 0.9, 0.95, 1.0)):
        return {str(f): min(curve, key=lambda p: abs(p[0] - f))[1] for f in fracs}

    sigma_curve, oracle_curve = risk_curve(sig_n, sig_se), risk_curve(err_n, err_se)
    return {
        "seed": seed, "n_test_patches": int(keep.sum()), "n_calib_pixels": sq.n,
        "q": {str(a): qs[a] for a in ALPHAS},
        "coverage": {str(a): float(cov[:, j].sum() / tot.sum()) for j, a in enumerate(ALPHAS)},
        "coverage_block_ci95": {str(a): [float(np.percentile(boot[:, j], 2.5)), float(np.percentile(boot[:, j], 97.5))]
                                for j, a in enumerate(ALPHAS)},
        "coverage_per_patch_p5_p95": {str(a): [float(np.percentile(per_patch[:, j], 5)),
                                               float(np.percentile(per_patch[:, j], 95))]
                                      for j, a in enumerate(ALPHAS)},
        "risk_coverage_rmse": {"sigma": at(sigma_curve), "oracle": at(oracle_curve),
                               "random": float(np.sqrt(sig_se.sum() / sig_n.sum()))},
        "patch_sigma_vs_rmse_spearman": spearman(p_sigma[keep], p_rmse[keep]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--calib-roi", default=CALIB_ROI)
    ap.add_argument("--test-roi", default=TEST_ROI)
    ap.add_argument("--step2-model", default="results/step2_interp_gpu_256_seed{seed}/model.pt")
    ap.add_argument("--uncertainty-model", default="results/uncertainty_gpu_256_seed{seed}/model.pt")
    ap.add_argument("--unc-ref-mode", default="prior_clean", choices=["prior_clean", "linear_interp"],
                    help="y_ref, з яким навчено σ-модель; linear_interp - для σ-голови поверх замороженої моделі (A6)")
    ap.add_argument("--ref-cache-dir", default="data/ref_cache")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--max-patches", type=int, default=None, help="лише для швидкої перевірки")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        f = out / f"seed{seed}.json"
        if f.exists():
            print(f"seed {seed}: вже є", flush=True)
            continue
        s = run_seed(seed, args, device)
        f.write_text(json.dumps(s, indent=2, ensure_ascii=False))
        print(f"seed {seed}: покриття α=0,1 {100 * s['coverage']['0.1']:.1f} % "
              f"[{100 * s['coverage_block_ci95']['0.1'][0]:.1f}; {100 * s['coverage_block_ci95']['0.1'][1]:.1f}], "
              f"ρ(σ, RMSE) за фрагментами {s['patch_sigma_vs_rmse_spearman']:.2f}", flush=True)


if __name__ == "__main__":
    main()
