#!/usr/bin/env python3
"""
Експеримент D (монографія, розд. 5.8 і 6.3): уже навчені моделі на НОВИХ тестових ділянках.

Без навчання й без калібрувальних даних: множник q пари «μ L1-моделі з y_ref = інтерполяція, σ σ-моделі»
береться з калібрування на asiaEast/139, що вже виконане validation_extras.py
(results/validation_extras/interp_gpu_seed{seed}/summary.json, поле q_l1_pair). Для кожної нової ділянки
й кожного seed рахуються ті самі метрики, що й на america/132: модель, лінійна інтерполяція,
найменш хмарний кадр без цілі, RMSE NDVI, парні різниці, покриття й ширина інтервалу.

    python scripts/multi_roi_eval.py --root data/raw --roi s2_europa_test/ROIs1868/17 --seeds 0 1 2 3 4 \
        --out results/multi_roi
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
from validation_extras import baselines, ndvi_metrics, predict, patch_width


def evaluate(root, spec, seed, args, device):
    step2 = ResidualUNet(n_input_channels(), n_bands()).to(device)
    step2.load_state_dict(torch.load(args.step2_model.format(seed=seed), map_location=device))
    unc = ResidualUNetUncertainty(n_input_channels(), n_bands()).to(device)
    unc.load_state_dict(torch.load(args.uncertainty_model.format(seed=seed), map_location=device))
    step2.eval(), unc.eval()
    q = json.load(open(args.calib_summary.format(seed=seed)))["q_l1_pair"]

    common = dict(img_size=args.img_size, target_mode="random_clean", target_max_cloud=0.3, seed=seed,
                  ref_cache_dir=args.ref_cache_dir, near_mode="nearest", ref_select="clearest",
                  max_patches_per_roi=args.max_patches)
    ds = build_dataset(root, [spec], ref_mode="linear_interp", **common)
    unc_ds = build_dataset(root, [spec], ref_mode="prior_clean", **common)  # σ-модель навчено з y_ref = копіювання

    rows, covered, total, wsum = [], 0, 0, 0.0
    for i in range(len(ds)):
        _, t, tc, recon, _, sigma = predict(step2, unc, ds, unc_ds, i, device)
        lc_strict, lin, _ = baselines(ds, i)
        mask = tc < CLEAN_PIXEL_THRESH
        m3 = np.broadcast_to(mask, t.shape)
        inside = (t >= recon - q * sigma) & (t <= recon + q * sigma)
        covered += int(inside[m3].sum())
        total += int(m3.sum())
        wsum += float((2 * q * sigma)[m3].sum())
        rows.append({"model": all_metrics(recon, t, mask), "linear_interp": all_metrics(lin, t, mask),
                     "lc_strict": all_metrics(lc_strict, t, mask),
                     "ndvi": {k: ndvi_metrics(v, t, mask) for k, v in
                              (("model", recon), ("linear_interp", lin), ("lc_strict", lc_strict))},
                     "width": patch_width(recon, sigma, q)})

    def col(*path):
        out = []
        for r in rows:
            v = r
            for k in path:
                v = v[k]
            out.append(v)
        return np.array(out, float)

    methods = ("model", "linear_interp", "lc_strict")
    return {
        "roi": spec, "seed": seed, "n_test": len(rows), "q_l1_pair_from_asiaEast139": q,
        "coverage": covered / max(total, 1), "mean_width": wsum / max(total, 1),
        "refl": {m: {k: float(np.nanmean(col(m, k))) for k in ("rmse", "mae", "ssim", "sam_deg")} for m in methods},
        "ndvi_rmse": {m: float(np.nanmean(col("ndvi", m, "rmse"))) for m in methods},
        "paired": {
            "rmse_model-linear_interp": bootstrap_diff_ci(col("model", "rmse"), col("linear_interp", "rmse")),
            "sam_model-linear_interp": bootstrap_diff_ci(col("model", "sam_deg"), col("linear_interp", "sam_deg")),
            "ndvi_model-linear_interp": bootstrap_diff_ci(col("ndvi", "model", "rmse"),
                                                          col("ndvi", "linear_interp", "rmse")),
            "rmse_linear_interp-lc_strict": bootstrap_diff_ci(col("linear_interp", "rmse"), col("lc_strict", "rmse")),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--roi", action="append", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--step2-model", default="results/step2_interp_gpu_256_seed{seed}/model.pt")
    ap.add_argument("--uncertainty-model", default="results/uncertainty_gpu_256_seed{seed}/model.pt")
    ap.add_argument("--calib-summary", default="results/validation_extras/interp_gpu_seed{seed}/summary.json")
    ap.add_argument("--ref-cache-dir", default="data/ref_cache")
    ap.add_argument("--max-patches", type=int, default=None, help="лише для швидкої перевірки")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for spec in args.roi:
        for seed in args.seeds:
            out = Path(args.out) / spec.replace("/", "_") / f"seed{seed}"
            if (out / "summary.json").exists():
                print(f"{spec} seed {seed}: вже є - пропускаю", flush=True)
                continue
            s = evaluate(args.root, spec, seed, args, device)
            out.mkdir(parents=True, exist_ok=True)
            (out / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False))
            d = s["paired"]["rmse_model-linear_interp"]
            print(f"{spec} seed {seed}: n={s['n_test']} RMSE модель {s['refl']['model']['rmse']:.4f} "
                  f"інтерп. {s['refl']['linear_interp']['rmse']:.4f} Δ={d} покриття {100 * s['coverage']:.1f} %",
                  flush=True)


if __name__ == "__main__":
    main()
