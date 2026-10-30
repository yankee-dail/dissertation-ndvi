#!/usr/bin/env python3
"""
Експеримент A (монографія, розд. 5.7 і 6): UnCRtainTS за НАШИМ протоколом.

Попереднє порівняння (REPORT §13.3) брало числа з test_reconstruct.py UnCRtainTS - інший вибір
цілі (найменш хмарний кадр ряду), 3 хмарні входи, метрики без маски. Тут на тих самих фрагментах,
що й validation_extras.py, і з тією самою ціллю (random_clean, seed):
  1. вхід UnCRtainTS - 3 кадри, найближчі в часі до цілі (ціль і кадри того самого дня виключено);
  2. метрики RMSE/MAE/SAM/RMSE NDVI - лише за чистими пікселями цілі (CLEAN_PIXEL_THRESH),
     парні різниці з linear_interp і (опційно) з нашою залишковою моделлю на тих самих цілях;
  3. конформне калібрування σ UnCRtainTS на калібрувальній ділянці (q - ⌈(n+1)(1−α)⌉/n), покриття
     й ширина на тесті - перевірка, що метод 2 не залежить від моделі, яка видає σ.

Запуск у Colab (GPU), після uncrtaints_minimal.py (UnCRtainTS клоновано, ваги noSAR_1 завантажено):
    python scripts/uncrtaints_ourprotocol.py --root data/raw \
        --uncrt-dir /content/UnCRtainTS --weights /content/UnCRtainTS/results \
        --calib-roi s2_asiaEast_test/ROIs1970/139 --seeds 0 1 2 3 4 \
        --our-model results/step2_interp_gpu_256_seed{seed}/model.pt \
        --out results/uncrtaints_ourprotocol
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from sen12mscrts_dataset import n_bands, n_input_channels, linear_interp_ref
from model import ResidualUNet
from train_step2 import build_dataset, ndvi
from eval_protocol import all_metrics, rmse, mae, bootstrap_diff_ci, CLEAN_PIXEL_THRESH, TOL_SAME_DAYS
from validation_extras import StreamingQuantile, TEST_ROI


def load_uncrtaints(uncrt_dir, weights, experiment, device):
    """Модель і конфіг так само, як у test_reconstruct.py, але без його argparse на рівні модуля."""
    model_dir = Path(uncrt_dir) / "model"
    sys.path.insert(0, str(model_dir))
    sys.path.insert(0, str(Path(uncrt_dir)))
    from parse_args import create_parser
    from src import utils
    from src.model_utils import get_model, load_checkpoint

    parser = create_parser(mode="test")
    cfg = parser.parse_args(["--experiment_name", experiment, "--weight_folder", weights, "--device", device])
    with open(Path(weights) / experiment / "conf.json") as f:
        saved = json.load(f)
    keep = {"experiment_name", "weight_folder", "device", "resume_at"}
    for k, v in saved.items():
        if k not in keep:
            setattr(cfg, k, v)
    cfg = utils.str2list(cfg, ["encoder_widths", "decoder_widths", "out_conv"])
    model = get_model(cfg).to(device)
    ckpt = f"_epoch_{cfg.resume_at}" if cfg.resume_at > 0 else ""
    load_checkpoint(cfg, weights, model, f"model{ckpt}")
    model.eval()
    return model, cfg


def input_frames(pick, n_in):
    """n_in кадрів, найближчих у часі до цілі (без цілі й кадрів того самого дня), у хронологічному порядку."""
    t_days, tgt = pick["t_days"], pick["target_idx"]
    cand = [t for t in range(len(t_days)) if abs(t_days[t] - t_days[tgt]) > TOL_SAME_DAYS]
    cand = sorted(cand, key=lambda t: abs(t_days[t] - t_days[tgt]))[:n_in]
    return sorted(cand, key=lambda t: t_days[t])


def predict_uncrtaints(model, frames, clouds, t_days, idx, device):
    """-> (mu, sigma) розміру (13, H, W) у шкалі відбиття [0, 1]."""
    x = torch.from_numpy(frames[idx])[None].to(device)                 # (1, T, 13, H, W)
    m = torch.from_numpy(clouds[idx])[None, :, None].to(device)        # (1, T, 1, H, W)
    dates = torch.tensor(t_days[idx], dtype=torch.float32)[None].to(device)
    with torch.no_grad():
        model.set_input({"A": x, "B": x[:, :1], "dates": dates, "masks": m})
        model.forward()
        model.get_loss_G()
        model.rescale()
        mu = model.fake_B[0, 0, : n_bands()]
        var = model.netG.variance
        model.netG.variance = None
    var = var[0, 0]
    if var.dim() == 4:  # повна коваріація (C, C, H, W) -> діагональ
        var = torch.diagonal(var, dim1=0, dim2=1).permute(2, 0, 1)
    sigma = var.clamp(min=1e-12).sqrt()
    return mu.clamp(0, 1).cpu().numpy(), sigma.cpu().numpy()


def run_split(ds, model, device, n_in, our=None):
    """Генератор по фрагментах: (target, mask, mu, sigma, interp, ours|None)."""
    for i in range(len(ds)):
        pick = ds.pick(i)
        frames, clouds = ds.full_series(i)
        x, y_ref, target, target_cloud, _ = ds[i]
        idx = input_frames(pick, n_in)
        mu, sigma = predict_uncrtaints(model, frames, clouds, pick["t_days"], idx, device)
        mine = None
        if our is not None:
            with torch.no_grad():
                mine = (y_ref[None].to(device) + our(x[None].to(device))).clamp(0, 1)[0].cpu().numpy()
        yield target.numpy(), target_cloud.numpy() < CLEAN_PIXEL_THRESH, mu, sigma, y_ref.numpy(), mine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--uncrt-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--experiment", default="noSAR_1")
    ap.add_argument("--test-roi", default=TEST_ROI)
    ap.add_argument("--calib-roi", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--input-t", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--our-model", default=None,
                    help="шлях до model.pt залишкової моделі з y_ref = linear_interp; {seed} підставляється")
    ap.add_argument("--ref-cache-dir", default="data/ref_cache")
    ap.add_argument("--max-patches", type=int, default=None, help="лише для швидкої перевірки")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_uncrtaints(args.uncrt_dir, args.weights, args.experiment, device)
    print(f"UnCRtainTS {args.experiment}: use_sar={getattr(cfg, 'use_sar', None)}, covmode={cfg.covmode}, "
          f"scale_by={cfg.scale_by}, device={device}")
    if getattr(cfg, "use_sar", False):
        sys.exit("Ця конфігурація вимагає SAR-входу - візьми noSAR_* ваги")

    for seed in args.seeds:
        out = Path(args.out) / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        common = dict(img_size=args.img_size, target_mode="random_clean", target_max_cloud=0.3, seed=seed,
                      max_patches_per_roi=args.max_patches, ref_mode="linear_interp",
                      ref_cache_dir=args.ref_cache_dir)
        calib = build_dataset(args.root, [args.calib_roi], **common)
        test = build_dataset(args.root, [args.test_roi], **common)

        our = None
        if args.our_model:
            our = ResidualUNet(n_input_channels(), n_bands()).to(device)
            our.load_state_dict(torch.load(args.our_model.format(seed=seed), map_location=device))
            our.eval()

        # 1) калібрування: score = |y − μ| / σ за чистими пікселями цілі (усі канали)
        sq = StreamingQuantile()
        for target, mask, mu, sigma, *_ in run_split(calib, model, device, args.input_t):
            m = np.broadcast_to(mask, target.shape)
            sq.add((np.abs(target - mu) / np.maximum(sigma, 1e-4))[m])
        q = sq.conformal(args.alpha)
        print(f"seed {seed}: q = {q:.4f} (n = {sq.n})")

        # 2) тест: метрики, покриття, ширина
        per_patch, covered, total, width_sum = [], 0, 0, 0.0
        for target, mask, mu, sigma, interp, mine in run_split(test, model, device, args.input_t, our):
            m = np.broadcast_to(mask, target.shape)
            lo, hi = mu - q * sigma, mu + q * sigma
            covered += int(((target >= lo) & (target <= hi))[m].sum())
            total += int(m.sum())
            width_sum += float((hi - lo)[m].sum())
            rec = {"uncrtaints": all_metrics(mu, target, mask=mask),
                   "linear_interp": all_metrics(interp, target, mask=mask),
                   "ndvi_rmse": {"uncrtaints": rmse(ndvi(mu)[None], ndvi(target)[None], mask),
                                 "linear_interp": rmse(ndvi(interp)[None], ndvi(target)[None], mask)},
                   "mean_sigma": float(sigma[m].mean()) if m.any() else None}
            if mine is not None:
                rec["ours"] = all_metrics(mine, target, mask=mask)
                rec["ndvi_rmse"]["ours"] = rmse(ndvi(mine)[None], ndvi(target)[None], mask)
            per_patch.append(rec)

        def mean(path):
            vals = [r for r in per_patch]
            for k in path:
                vals = [v[k] for v in vals]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            return float(np.mean(vals))

        def paired(a, b):
            va, vb = [], []
            for r in per_patch:
                x, y = r, r
                for k in a:
                    x = x[k]
                for k in b:
                    y = y[k]
                if np.isfinite(x) and np.isfinite(y):
                    va.append(x), vb.append(y)
            return bootstrap_diff_ci(np.array(va), np.array(vb))

        methods = ["uncrtaints", "linear_interp"] + (["ours"] if our is not None else [])
        summary = {
            "seed": seed, "n_test": len(per_patch), "n_calib": len(calib), "input_t": args.input_t,
            "alpha": args.alpha, "q": q, "coverage": covered / total, "mean_width": width_sum / total,
            "metrics": {mth: {k: mean([mth, k]) for k in ("rmse", "mae", "ssim", "sam_deg")} for mth in methods},
            "ndvi_rmse": {mth: mean(["ndvi_rmse", mth]) for mth in methods},
            "paired": {
                "rmse_uncrtaints-linear_interp": paired(["uncrtaints", "rmse"], ["linear_interp", "rmse"]),
                "sam_uncrtaints-linear_interp": paired(["uncrtaints", "sam_deg"], ["linear_interp", "sam_deg"]),
            },
        }
        if our is not None:
            summary["paired"]["rmse_uncrtaints-ours"] = paired(["uncrtaints", "rmse"], ["ours", "rmse"])
            summary["paired"]["sam_uncrtaints-ours"] = paired(["uncrtaints", "sam_deg"], ["ours", "sam_deg"])
            summary["paired"]["ndvi_uncrtaints-ours"] = paired(["ndvi_rmse", "uncrtaints"], ["ndvi_rmse", "ours"])
        (out / "per_patch.json").write_text(json.dumps(per_patch, indent=1))
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
