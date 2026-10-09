#!/usr/bin/env python3
"""
Крок 2 (прототип): детермінована залишкова U-Net ŷ = y_ref + Δθ(x).

Критерій прийняття з плану: модель має бути НЕ ГІРШОЮ за базовий метод за RMSE і SAM
(парний тест). Тут базовий метод для порівняння - "скопіювати y_ref" (Δ=0), це
структурний аналог least_cloudy_strict/linear_interp: обидва намагаються оцінити
ціль з останнього доступного чистого спостереження, лише без навченої поправки.

Важливо: дані - лише один ROI (ROIs1868/100, asiaWest, test split SEN12MS-CR-TS).
Тренувальний і валідаційний патчі розділені БЛОКОМ індексів (не випадково), щоб
хоч трохи зменшити (не усунути) просторову витікання між суміжними тайлами одного
поля - але це прототип на малих даних, не фінальний дисертаційний результат.

Запуск:
    python train_step2.py --root data/raw/s2_asiaWest_test --roi-dir ROIs1868 --roi 100 \
        --epochs 5 --img-size 128 --out results/step2_prototype
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from sen12mscrts_dataset import SEN12MSCRTSPatchDataset, n_bands, n_input_channels
from model import ResidualUNet
from eval_protocol import rmse, mae, ssim_score, sam_degrees, bootstrap_diff_ci


def count_patches(root, roi_dir, roi):
    d = Path(root) / roi_dir / roi / "S2" / "0"
    return len([f for f in d.iterdir() if f.suffix == ".tif"])


def split_indices(n, val_frac, min_val=10):
    n_val = max(min_val, int(round(n * val_frac)))
    n_val = min(n_val, n - 1)
    return list(range(0, n - n_val)), list(range(n - n_val, n))


def evaluate(model, loader, device):
    """Повертає список словників per-sample метрик для моделі й для базового рівня y_ref."""
    model.eval()
    results = []
    with torch.no_grad():
        for x, y_ref, target in loader:
            x, y_ref, target = x.to(device), y_ref.to(device), target.to(device)
            delta = model(x)
            pred = (y_ref + delta).clamp(0, 1)
            for b in range(x.shape[0]):
                p = pred[b].cpu().numpy()
                r = y_ref[b].cpu().numpy()
                t = target[b].cpu().numpy()
                results.append({
                    "model": {"rmse": rmse(p, t), "mae": mae(p, t), "ssim": ssim_score(p, t),
                              "sam_deg": sam_degrees(p, t)},
                    "baseline_yref": {"rmse": rmse(r, t), "mae": mae(r, t), "ssim": ssim_score(r, t),
                                       "sam_deg": sam_degrees(r, t)},
                })
    return results


def summarize(results, key):
    rows = [r[key] for r in results]
    return {m: (float(np.mean([row[m] for row in rows])), float(np.std([row[m] for row in rows])))
            for m in ["rmse", "mae", "ssim", "sam_deg"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--roi-dir", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=128, help="просторовий даунсемпл для швидкості прототипу")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/step2_prototype")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    n_patches = count_patches(args.root, args.roi_dir, args.roi)
    train_idx, val_idx = split_indices(n_patches, args.val_frac)
    print(f"Патчів у ROI {args.roi_dir}/{args.roi}: {n_patches} "
          f"(train={len(train_idx)}, val={len(val_idx)}, блочний поділ за індексом)")

    common = dict(root=args.root, roi_dir=args.roi_dir, roi=args.roi, img_size=args.img_size, seed=args.seed)
    train_ds = SEN12MSCRTSPatchDataset(patch_indices=train_idx, **common)
    val_ds = SEN12MSCRTSPatchDataset(patch_indices=val_idx, **common)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ResidualUNet(in_ch=n_input_channels(), out_ch=n_bands()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y_ref, target in train_loader:
            x, y_ref, target = x.to(device), y_ref.to(device), target.to(device)
            delta = model(x)
            pred = y_ref + delta
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, baseline_losses = [], []
        with torch.no_grad():
            for x, y_ref, target in val_loader:
                x, y_ref, target = x.to(device), y_ref.to(device), target.to(device)
                pred = y_ref + model(x)
                val_losses.append(criterion(pred, target).item())
                baseline_losses.append(criterion(y_ref, target).item())
        print(f"epoch {epoch}/{args.epochs}  train_L1={np.mean(train_losses):.4f}  "
              f"val_L1={np.mean(val_losses):.4f}  baseline_yref_L1={np.mean(baseline_losses):.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")

    results = evaluate(model, val_loader, device)
    model_tab = summarize(results, "model")
    base_tab = summarize(results, "baseline_yref")

    rmse_model = [r["model"]["rmse"] for r in results]
    rmse_base = [r["baseline_yref"]["rmse"] for r in results]
    sam_model = [r["model"]["sam_deg"] for r in results]
    sam_base = [r["baseline_yref"]["sam_deg"] for r in results]
    d_rmse, lo_rmse, hi_rmse = bootstrap_diff_ci(rmse_model, rmse_base, n_boot=args.n_boot, seed=args.seed)
    d_sam, lo_sam, hi_sam = bootstrap_diff_ci(sam_model, sam_base, n_boot=args.n_boot, seed=args.seed)

    def fmt(tab):
        return (f"RMSE {tab['rmse'][0]:.4f}±{tab['rmse'][1]:.4f}  "
                f"MAE {tab['mae'][0]:.4f}±{tab['mae'][1]:.4f}  "
                f"SSIM {tab['ssim'][0]:.4f}±{tab['ssim'][1]:.4f}  "
                f"SAM {tab['sam_deg'][0]:.2f}±{tab['sam_deg'][1]:.2f}°")

    report = [
        "# Крок 2 (прототип): залишкова U-Net проти копіювання y_ref",
        "",
        f"ROI: {args.roi_dir}/{args.roi}. Патчів: train={len(train_idx)}, val={len(val_idx)} "
        f"(блочний поділ за індексом - лише один ROI, можлива просторова кореляція між блоками).",
        f"Епох: {args.epochs}, розмір зображення (даунсемпл): {args.img_size}x{args.img_size}.",
        "",
        f"- Модель (y_ref + Δθ):  {fmt(model_tab)}",
        f"- Базовий рівень (y_ref): {fmt(base_tab)}",
        "",
        f"Δ RMSE (модель - базовий) = {d_rmse:+.4f} [{lo_rmse:+.4f}, {hi_rmse:+.4f}] "
        f"({'краще' if d_rmse < 0 else 'гірше'}, якщо <0)",
        f"Δ SAM  (модель - базовий) = {d_sam:+.2f}° [{lo_sam:+.2f}, {hi_sam:+.2f}] "
        f"({'краще' if d_sam < 0 else 'гірше'}, якщо <0)",
        "",
    ]
    criterion_met = d_rmse <= 0 and d_sam <= 0
    report.append(
        "Критерій кроку 2 (не гірше за базовий метод за RMSE і SAM) "
        + ("ВИКОНАНО" if criterion_met else "НЕ виконано") + " на цьому прототипі."
    )
    report.append(
        "\nЦе прототип на одному ROI (300 патчів, 1 локація) з короткою кількістю епох - "
        "недостатньо для дисертаційного висновку. Потрібно: більше ROI/регіонів, "
        "повна роздільність, >=5 запусків з різними seed, статистична поправка на множинні порівняння."
    )

    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    (out_dir / "val_metrics_raw.json").write_text(json.dumps(results, indent=2))
    print("\n" + "\n".join(report))
    print(f"\nЗбережено: {out_dir / 'report.md'}, {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
