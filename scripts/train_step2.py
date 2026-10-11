#!/usr/bin/env python3
"""
Крок 2 (прототип): детермінована залишкова U-Net ŷ = y_ref + Δθ(x).

Критерій прийняття з плану: модель має бути НЕ ГІРШОЮ за базовий метод за RMSE і SAM
(парний тест). Тут базовий метод для порівняння - "скопіювати y_ref" (Δ=0), це
структурний аналог least_cloudy_strict/linear_interp: обидва намагаються оцінити
ціль з останнього доступного чистого спостереження, лише без навченої поправки.

Три незалежні групи ROI (train/val/test), усі - різні географічні локації:
    --train-roi - на чому вчимо
    --val-roi   - лише для вибору найкращої епохи (early stopping)
    --test-roi  - лише для фінального звіту; НЕ бачиться моделлю й не впливає на вибір епохи

Раніше val і test були тим самим ROI (вибір чекпоінту й фінальне число рахувались на
одному наборі) - це давало оптимістичний зсув. Тепер вони розділені.

Хмарні пікселі ЦІЛІ маскуються в RMSE/MAE/SAM (target_cloud < 0.2): target_max_cloud
- це поріг на СЕРЕДНЮ хмарність патча, тож окремі пікселі цілі можуть лишатись
хмарними; порівнювати прогноз із хмарою замість поверхні немає сенсу. SSIM
НЕ маскується (віконна метрика, див. eval_protocol.ssim_score).

Запуск:
    python train_step2.py --root data/raw --train-roi s2_america_test/ROIs1158/106 \
        --val-roi s2_asiaEast_test/ROIs1868/119 --test-roi s2_america_test/ROIs1970/132 \
        --epochs 10 --img-size 128 --out results/step2_v3
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
from eval_protocol import all_metrics, bootstrap_diff_ci, CLEAN_PIXEL_THRESH


def count_patches(root, roi_dir, roi):
    d = Path(root) / roi_dir / roi / "S2" / "0"
    return len([f for f in d.iterdir() if f.suffix == ".tif"])


def resolve_roi_spec(root, spec):
    """'s2_america_test/ROIs1158/106' -> (root/s2_america_test, 'ROIs1158', '106')."""
    parts = spec.strip("/").split("/")
    roi_dir, roi = parts[-2], parts[-1]
    prefix = parts[:-2]
    full_root = Path(root).joinpath(*prefix) if prefix else Path(root)
    return full_root, roi_dir, roi


def build_dataset(root, specs, img_size, target_mode, target_max_cloud, seed, max_patches_per_roi=None):
    """max_patches_per_roi дозволяє зрівняти ЗАГАЛЬНИЙ обсяг даних між прогонами з різною
    кількістю ROI (контроль обсягу при перевірці ефекту різноманітності локацій) - бере
    перші N патчів з кожного переліченого ROI замість усіх."""
    from torch.utils.data import ConcatDataset
    parts = []
    for spec in specs:
        r, roi_dir, roi = resolve_roi_spec(root, spec)
        n = count_patches(r, roi_dir, roi)
        if max_patches_per_roi is not None:
            n = min(n, max_patches_per_roi)
        ds = SEN12MSCRTSPatchDataset(root=r, roi_dir=roi_dir, roi=roi, patch_indices=range(n),
                                      img_size=img_size, target_mode=target_mode,
                                      target_max_cloud=target_max_cloud, seed=seed)
        print(f"  {spec}: {n} патчів")
        parts.append(ds)
    return ConcatDataset(parts) if len(parts) > 1 else parts[0]


def evaluate(model, loader, device, clean_thresh=CLEAN_PIXEL_THRESH):
    """Повертає список словників per-sample метрик для моделі й для базового рівня y_ref.

    RMSE/MAE/SAM рахуються лише по пікселях цілі з cloud < clean_thresh (SSIM - без
    маски, див. eval_protocol.ssim_score)."""
    model.eval()
    results = []
    with torch.no_grad():
        for x, y_ref, target, target_cloud in loader:
            x, y_ref, target = x.to(device), y_ref.to(device), target.to(device)
            delta = model(x)
            pred = (y_ref + delta).clamp(0, 1)
            for b in range(x.shape[0]):
                p = pred[b].cpu().numpy()
                r = y_ref[b].cpu().numpy()
                t = target[b].cpu().numpy()
                mask = target_cloud[b].numpy() < clean_thresh
                results.append({
                    "model": all_metrics(p, t, mask=mask),
                    "baseline_yref": all_metrics(r, t, mask=mask),
                    "n_clear_px": int(mask.sum()), "n_px": int(mask.size),
                })
    return results


def summarize(results, key):
    rows = [r[key] for r in results]
    return {m: (float(np.nanmean([row[m] for row in rows])), float(np.nanstd([row[m] for row in rows])))
            for m in ["rmse", "mae", "ssim", "sam_deg"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="коренева директорія (data/raw)")
    ap.add_argument("--train-roi", action="append", required=True,
                     help="'<підшлях>/<ROIs-dir>/<roi>', можна кілька разів для кількох локацій")
    ap.add_argument("--val-roi", action="append", required=True,
                     help="лише для вибору найкращої епохи - МАЄ бути іншим ROI, ніж --train-roi/--test-roi")
    ap.add_argument("--test-roi", action="append", required=True,
                     help="лише для фінального звіту - МАЄ бути іншим ROI, ніж --train-roi/--val-roi")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=128, help="просторовий даунсемпл для швидкості прототипу")
    ap.add_argument("--target-mode", default="random_clean", choices=["argmin", "random_clean"])
    ap.add_argument("--target-max-cloud", type=float, default=0.3)
    ap.add_argument("--max-train-patches-per-roi", type=int, default=None,
                     help="обмежити кожен train ROI цією кількістю патчів - для контролю "
                          "загального обсягу при порівнянні кількості ROI")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/step2_prototype")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    all_specs = set(args.train_roi) | set(args.val_roi) | set(args.test_roi)
    if len(all_specs) != len(args.train_roi) + len(args.val_roi) + len(args.test_roi):
        sys.exit("train-roi / val-roi / test-roi перетинаються - вибір епохи й фінальний звіт будуть нечесними")

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    print("Train ROI:")
    train_ds = build_dataset(args.root, args.train_roi, args.img_size, args.target_mode,
                              args.target_max_cloud, args.seed, args.max_train_patches_per_roi)
    print("Val ROI (лише для вибору епохи, інша локація):")
    val_ds = build_dataset(args.root, args.val_roi, args.img_size, args.target_mode,
                            args.target_max_cloud, args.seed)
    print("Test ROI (лише для фінального звіту, ще інша локація):")
    test_ds = build_dataset(args.root, args.test_roi, args.img_size, args.target_mode,
                             args.target_max_cloud, args.seed)
    print(f"Разом: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = ResidualUNet(in_ch=n_input_channels(), out_ch=n_bands()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val, best_epoch, best_state = float("inf"), None, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y_ref, target, _ in train_loader:
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
            for x, y_ref, target, _ in val_loader:
                x, y_ref, target = x.to(device), y_ref.to(device), target.to(device)
                pred = y_ref + model(x)
                val_losses.append(criterion(pred, target).item())
                baseline_losses.append(criterion(y_ref, target).item())
        val_l1 = float(np.mean(val_losses))
        marker = ""
        if val_l1 < best_val:
            best_val, best_epoch = val_l1, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = "  (найкращий дотепер)"
        print(f"epoch {epoch}/{args.epochs}  train_L1={np.mean(train_losses):.4f}  "
              f"val_L1={val_l1:.4f}  baseline_yref_L1={np.mean(baseline_losses):.4f}{marker}")

    print(f"\nНайкраща епоха за val_L1: {best_epoch} (val_L1={best_val:.4f}) - саме цей чекпоінт оцінюється й зберігається.")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "model.pt")

    results = evaluate(model, test_loader, device)  # ТЕСТ, не val - епоху обирали за val
    model_tab = summarize(results, "model")
    base_tab = summarize(results, "baseline_yref")
    n_clear = sum(r["n_clear_px"] for r in results)
    n_total = sum(r["n_px"] for r in results)

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

    rmse_sig = "значуще краще" if hi_rmse < 0 else ("не значуще (CI перетинає 0)" if lo_rmse < 0 < hi_rmse else "значуще гірше")
    sam_sig = "значуще краще" if hi_sam < 0 else ("не значуще (CI перетинає 0)" if lo_sam < 0 < hi_sam else "значуще гірше")

    report = [
        "# Крок 2 (прототип): залишкова U-Net проти копіювання y_ref",
        "",
        f"Train ROI: {', '.join(args.train_roi)} ({len(train_ds)} патчів).",
        f"Val ROI (вибір епохи): {', '.join(args.val_roi)} ({len(val_ds)} патчів).",
        f"Test ROI (фінальний звіт, НЕ бачений при виборі епохи): {', '.join(args.test_roi)} ({len(test_ds)} патчів).",
        f"Епох: {args.epochs}, розмір зображення (даунсемпл): {args.img_size}x{args.img_size}. "
        f"Найкраща епоха за val_L1: {best_epoch}.",
        f"RMSE/MAE/SAM рахувались лише по пікселях цілі з хмарністю < {CLEAN_PIXEL_THRESH} "
        f"({n_clear}/{n_total} = {100*n_clear/max(n_total,1):.1f}% пікселів test-набору; SSIM - без маски). "
        f"Лосс тренування й вибору епохи - НЕ маскований (повний target).",
        "",
        f"- Модель (y_ref + Δθ):  {fmt(model_tab)}",
        f"- Базовий рівень (y_ref): {fmt(base_tab)}",
        "",
        f"Δ RMSE (модель - базовий) = {d_rmse:+.4f} [{lo_rmse:+.4f}, {hi_rmse:+.4f}] - {rmse_sig}",
        f"Δ SAM  (модель - базовий) = {d_sam:+.2f}° [{lo_sam:+.2f}, {hi_sam:+.2f}] - {sam_sig}",
        "",
    ]
    criterion_met = hi_rmse <= 0 and hi_sam <= 0  # обидва мають бути ЗНАЧУЩО не гірші, не лише точково
    report.append(
        "Критерій кроку 2 (статистично значуще не гірше за базовий метод за RMSE І SAM) "
        + ("ВИКОНАНО" if criterion_met else "НЕ виконано") + " на цьому прототипі."
    )
    report.append(
        f"\nЦе прототип на {len(args.train_roi)} train-, {len(args.val_roi)} val- і {len(args.test_roi)} test-ROI, "
        "1 запуск (не >=5 із різними seed) - недостатньо для дисертаційного висновку. CI тут відображає лише "
        "розкид між тестовими патчами в межах цього одного навченого запуску, НЕ розкид між повторними тренуваннями."
    )

    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    (out_dir / "test_metrics_raw.json").write_text(json.dumps(results, indent=2))
    print("\n" + "\n".join(report))
    print(f"\nЗбережено: {out_dir / 'report.md'}, {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
