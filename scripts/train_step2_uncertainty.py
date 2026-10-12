#!/usr/bin/env python3
"""
Метод 2 концепції: оцінка невизначеності реконструкції + конформне калібрування.

ŷ = y_ref + Δθ(x), σ = exp(log_σθ(x)) - гетероскедастична голова (основний варіант,
не ансамбль і не дифузія). Навчання - Gaussian NLL:
    L = 0.5*log(2π σ²) + (y - ŷ)² / (2σ²)

Конформне калібрування: на ОКРЕМОМУ калібрувальному ROI (інша локація, ніж train
і test) підбирається множник q такий, що інтервал [ŷ - qσ, ŷ + qσ] покриває цільове
значення з заданою ймовірністю (напр. 90%) - за часткою нонконформних оцінок
r_i = |y_i - ŷ_i| / σ_i на калібрувальному наборі. Покриття потім перевіряється на
ТРЕТЬОМУ, зовсім іншому ROI (test), щоб уникнути оптимістичного зсуву.

Три ROI (train / calib / test) - три різні географічні локації, без перетину.

Запуск:
    python train_step2_uncertainty.py --root data/raw \
        --train-roi s2_america_test/ROIs1158/106 --train-roi s2_africa_test/ROIs2017/32 \
        --train-roi s2_africa_test/ROIs2017/140 --train-roi s2_asiaEast_test/ROIs1868/73 \
        --train-roi s2_asiaEast_test/ROIs1868/119 \
        --calib-roi s2_asiaEast_test/ROIs1970/139 \
        --test-roi s2_america_test/ROIs1970/132 \
        --epochs 10 --img-size 128 --coverage 0.9 --out results/step2_uncertainty
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
from sen12mscrts_dataset import n_bands, n_input_channels
from model import ResidualUNetUncertainty
from train_step2 import build_dataset
from eval_protocol import all_metrics, CLEAN_PIXEL_THRESH


def gaussian_nll(delta, log_sigma, y_ref, target):
    mu = y_ref + delta
    sigma2 = torch.exp(2 * log_sigma)
    return (0.5 * np.log(2 * np.pi) + log_sigma + (target - mu) ** 2 / (2 * sigma2)).mean()


def predict(model, x, y_ref):
    delta, log_sigma = model(x)
    mu = (y_ref + delta).clamp(0, 1)
    sigma = torch.exp(log_sigma)
    return mu, sigma


def nonconformity_scores(model, loader, device, clean_thresh=CLEAN_PIXEL_THRESH):
    """r_i = |target - mu| / sigma, лише по пікселях ЦІЛІ з cloud < clean_thresh.

    Калібрування за хмарними пікселями цілі було б калібруванням проти хмари, а не
    проти поверхні - спотворило б q у довільний бік."""
    model.eval()
    scores = []
    with torch.no_grad():
        for x, y_ref, target, target_cloud, _ in loader:
            x, y_ref, target = x.to(device), y_ref.to(device), target.to(device)
            mu, sigma = predict(model, x, y_ref)
            r = (target - mu).abs() / sigma.clamp(min=1e-4)
            mask = (target_cloud < clean_thresh).unsqueeze(1).expand_as(r)  # той самий маск на всі канали
            scores.append(r[mask].cpu().numpy())
    return np.concatenate(scores)


def evaluate_with_intervals(model, loader, device, q, clean_thresh=CLEAN_PIXEL_THRESH):
    model.eval()
    per_sample = []
    coverage_flags = []
    widths = []
    n_clear_total, n_px_total = 0, 0
    with torch.no_grad():
        for x, y_ref, target, target_cloud, _ in loader:
            x, y_ref, target = x.to(device), y_ref.to(device), target.to(device)
            mu, sigma = predict(model, x, y_ref)
            lo, hi = mu - q * sigma, mu + q * sigma
            covered = (target >= lo) & (target <= hi)
            px_mask = (target_cloud < clean_thresh).unsqueeze(1).expand_as(covered)
            # Покриття рахується лише по чистих пікселях цілі - не можна перевірити,
            # чи інтервал "накрив" значення, якого саме через хмару не видно насправді.
            coverage_flags.append(covered[px_mask].cpu().numpy())
            widths.append((hi - lo)[px_mask].cpu().numpy())
            n_clear_total += int(px_mask.sum())
            n_px_total += int(px_mask.numel())
            for b in range(x.shape[0]):
                m, t, r = mu[b].cpu().numpy(), target[b].cpu().numpy(), y_ref[b].cpu().numpy()
                mask = target_cloud[b].numpy() < clean_thresh
                per_sample.append({
                    "model_mu": all_metrics(m, t, mask=mask),
                    "baseline_yref": all_metrics(r, t, mask=mask),
                    "mean_sigma": float(sigma[b].mean().cpu()),
                })
    coverage = float(np.concatenate(coverage_flags).mean())
    mean_width = float(np.concatenate(widths).mean())
    return per_sample, coverage, mean_width, n_clear_total, n_px_total


def summarize(results, key):
    rows = [r[key] for r in results]
    return {m: (float(np.nanmean([row[m] for row in rows])), float(np.nanstd([row[m] for row in rows])))
            for m in ["rmse", "mae", "ssim", "sam_deg"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--train-roi", action="append", required=True)
    ap.add_argument("--calib-roi", action="append", required=True, help="ROI лише для підбору q, інший за train/test")
    ap.add_argument("--test-roi", action="append", required=True, help="ROI лише для звіту про покриття")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--target-mode", default="random_clean", choices=["argmin", "random_clean"])
    ap.add_argument("--target-max-cloud", type=float, default=0.3)
    ap.add_argument("--coverage", type=float, default=0.9, help="номінальний рівень покриття інтервалу")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/step2_uncertainty")
    args = ap.parse_args()

    all_specs = set(args.train_roi) | set(args.calib_roi) | set(args.test_roi)
    if len(all_specs) != len(args.train_roi) + len(args.calib_roi) + len(args.test_roi):
        sys.exit("train-roi / calib-roi / test-roi перетинаються - калібрування й оцінка будуть нечесними")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Пристрій: {device}")

    print("Train ROI:")
    train_ds = build_dataset(args.root, args.train_roi, args.img_size, args.target_mode, args.target_max_cloud, args.seed)
    print("Calib ROI (для підбору q, окрема локація):")
    calib_ds = build_dataset(args.root, args.calib_roi, args.img_size, args.target_mode, args.target_max_cloud, args.seed)
    print("Test ROI (для звіту про покриття, окрема локація):")
    test_ds = build_dataset(args.root, args.test_roi, args.img_size, args.target_mode, args.target_max_cloud, args.seed)
    print(f"Разом: train={len(train_ds)}, calib={len(calib_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    calib_loader = DataLoader(calib_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = ResidualUNetUncertainty(in_ch=n_input_channels(), out_ch=n_bands()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val, best_epoch, best_state = float("inf"), None, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y_ref, target, _, ref_days in train_loader:
            x, y_ref, target = x.to(device), y_ref.to(device), target.to(device)
            delta, log_sigma = model(x)
            loss = gaussian_nll(delta, log_sigma, y_ref, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        calib_losses = []
        with torch.no_grad():
            for x, y_ref, target, _, _ in calib_loader:
                x, y_ref, target = x.to(device), y_ref.to(device), target.to(device)
                delta, log_sigma = model(x)
                calib_losses.append(gaussian_nll(delta, log_sigma, y_ref, target).item())
        calib_nll = float(np.mean(calib_losses))
        marker = ""
        if calib_nll < best_val:
            best_val, best_epoch = calib_nll, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = "  (найкращий дотепер)"
        print(f"epoch {epoch}/{args.epochs}  train_NLL={np.mean(train_losses):.4f}  "
              f"calib_NLL={calib_nll:.4f}{marker}")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "model.pt")

    # Конформне калібрування: q = емпіричний (1-alpha)-квантиль нонконформних оцінок на calib-наборі.
    scores = nonconformity_scores(model, calib_loader, device)
    q = float(np.quantile(scores, args.coverage))
    print(f"\nКонформний множник q (рівень {args.coverage:.0%}): {q:.3f} (з {len(scores)} піксель-каналів calib-набору)")

    test_results, test_coverage, test_width, n_clear, n_px = evaluate_with_intervals(model, test_loader, device, q)
    model_tab = summarize(test_results, "model_mu")
    base_tab = summarize(test_results, "baseline_yref")

    def fmt(tab):
        return (f"RMSE {tab['rmse'][0]:.4f}±{tab['rmse'][1]:.4f}  MAE {tab['mae'][0]:.4f}±{tab['mae'][1]:.4f}  "
                f"SSIM {tab['ssim'][0]:.4f}±{tab['ssim'][1]:.4f}  SAM {tab['sam_deg'][0]:.2f}±{tab['sam_deg'][1]:.2f}°")

    report = [
        "# Метод 2 (прототип): гетероскедастична голова + конформне калібрування",
        "",
        f"Train ROI: {', '.join(args.train_roi)} ({len(train_ds)} патчів).",
        f"Calib ROI: {', '.join(args.calib_roi)} ({len(calib_ds)} патчів) - лише для підбору q.",
        f"Test ROI: {', '.join(args.test_roi)} ({len(test_ds)} патчів) - лише для звіту про покриття.",
        f"Усі три - різні географічні локації. Найкраща епоха за calib NLL: {best_epoch}.",
        "",
        f"Номінальне покриття: {args.coverage:.0%}. Конформний множник q = {q:.3f}.",
        f"**Фактичне покриття на test ROI: {test_coverage:.1%}** (ціль: {args.coverage:.0%}). "
        f"Середня ширина інтервалу: {test_width:.4f} (у шкалі [0,1] відбиття).",
        f"Покриття, ширина, RMSE/MAE/SAM рахувались лише по пікселях цілі з хмарністю < {CLEAN_PIXEL_THRESH} "
        f"({n_clear}/{n_px} = {100*n_clear/max(n_px,1):.1f}% пікселів test-набору; SSIM - без маски). "
        f"Калібрування (підбір q) також рахувалось лише по чистих пікселях calib-набору.",
        "",
        f"- μ-прогноз моделі: {fmt(model_tab)}",
        f"- Базовий рівень (y_ref): {fmt(base_tab)}",
        "",
        "Інтерпретація: якщо фактичне покриття помітно нижче номінального - модель переоцінює "
        "свою впевненість (σ занижена) і калібрування потрібно посилити (більший q або перетренування "
        "голови невизначеності); якщо значно вище - інтервали завширокі (неінформативні), калібрування "
        "занадто консервативне.",
        "",
        "Застереження: calib/test ROI - по одній локації кожен (не per-страта: культура/фаза/рельєф, "
        "як того вимагає методологія концепції), тож покриття - це грубий прототип гарантії, не "
        "дисертаційний результат. Потрібно більше ROI на страту й блокове калібрування з відстанню "
        "між блоками більшою за радіус варіограми.",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    (out_dir / "test_metrics_raw.json").write_text(json.dumps(test_results, indent=2))
    print("\n" + "\n".join(report))
    print(f"\nЗбережено: {out_dir / 'report.md'}, {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
