#!/usr/bin/env python3
"""Генерує PNG-рисунки для статті з уже отриманих і верифікованих результатів.

fig1 - криві тренування кроку 2 (5 train ROI) з лог-файлу.
fig2 - ефект масштабування кількості тренувальних ROI (1/3/5) на ΔRMSE/ΔSAM,
       усі три точки - найкращий чекпоінт (early stopping), для чесного порівняння.
fig3 - вплив масштабу даних на якість конформного калібрування (покриття, гострота).
fig4 - якісний приклад реконструкції на реальному валідаційному патчі + карта
       похибки та карта невизначеності (σ) із навчених моделей.

Числа для fig2/fig3 узгоджені з REPORT.md (раніше перевірені в логах прогонів),
тут лише візуалізуються, не перераховуються.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from sen12mscrts_dataset import n_bands, n_input_channels
from model import ResidualUNet, ResidualUNetUncertainty
from train_step2 import build_dataset

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PALETTE = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a",
               good="#0ca30c", critical="#d03b3b",
               ink="#0b0b0b", secondary="#52514e", muted="#898781", grid="#e1e0d9")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": PALETTE["grid"],
    "axes.labelcolor": PALETTE["ink"],
    "text.color": PALETTE["ink"],
    "xtick.color": PALETTE["secondary"],
    "ytick.color": PALETTE["secondary"],
    "axes.grid": True,
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def _strip_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig1_training_curves():
    log_path = ROOT / "results" / "step2_final.log"
    epochs, train_l1, val_l1, baseline = [], [], [], None
    pattern = re.compile(
        r"epoch (\d+)/\d+\s+train_L1=([\d.]+)\s+val_L1=([\d.]+)\s+baseline_yref_L1=([\d.]+)")
    for line in log_path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            epochs.append(int(m.group(1)))
            train_l1.append(float(m.group(2)))
            val_l1.append(float(m.group(3)))
            baseline = float(m.group(4))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_l1, color=PALETTE["blue"], lw=2, marker="o", ms=4,
            label="Тренувальна вибірка (train L1)")
    ax.plot(epochs, val_l1, color=PALETTE["orange"], lw=2, marker="o", ms=4,
            label="Валідація, інший ROI (val L1)")
    ax.axhline(baseline, color=PALETTE["muted"], lw=1.5, ls="--",
               label=f"Базовий рівень y_ref ({baseline:.4f})")
    ax.set_xlabel("Епоха")
    ax.set_ylabel("L1 (MAE), шкала відбиття [0,1]")
    ax.set_title("Крок 2: тренування залишкової U-Net (4 train ROI, val≠test)")
    ax.legend(frameon=False, fontsize=9)
    _strip_spines(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_training_curves.png")
    plt.close(fig)


def fig2_scaling():
    # Усі три точки - ОДНАКОВА методологія: train ROI варіюється, val=asiaEast/119
    # (вибір епохи) і test=america/132 (звіт) фіксовані для всіх трьох, RMSE/MAE/SAM
    # маскують хмарні пікселі цілі. results/step2_{1,3}roi_v2 + results/step2_final (4 ROI).
    # CI - з bootstrap по тестових патчах (train_step2.py), не з повторних тренувань.
    labels = ["1 ROI", "3 ROI", "4 ROI"]
    d_rmse = [0.0028, -0.0032, -0.0059]
    rmse_lo = [0.0014, -0.0043, -0.0072]
    rmse_hi = [0.0043, -0.0022, -0.0048]
    d_sam = [-0.08, -0.51, -1.03]
    sam_lo = [-0.22, -0.68, -1.27]
    sam_hi = [0.05, -0.34, -0.79]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, deltas, lo, hi, ylabel, title in [
        (axes[0], d_rmse, rmse_lo, rmse_hi, "Δ RMSE (модель − базовий)", "RMSE: <0 означає краще за базовий"),
        (axes[1], d_sam, sam_lo, sam_hi, "Δ SAM, ° (модель − базовий)", "SAM: <0 означає краще за базовий"),
    ]:
        colors = [PALETTE["critical"] if v > 0 else PALETTE["good"] for v in deltas]
        err_lo = [d - l for d, l in zip(deltas, lo)]
        err_hi = [h - d for d, h in zip(deltas, hi)]
        ax.bar(labels, deltas, color=colors, width=0.5,
               yerr=[err_lo, err_hi], capsize=5,
               error_kw=dict(ecolor=PALETTE["ink"], lw=1.3, capthick=1.3))
        ax.axhline(0, color=PALETTE["ink"], lw=1)
        ax.set_xlabel("Кількість тренувальних ROI (val/test фіксовані, однакові для всіх)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        _strip_spines(ax)

    fig.suptitle("Ефект масштабування тренувальних ROI (вуса = 95% bootstrap CI по тестових патчах;\n"
                  "val≠test, маскування хмарних пікселів цілі - однаково для всіх точок)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_roi_scaling.png")
    plt.close(fig)


def fig3_calibration():
    # Обидві точки - ОДНАКОВИЙ calib ROI (asiaEast/139) і test ROI (america/132),
    # лише train варіюється: results/uncertainty_1roi_v2 vs results/uncertainty_final.
    labels = ["1 train ROI", "4 train ROI"]
    coverage = [88.7, 90.8]
    width = [0.2271, 0.1994]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(labels, coverage, color=PALETTE["blue"], width=0.5)
    axes[0].axhline(90, color=PALETTE["muted"], lw=1.5, ls="--", label="Номінальна ціль 90%")
    axes[0].set_ylabel("Фактичне покриття, %")
    axes[0].set_ylim(80, 100)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title("Калібрування: ближче до 90% = краще", fontsize=10)

    axes[1].bar(labels, width, color=PALETTE["orange"], width=0.5)
    axes[1].set_ylabel("Середня ширина інтервалу [0,1]")
    axes[1].set_title("Гострота: менше = інформативніше", fontsize=10)

    for ax in axes:
        _strip_spines(ax)
    fig.suptitle("Конформне калібрування: вплив масштабу тренувальних даних")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_calibration.png")
    plt.close(fig)


def _to_rgb(arr):
    """arr: (13, H, W) нормалізоване відбиття [0,1]. Повертає (H, W, 3) true-color."""
    rgb = np.stack([arr[3], arr[2], arr[1]], axis=-1)  # B4,B3,B2
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)


def _cloud_frac(img):
    bright = np.clip(img, 0, 1).mean(axis=0)
    return float(np.clip((bright - 0.25) / 0.20, 0, 1).mean())


def fig4_qualitative(n_candidates=25):
    ds = build_dataset(str(ROOT / "data/raw"), ["s2_america_test/ROIs1970/132"], img_size=128,
                        target_mode="random_clean", target_max_cloud=0.3, seed=0)

    # Для ілюстрації обираємо патч із візуально чистою ціллю (а не будь-який
    # достатньо чистий за порогом тренування) - інакше "справжнє значення" на
    # малюнку саме виглядає хмарним, що плутає читача.
    best_idx, best_frac = 0, 1.0
    for i in range(min(n_candidates, len(ds))):
        _, _, t, _, _ = ds[i]
        f = _cloud_frac(t.numpy())
        if f < best_frac:
            best_idx, best_frac = i, f
    print(f"Обрано патч #{best_idx} як приклад (хмарність цілі ~{best_frac:.1%} серед {n_candidates} кандидатів)")
    x, y_ref, target, target_cloud, _ = ds[best_idx]
    sample_idx = best_idx

    model = ResidualUNet(in_ch=n_input_channels(), out_ch=n_bands())
    model.load_state_dict(torch.load(ROOT / "results/step2_final/model.pt", map_location="cpu"))
    model.eval()
    unc = ResidualUNetUncertainty(in_ch=n_input_channels(), out_ch=n_bands())
    unc.load_state_dict(torch.load(ROOT / "results/uncertainty_final/model.pt", map_location="cpu"))
    unc.eval()

    with torch.no_grad():
        delta = model(x.unsqueeze(0))
        pred = (y_ref.unsqueeze(0) + delta).clamp(0, 1)[0]
        d2, log_sigma = unc(x.unsqueeze(0))
        sigma = torch.exp(log_sigma)[0]

    err = (pred - target).abs().mean(0).numpy()
    sigma_map = sigma.mean(0).numpy()

    fig, axes = plt.subplots(1, 5, figsize=(17, 3.4))
    for ax, title, img in [
        (axes[0], "y_ref\n(останнє чисте спостереження)", _to_rgb(y_ref.numpy())),
        (axes[1], "Ціль\n(справжнє значення)", _to_rgb(target.numpy())),
        (axes[2], "Прогноз моделі\n(крок 2)", _to_rgb(pred.numpy())),
    ]:
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    im3 = axes[3].imshow(err, cmap="Blues", vmin=0)
    axes[3].set_title("Абсолютна похибка\n(середнє по каналах)", fontsize=9)
    axes[3].axis("off")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    im4 = axes[4].imshow(sigma_map, cmap="Oranges", vmin=0)
    axes[4].set_title("σ невизначеності\n(середнє по каналах)", fontsize=9)
    axes[4].axis("off")
    fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    fig.suptitle(f"Приклад реконструкції на валідаційному патчі ROIs1970/132 (#{sample_idx})")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_qualitative_example.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1_training_curves()
    fig2_scaling()
    fig3_calibration()
    fig4_qualitative()
    print(f"Збережено рисунки в {OUT}")
