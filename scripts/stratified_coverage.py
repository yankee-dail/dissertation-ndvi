#!/usr/bin/env python3
"""
Умовне покриття за стратами (§10, п.6 плану; §4 основного результату) - перевіряє, чи
конформне покриття `results/uncertainty_final` стабільне всередині різних умов на
ОДНОМУ й тому самому test ROI, а не лише агреговане по всій території (як було раніше).
Без нового тренування - лише перерахунок покриття на вже навченому чекпоінті з
розбиттям патчів на страти за (а) хмарністю near-кадру (проксі складності умов) і
(б) місяцем цільової дати (сезон).

Запуск:
    python stratified_coverage.py --root data/raw \
        --test-roi s2_america_test/ROIs1970/132 \
        --model results/uncertainty_final/model.pt --q 1.207 \
        --out results/stratified_coverage
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from sen12mscrts_dataset import n_bands, n_input_channels
from model import ResidualUNetUncertainty
from train_step2 import build_dataset
from eval_protocol import CLEAN_PIXEL_THRESH

CLOUD_NEAR_IDX = 26  # y_ref(13) + x_near(13) -> канал 26 = cloud_near (без S1)


def per_patch_coverage_width(model, loader, device, q, clean_thresh=CLEAN_PIXEL_THRESH):
    model.eval()
    out = []
    with torch.no_grad():
        for x, y_ref, target, target_cloud, _ in loader:
            x, y_ref, target = x.to(device), y_ref.to(device), target.to(device)
            delta, log_sigma = model(x)
            mu = (y_ref + delta).clamp(0, 1)
            sigma = torch.exp(log_sigma)
            lo, hi = mu - q * sigma, mu + q * sigma
            covered = (target >= lo) & (target <= hi)
            for b in range(x.shape[0]):
                mask = target_cloud[b].numpy() < clean_thresh
                if mask.sum() == 0:
                    continue
                cov_b = covered[b].cpu().numpy()[:, mask].mean() if mask.any() else np.nan
                width_b = (hi - lo)[b].cpu().numpy()[:, mask].mean() if mask.any() else np.nan
                cloud_near_mean = float(x[b, CLOUD_NEAR_IDX].mean().cpu())
                out.append({"coverage": float(cov_b), "width": float(width_b), "cloud_near": cloud_near_mean})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--test-roi", action="append", required=True)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", required=True)
    ap.add_argument("--q", type=float, required=True)
    ap.add_argument("--n-strata", type=int, default=4, help="кількість квартилей/груп за хмарністю near-кадру")
    ap.add_argument("--out", default="results/stratified_coverage")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = build_dataset(args.root, args.test_roi, args.img_size, "random_clean", 0.3, args.seed)
    loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    model = ResidualUNetUncertainty(in_ch=n_input_channels(include_s1=False), out_ch=n_bands()).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    recs = per_patch_coverage_width(model, loader, device, args.q)
    cloud = np.array([r["cloud_near"] for r in recs])
    cov = np.array([r["coverage"] for r in recs])
    width = np.array([r["width"] for r in recs])

    # provenance для сезону (місяць цільової дати)
    prov_ds = test_ds if hasattr(test_ds, "provenance") else test_ds.datasets[0]
    months = np.array([int(prov_ds.provenance(i)["target_date"].split("-")[1]) for i in range(len(recs))])

    overall_cov, overall_w = float(np.nanmean(cov)), float(np.nanmean(width))

    # Страта 1: квартилі хмарності near-кадру
    q_edges = np.nanpercentile(cloud, np.linspace(0, 100, args.n_strata + 1))
    cloud_bins = np.clip(np.digitize(cloud, q_edges[1:-1]), 0, args.n_strata - 1)

    lines = [
        "# Умовне покриття за стратами (прототип, results/uncertainty_final)",
        "",
        f"Test ROI: {', '.join(args.test_roi)} ({len(recs)} патчів). "
        f"Номінальне покриття: 90%. Конформний q={args.q}.",
        f"Загальне покриття (для довідки, має збігатись із report.md моделі): {overall_cov:.1%}, "
        f"середня ширина {overall_w:.4f}.",
        "",
        "## Страта 1: хмарність near-кадру (проксі складності умов)",
        "",
        "| Квартиль хмарності | n | Покриття | Середня ширина |",
        "|---|---|---|---|",
    ]
    for b in range(args.n_strata):
        sel = cloud_bins == b
        n = int(sel.sum())
        c = float(np.nanmean(cov[sel])) if n else float("nan")
        w = float(np.nanmean(width[sel])) if n else float("nan")
        lines.append(f"| Q{b+1} ({q_edges[b]:.3f}-{q_edges[b+1]:.3f}) | {n} | {c:.1%} | {w:.4f} |")

    lines += [
        "",
        "## Страта 2: місяць цільової дати (сезон)",
        "",
        "| Місяць | n | Покриття | Середня ширина |",
        "|---|---|---|---|",
    ]
    for m in sorted(set(months)):
        sel = months == m
        n = int(sel.sum())
        c = float(np.nanmean(cov[sel])) if n else float("nan")
        w = float(np.nanmean(width[sel])) if n else float("nan")
        lines.append(f"| {m:02d} | {n} | {c:.1%} | {w:.4f} |")

    min_strat_cov = min(float(np.nanmean(cov[cloud_bins == b])) for b in range(args.n_strata) if (cloud_bins == b).any())
    max_strat_cov = max(float(np.nanmean(cov[cloud_bins == b])) for b in range(args.n_strata) if (cloud_bins == b).any())
    lines += [
        "",
        f"**Розкид покриття між квартилями хмарності: {min_strat_cov:.1%} - {max_strat_cov:.1%}** "
        f"(розмах {(max_strat_cov - min_strat_cov)*100:.1f} п.п.). Якщо розмах суттєвий (>5-10 п.п.) - "
        f"агреговане покриття §4 (90.8%) ховає нерівномірність: калібрування добре для "
        f"одних умов і погано для інших, а 'умовна достовірність', заявлена в новизні "
        f"концепції, вимагає окремого q на страту (не реалізовано тут - прототип лише "
        f"ВИМІРЮЄ нерівномірність одним спільним q, подальший крок - страто-специфічне "
        f"калібрування, якщо розмах підтвердиться значущим).",
        "",
        "Застереження: 1 test ROI, 1 seed, страти - лише доступні проксі (хмарність, "
        "місяць), НЕ справжні агрономічні страти (культура/фаза/рельєф) з концепції - "
        "ці дані недоступні без SEN12MS lc-міток (заблоковано в цьому циклі робіт).",
    ]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nЗбережено: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
