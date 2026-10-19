#!/usr/bin/env python3
"""
П.4 з плану "зміна моделі/розрахунків для захисту" - НАЙДЕШЕВШИЙ крок (без нового
тренування): перегрупувати вже наявні результати абляції (results/step2_final,
results/phen_full, results/era5_phen_full, results/phen_grid_best) за тривалістю
розриву y_ref->ціль (днів без чистого спостереження), і перевірити, чи ефект
фенологічного члена концентрується саме в ДОВГИХ розривах - сценарій, під який
компонент розроблявся (концепція §2.4), а не розмазаний по всій вибірці, де
"у середньому" ефект тонув у шумі (§8.2-8.2.4 REPORT.md).

Дані test_metrics_raw.json зберігають per-патч метрики (RMSE/MAE/SSIM/SAM), але
НЕ довжину розриву - тому перебудовуємо test-набір з тим самим --root/--test-roi/
--seed (детерміновано, той самий порядок патчів) і дістаємо provenance() для
кожного патча, щоб зіставити довжину розриву з уже збереженими метриками.

Запуск:
    python stratify_by_gap.py --root data/raw --test-roi s2_america_test/ROIs1970/132 \
        --img-size 128 --seed 0 \
        --baseline results/step2_final/test_metrics_raw.json \
        --variant results/phen_full/test_metrics_raw.json --variant-label "phen (дні, вихідна)" \
        --variant results/era5_phen_full/test_metrics_raw.json --variant-label "phen (GDD)" \
        --variant results/phen_grid_best/test_metrics_raw.json --variant-label "phen (підібрана)" \
        --out results/stratify_by_gap
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train_step2 import build_dataset


def gap_days(root, test_roi, img_size, seed):
    ds = build_dataset(root, test_roi, img_size, "random_clean", 0.3, seed)
    prov_ds = ds if hasattr(ds, "provenance") else ds.datasets[0]
    from datetime import date
    gaps = []
    for i in range(len(ds)):
        p = prov_ds.provenance(i)
        d_ref = date.fromisoformat(p["ref_date"])
        d_tgt = date.fromisoformat(p["target_date"])
        gaps.append(abs((d_tgt - d_ref).days))
    return np.array(gaps)


def load_rmse(path):
    data = json.load(open(path))
    return np.array([r["model"]["rmse"] for r in data]), np.array([r["baseline_yref"]["rmse"] for r in data])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--test-roi", action="append", required=True)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", required=True, help="test_metrics_raw.json чистого L1 (step2_final)")
    ap.add_argument("--variant", action="append", required=True, help="test_metrics_raw.json варіанту для порівняння")
    ap.add_argument("--variant-label", action="append", required=True)
    ap.add_argument("--n-strata", type=int, default=3, help="скільки груп за довжиною розриву (терцилі за замовчуванням)")
    ap.add_argument("--out", default="results/stratify_by_gap")
    args = ap.parse_args()

    if len(args.variant) != len(args.variant_label):
        sys.exit("--variant і --variant-label мають бути в однаковій кількості й порядку")

    gaps = gap_days(args.root, args.test_roi, args.img_size, args.seed)
    base_model_rmse, base_ref_rmse = load_rmse(args.baseline)
    base_drmse = base_model_rmse - base_ref_rmse  # ΔRMSE чистого L1 проти y_ref, по патчу

    edges = np.percentile(gaps, np.linspace(0, 100, args.n_strata + 1))
    strata = np.clip(np.digitize(gaps, edges[1:-1]), 0, args.n_strata - 1)

    lines = [
        "# Стратифікований аналіз за тривалістю розриву y_ref->ціль (п.4 плану, без нового тренування)",
        "",
        f"Test ROI: {', '.join(args.test_roi)} ({len(gaps)} патчів). "
        f"Розрив поділено на {args.n_strata} терцилі за днями без чистого спостереження.",
        f"Межі терциль (дні): {', '.join(f'{e:.0f}' for e in edges)}",
        "",
    ]

    for b in range(args.n_strata):
        sel = strata == b
        n = int(sel.sum())
        lo, hi = edges[b], edges[b + 1]
        lines.append(f"## Терциль {b+1}: {lo:.0f}-{hi:.0f} днів розриву (n={n})")
        lines.append("")
        lines.append("| Варіант | ΔRMSE (проти y_ref) у цій терцилі | Довідково: ΔRMSE чистого L1 у цій терцилі |")
        lines.append("|---|---|---|")
        for vpath, vlabel in zip(args.variant, args.variant_label):
            v_model_rmse, v_ref_rmse = load_rmse(vpath)
            v_drmse = v_model_rmse - v_ref_rmse
            lines.append(
                f"| {vlabel} | {np.nanmean(v_drmse[sel]):+.4f} | {np.nanmean(base_drmse[sel]):+.4f} |"
            )
        lines.append("")

    # Пряме порівняння: чи фенологічний варіант наближається/перевершує чистий L1 САМЕ в довгій терцилі
    lines.append("## Чи звужується розрив (variant - baseline) у довгій терцилі порівняно з короткою?")
    lines.append("")
    lines.append("| Варіант | Δ(variant-baseline) коротка терциль | Δ(variant-baseline) довга терциль |")
    lines.append("|---|---|---|")
    sel_short = strata == 0
    sel_long = strata == args.n_strata - 1
    for vpath, vlabel in zip(args.variant, args.variant_label):
        v_model_rmse, v_ref_rmse = load_rmse(vpath)
        v_drmse = v_model_rmse - v_ref_rmse
        diff_short = float(np.nanmean((v_drmse - base_drmse)[sel_short]))
        diff_long = float(np.nanmean((v_drmse - base_drmse)[sel_long]))
        lines.append(f"| {vlabel} | {diff_short:+.4f} | {diff_long:+.4f} |")

    lines += [
        "",
        "Інтерпретація: `variant - baseline` - наскільки гірше (додатне) чи краще (від'ємне) "
        "фенологічний варіант за чистий L1 У ЦІЙ терцилі. Якщо це число ближче до нуля "
        "(чи від'ємне) у довгій терцилі, ніж у короткій - це свідчить, що фенологічний член "
        "дійсно концентрує корисність саме в довгих розривах, як і задумано (§2.4 концепції), "
        "навіть якщо в середньому по всій вибірці (§8.2 REPORT.md) ефект губиться в шумі. "
        "Якщо різниці немає чи вона в інший бік - це ще один чесний аргумент НЕ рятувати "
        "компонент цим шляхом, а розглядати п.1-3 плану.",
        "",
        "Застереження: 1 test ROI, 1 seed на варіант (крім підібраної конфігурації, де є 5 seed "
        "в REPORT.md §8.2.4, тут використано лише seed=0) - це розвідувальний аналіз наявних "
        "даних, не новий дисертаційний результат сам по собі.",
    ]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nЗбережено: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
