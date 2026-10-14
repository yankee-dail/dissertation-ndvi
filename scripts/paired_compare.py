#!/usr/bin/env python3
"""
Пряме парне порівняння двох навчених чекпоінтів train_step2.py (а не кожного окремо
проти y_ref-базового рівня). Обидва чекпоінти оцінюються на ОДНОМУ й тому самому
test-наборі (той самий --test-roi, img-size, seed -> та сама вибірка цілей), тож
bootstrap_diff_ci тут - це парний бутстреп по РІЗНИЦІ прогнозів моделей на кожному
патчі, що дає вужчий і коректніший CI, ніж порівняння двох незалежних "модель vs
базовий рівень" результатів.

Вимога: обидва чекпоінти мають бути навчені з ОДНАКОВИМ --seed (впливає на вибір
target у test-наборі через random_clean) і однаковим --test-roi/--img-size/--use-s1,
інакше патчі/цілі не збігаються і порівняння нечесне.

Запуск:
    python paired_compare.py --root data/raw --test-roi s2_america_test/ROIs1970/132 \
        --img-size 128 --seed 0 \
        --model-a results/phen_grid_best/model.pt --label-a "phen w=0.1 asym=2" \
        --model-b results/step2_final/model.pt --label-b "pure L1"
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from sen12mscrts_dataset import n_bands, n_input_channels
from model import ResidualUNet
from eval_protocol import all_metrics, bootstrap_diff_ci, CLEAN_PIXEL_THRESH
from train_step2 import build_dataset


def load_model(path, device, in_ch, out_ch):
    model = ResidualUNet(in_ch=in_ch, out_ch=out_ch).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--test-roi", action="append", required=True)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0,
                     help="МАЄ збігатись із --seed обох порівнюваних тренувань (впливає на вибір target)")
    ap.add_argument("--use-s1", action="store_true")
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--model-b", required=True)
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None, help="якщо задано - записати звіт у цей файл")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = build_dataset(args.root, args.test_roi, args.img_size, "random_clean", 0.3,
                             args.seed, use_s1=args.use_s1, era5_root=None)
    loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    in_ch = n_input_channels(include_s1=args.use_s1)
    out_ch = n_bands()
    model_a = load_model(args.model_a, device, in_ch, out_ch)
    model_b = load_model(args.model_b, device, in_ch, out_ch)

    rmse_a, rmse_b, sam_a, sam_b = [], [], [], []
    with torch.no_grad():
        for x, y_ref, target, target_cloud, _ in loader:
            x, y_ref, target = x.to(device), y_ref.to(device), target.to(device)
            pred_a = (y_ref + model_a(x)).clamp(0, 1)
            pred_b = (y_ref + model_b(x)).clamp(0, 1)
            for b in range(x.shape[0]):
                mask = target_cloud[b].numpy() < CLEAN_PIXEL_THRESH
                t = target[b].cpu().numpy()
                ma = all_metrics(pred_a[b].cpu().numpy(), t, mask=mask)
                mb = all_metrics(pred_b[b].cpu().numpy(), t, mask=mask)
                rmse_a.append(ma["rmse"]); rmse_b.append(mb["rmse"])
                sam_a.append(ma["sam_deg"]); sam_b.append(mb["sam_deg"])

    d_rmse, lo_rmse, hi_rmse = bootstrap_diff_ci(rmse_a, rmse_b, n_boot=args.n_boot, seed=args.seed)
    d_sam, lo_sam, hi_sam = bootstrap_diff_ci(sam_a, sam_b, n_boot=args.n_boot, seed=args.seed)

    def sig(lo, hi):
        if hi < 0:
            return "A значуще краще за B"
        if lo > 0:
            return "A значуще гірше за B"
        return "не значуще (CI перетинає 0, A і B статистично нерозрізнимі)"

    lines = [
        f"# Пряме парне порівняння: {args.label_a} (A) проти {args.label_b} (B)",
        "",
        f"Той самий test-набір ({', '.join(args.test_roi)}, img-size={args.img_size}, seed={args.seed}) "
        f"для обох моделей - парний бутстреп по РІЗНИЦІ прогнозів на кожному патчі (n={len(rmse_a)}).",
        "",
        f"A = {args.model_a}",
        f"B = {args.model_b}",
        "",
        f"Δ RMSE (A - B) = {d_rmse:+.4f} [{lo_rmse:+.4f}, {hi_rmse:+.4f}] - {sig(lo_rmse, hi_rmse)}",
        f"Δ SAM  (A - B) = {d_sam:+.2f}° [{lo_sam:+.2f}, {hi_sam:+.2f}] - {sig(lo_sam, hi_sam)}",
    ]
    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report + "\n")
        print(f"\nЗбережено: {args.out}")


if __name__ == "__main__":
    main()
