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
from sen12mscrts_dataset import (SEN12MSCRTSPatchDataset, n_bands, n_input_channels,
                                  read_tif_decimated, simple_cloud_prob, REFL_MAX)
from model import ResidualUNet
from eval_protocol import all_metrics, bootstrap_diff_ci, CLEAN_PIXEL_THRESH

RED_IDX, NIR_IDX = 3, 7  # B4, B8 у 13-канальному стеку SEN12MS-CR-TS


def ndvi(img):
    """img: (..., C, H, W) з C=13 (нормалізоване відбиття [0,1] або сире - формула інваріантна
    до масштабу, бо це відношення). Повертає (..., H, W)."""
    red, nir = img[..., RED_IDX, :, :], img[..., NIR_IDX, :, :]
    return (nir - red) / (nir + red + 1e-6)


def compute_ndvi_rate_bounds(root, specs, lo_pct=5, hi_pct=95, patch_stride=5, clean_thresh=0.2,
                              era5_root=None):
    """Мінімальна версія ідеї Zhang (порогове визначення фенологічної динаміки з власного
    часового ряду, без фітингу логістичної кривої): на підвибірці патчів з train ROI рахує
    швидкість зміни NDVI між послідовними ЧИСТИМИ (хмарність < clean_thresh) спостереженнями,
    повертає lo_pct/hi_pct перцентилі - емпіричну "смугу правдоподібності" для динаміки NDVI
    (§фенологія). Без era5_root - швидкість у NDVI/день; з era5_root - у NDVI/GDD (точніше,
    бо фенофаза залежить від суми ефективних температур, не календарного часу) - МАЄ бути
    тим самим вибором одиниць, що й у phenological_penalty, інакше масштаб штрафу зламається."""
    from datetime import datetime

    era5_luts = {}
    if era5_root is not None:
        from era5_lookup import Era5Lookup, resolve_era5_path
        for spec in specs:
            era5_luts[spec] = Era5Lookup(resolve_era5_path(spec, era5_root))

    rates = []
    for spec in specs:
        r, roi_dir, roi = resolve_roi_spec(root, spec)
        n = count_patches(r, roi_dir, roi)
        s2_dirs = [Path(r) / roi_dir / roi / "S2" / str(tdx) for tdx in range(30)]
        series_per_tdx = [sorted(f for f in d.iterdir() if f.suffix == ".tif") for d in s2_dirs]
        lut = era5_luts.get(spec)
        for pdx in range(0, n, patch_stride):
            obs = []
            for tdx in range(30):
                if pdx >= len(series_per_tdx[tdx]):
                    continue
                f = series_per_tdx[tdx][pdx]
                img = read_tif_decimated(f, out_hw=16)
                frac = float(simple_cloud_prob(img).mean())
                if frac >= clean_thresh:
                    continue
                dt_obj = datetime.strptime(f.name.split("_")[5], "%Y-%m-%d")
                x_val = lut(dt_obj.date())[0] if lut is not None else (dt_obj - datetime(2014, 4, 3)).days
                n_val = float(np.clip(ndvi(np.clip(img, 0, REFL_MAX) / REFL_MAX), -1, 1).mean())
                obs.append((x_val, n_val))
            obs.sort()
            for i in range(1, len(obs)):
                dx = obs[i][0] - obs[i - 1][0]
                if dx > 0:
                    rates.append((obs[i][1] - obs[i - 1][1]) / dx)

    if len(rates) < 20:
        print(f"[warn] лише {len(rates)} оцінок швидкості NDVI - пороги можуть бути нестабільними")
    rate_lo, rate_hi = float(np.percentile(rates, lo_pct)), float(np.percentile(rates, hi_pct))
    unit = "NDVI/GDD" if era5_root is not None else "NDVI/день"
    print(f"Пороги динаміки NDVI (з {len(rates)} пар чистих спостережень train ROI): "
          f"[{rate_lo:.4f}, {rate_hi:.4f}] {unit} (p{lo_pct}/p{hi_pct})")
    return rate_lo, rate_hi


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


def resolve_s1_root(s2_root):
    """root/s2_africa_test -> root/s1_africa_test (сусідня директорія, той самий датасет,
    інша модальність) - підходить лише коли останній сегмент шляху починається з 's2_'."""
    name = s2_root.name
    if not name.startswith("s2_"):
        raise ValueError(f"не можу вивести S1-шлях з '{s2_root}' - очікував сегмент, що починається з 's2_'")
    return s2_root.parent / ("s1_" + name[len("s2_"):])


def build_dataset(root, specs, img_size, target_mode, target_max_cloud, seed, max_patches_per_roi=None,
                   use_s1=False, era5_root=None, ref_mode="prior_clean", ref_cache_dir=None,
                   near_mode="nearest", samples_per_patch=1, ref_select="clearest"):
    """max_patches_per_roi дозволяє зрівняти ЗАГАЛЬНИЙ обсяг даних між прогонами з різною
    кількістю ROI (контроль обсягу при перевірці ефекту різноманітності локацій) - бере
    перші N патчів з кожного переліченого ROI замість усіх. use_s1 додає VV/VH з сусідньої
    s1_*-директорії як додаткові вхідні канали (див. resolve_s1_root). era5_root вмикає
    GDD-масштабування фенологічного штрафу замість днів (див. era5_lookup.py)."""
    from torch.utils.data import ConcatDataset
    from era5_lookup import resolve_era5_path
    parts = []
    for spec in specs:
        r, roi_dir, roi = resolve_roi_spec(root, spec)
        n = count_patches(r, roi_dir, roi)
        if max_patches_per_roi is not None:
            n = min(n, max_patches_per_roi)
        s1_root = resolve_s1_root(r) if use_s1 else None
        era5_path = resolve_era5_path(spec, era5_root) if era5_root else None
        ds = SEN12MSCRTSPatchDataset(root=r, roi_dir=roi_dir, roi=roi, patch_indices=range(n),
                                      img_size=img_size, target_mode=target_mode,
                                      target_max_cloud=target_max_cloud, seed=seed, s1_root=s1_root,
                                      era5_path=era5_path, ref_mode=ref_mode, ref_cache_dir=ref_cache_dir,
                                      near_mode=near_mode, samples_per_patch=samples_per_patch,
                                      ref_select=ref_select)
        tags = ("+S1" if use_s1 else "") + ("+ERA5" if era5_root else "") + ("+yref=interp" if ref_mode == "linear_interp" else "") + ("+causal_near" if near_mode == "prior" else "")
        print(f"  {spec}: {n} патчів x {samples_per_patch} дат" + (f" ({tags})" if tags else ""))
        parts.append(ds)
    return ConcatDataset(parts) if len(parts) > 1 else parts[0]


def evaluate(model, loader, device, clean_thresh=CLEAN_PIXEL_THRESH):
    """Повертає список словників per-sample метрик для моделі й для базового рівня y_ref.

    RMSE/MAE/SAM рахуються лише по пікселях цілі з cloud < clean_thresh (SSIM - без
    маски, див. eval_protocol.ssim_score)."""
    model.eval()
    results = []
    with torch.no_grad():
        for x, y_ref, target, target_cloud, _ in loader:
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
    ap.add_argument("--use-s1", action="store_true",
                     help="додати VV/VH (Sentinel-1) ref і near кадрів як вхідні канали; "
                          "потребує сусідньої директорії s1_<регіон> з тією самою структурою")
    ap.add_argument("--phen-weight", type=float, default=0.0,
                     help="вага фенологічного члена втрат (0 = вимкнено, L1 як раніше). "
                          "Пороги швидкості NDVI оцінюються з train ROI автоматично (Zhang-поріг, "
                          "мінімальна версія - без фітингу логістичної кривої чи зовнішніх даних)")
    ap.add_argument("--phen-asym-ratio", type=float, default=5.0,
                     help="у скільки разів сильніше штрафується перевищення верхньої межі "
                          "(неправдоподібне позеленіння), ніж падіння нижче нижньої (стрес)")
    ap.add_argument("--era5-root", default=None,
                     help="директорія з <регіон>_<roi>.nc від fetch_era5.py - вмикає GDD-"
                          "масштабування фенологічного штрафу замість днів")
    ap.add_argument("--phen-ramp-epochs", type=int, default=0,
                     help="якщо >0, вага фенологічного штрафу лінійно зростає від 0 до "
                          "--phen-weight за перші N епох (замість повної ваги з 1-ї епохи) - "
                          "щоб не заважати ранній стабілізації L1 на старті тренування")
    ap.add_argument("--phen-threshold-roi", action="append", default=None,
                     help="ROI (можна кілька разів) для оцінки порогів швидкості NDVI "
                          "(compute_ndvi_rate_bounds) ОКРЕМО від --train-roi - ширший набір "
                          "дає стійкіші пороги; якщо не задано, використовується --train-roi "
                          "(попередня поведінка)")
    ap.add_argument("--ref-mode", default="prior_clean", choices=["prior_clean", "linear_interp"],
                     help="опорне значення y_ref у ŷ = y_ref + Δθ: останнє чисте попереднє спостереження "
                          "або лінійна інтерполяція між чистими пікселями до і після цілі (формула 2.11)")
    ap.add_argument("--ref-cache-dir", default=None,
                     help="кеш інтерпольованих y_ref (.npy) - щоб не читати 30 кадрів на кожній епосі")
    ap.add_argument("--near-mode", default="nearest", choices=["nearest", "prior"],
                     help="x_near: найближчий кадр (може бути ПІСЛЯ цілі) або лише з кадрів ДО цілі "
                          "(причинний режим для раннього виявлення; разом із --ref-mode prior_clean)")
    ap.add_argument("--ref-select", default="clearest", choices=["clearest", "latest_clean"],
                     help="y_ref серед попередніх: найчистіший (історично) або найпізніший чистий (≤10%% хмар)")
    ap.add_argument("--samples-per-patch", type=int, default=1,
                     help="скільки різних цільових дат брати з кожного фрагмента (для щільних L2A-рядів)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/step2_prototype")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    all_specs = set(args.train_roi) | set(args.val_roi) | set(args.test_roi)
    if len(all_specs) != len(args.train_roi) + len(args.val_roi) + len(args.test_roi):
        sys.exit("train-roi / val-roi / test-roi перетинаються - вибір епохи й фінальний звіт будуть нечесними")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Пристрій: {device}")

    print("Train ROI:")
    train_ds = build_dataset(args.root, args.train_roi, args.img_size, args.target_mode,
                              args.target_max_cloud, args.seed, args.max_train_patches_per_roi,
                              use_s1=args.use_s1, era5_root=args.era5_root,
                              ref_mode=args.ref_mode, ref_cache_dir=args.ref_cache_dir,
                              near_mode=args.near_mode, samples_per_patch=args.samples_per_patch,
                              ref_select=args.ref_select)
    print("Val ROI (лише для вибору епохи, інша локація):")
    val_ds = build_dataset(args.root, args.val_roi, args.img_size, args.target_mode,
                            args.target_max_cloud, args.seed, use_s1=args.use_s1, era5_root=args.era5_root,
                              ref_mode=args.ref_mode, ref_cache_dir=args.ref_cache_dir,
                              near_mode=args.near_mode, samples_per_patch=args.samples_per_patch,
                              ref_select=args.ref_select)
    print("Test ROI (лише для фінального звіту, ще інша локація):")
    test_ds = build_dataset(args.root, args.test_roi, args.img_size, args.target_mode,
                             args.target_max_cloud, args.seed, use_s1=args.use_s1, era5_root=args.era5_root,
                              ref_mode=args.ref_mode, ref_cache_dir=args.ref_cache_dir,
                              near_mode=args.near_mode, samples_per_patch=args.samples_per_patch,
                              ref_select=args.ref_select)
    print(f"Разом: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    rate_lo = rate_hi = None
    if args.phen_weight > 0:
        threshold_specs = args.phen_threshold_roi if args.phen_threshold_roi else args.train_roi
        rate_lo, rate_hi = compute_ndvi_rate_bounds(args.root, threshold_specs, era5_root=args.era5_root)

    model = ResidualUNet(in_ch=n_input_channels(include_s1=args.use_s1), out_ch=n_bands()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    def phenological_penalty(pred, y_ref, ref_days):
        """Δ NDVI (модель - y_ref) за межами емпіричної смуги правдоподібності, масштабованої
        на РЕАЛЬНИЙ проміжок y_ref->ціль (ref_days - з Dataset, НЕ плутати з каналом day_gap
        у x, який кодує near->ціль, інший інтервал; одиниці - дні або GDD, залежно від
        --era5-root, мають бути узгоджені з rate_lo/rate_hi з compute_ndvi_rate_bounds). Перевищення
        верхньої межі (неправдоподібне позеленіння) карається w_up разів сильніше за падіння
        нижче нижньої (правдоподібний стрес) - асиметрія з концепції Методу 1."""
        gap = ref_days.to(pred.device).clamp(min=1.0).view(-1, 1, 1)  # (B,1,1), мін. 1 одиниця (день/GDD)
        # pred тут НЕ clamp-нутий (на відміну від evaluate()) - на старті тренування
        # необмежена модель може дати відбиття поза [0,1] чи близько нуля в обох каналах,
        # що робить ділення NDVI нестабільним (вибух градієнта). Для самого штрафу
        # клампимо лише локально - основний L1-лосс далі рахується на нестиснутому pred.
        delta = ndvi(pred.clamp(0, 1)) - ndvi(y_ref)  # (B, H, W)
        allowed_hi = rate_hi * gap
        allowed_lo = rate_lo * gap
        over = torch.relu(delta - allowed_hi)
        under = torch.relu(allowed_lo - delta)
        return args.phen_asym_ratio * over.mean() + under.mean()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val, best_epoch, best_state = float("inf"), None, None

    for epoch in range(1, args.epochs + 1):
        if args.phen_ramp_epochs > 0:
            epoch_phen_weight = args.phen_weight * min(1.0, epoch / args.phen_ramp_epochs)
        else:
            epoch_phen_weight = args.phen_weight

        model.train()
        train_losses = []
        for x, y_ref, target, _, ref_days in train_loader:
            x, y_ref, target = x.to(device), y_ref.to(device), target.to(device)
            delta = model(x)
            pred = y_ref + delta
            loss = criterion(pred, target)
            if epoch_phen_weight > 0:
                loss = loss + epoch_phen_weight * phenological_penalty(pred, y_ref, ref_days)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, baseline_losses = [], []
        with torch.no_grad():
            for x, y_ref, target, _, _ in val_loader:
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
        f"Вхід включає Sentinel-1 (VV/VH): {'так' if args.use_s1 else 'ні'}.",
        f"x_near: {'лише кадри до цілі (причинний режим)' if args.near_mode == 'prior' else 'найближчий кадр (може бути після цілі)'}.",
        f"Вибір y_ref серед попередніх: {args.ref_select}.",
        f"Опорне значення y_ref: {'лінійна інтерполяція між чистими пікселями до і після цілі' if args.ref_mode == 'linear_interp' else 'останнє чисте попереднє спостереження'}.",
        f"Фенологічний член втрат: {'так, вага ' + str(args.phen_weight) + f', межі [{rate_lo:.4f},{rate_hi:.4f}] NDVI/' + ('GDD' if args.era5_root else 'день') + ', асиметрія x' + str(args.phen_asym_ratio) + (f', лінійний розігрів ваги за {args.phen_ramp_epochs} епох' if args.phen_ramp_epochs > 0 else '') + (f', пороги оцінено з {len(args.phen_threshold_roi)} окремих threshold-ROI' if args.phen_threshold_roi else '') if args.phen_weight > 0 else 'ні'}.",
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
