#!/usr/bin/env python3
"""
Прототип інформаційної моделі стану поля (§11 REPORT.md) - PROV-подібний ланцюг
походження "виміряне -> відновлене -> достовірність -> рекомендація" поверх уже
навчених чекпоінтів results/step2_final (реконструкція) і results/uncertainty_final
(оцінка невизначеності). Не тренує нічого нового - лише об'єднує існуючі моделі в
один трасований запис на патч, і додає третій, досі не інтегрований сигнал:
фенологічну правдоподібність динаміки NDVI як ПОСТФАКТУМ перевірку (не як штраф у
лосі, як у §8.2 - тут вона працює як анотація "чи варто довіряти цьому відновленню",
ідея з обговорення "якщо фенологічний член не допомагає як loss-term, спробуйте
постобробку").

Схема запису (по одному на тестовий патч), PROV-термінологія (entity/activity):
  entity:measured       - ref-кадр (останнє чисте спостереження) + його походження
                           (файл, дата, ROI)
  activity:reconstruction - модель step2_final, вхід measured -> entity:reconstructed
  entity:reconstructed  - прогноз (y_ref + Δθ), НЕ порівнюється тут із target (ціль
                           невідома в продакшн-сценарії; на test ROI target є лише
                           для sanity-перевірки нижче, окремо)
  activity:calibration  - модель uncertainty_final + конформний q, вхід measured ->
                           entity:confidence
  entity:confidence     - μ, σ, [lo,hi] інтервал (конформно калібрований), ширина
  activity:recommendation - правило з §11 (нижче), вхід reconstructed+confidence ->
                           entity:recommendation
  entity:recommendation - прапорець {trust, review_dynamics, low_confidence} +
                           причина

Правило рекомендації (прозоре, не чорна скринька):
  - якщо швидкість зміни NDVI (reconstructed відносно measured, масштабована на
    реальний проміжок ref->target) ВИХОДИТЬ за емпіричну смугу правдоподібності
    (ту саму, що й у §8.2, compute_ndvi_rate_bounds) -> "review_dynamics"
    (неправдоподібна динаміка - можливий артефакт реконструкції АБО реальний
    агрономічний стрес/стрибок, модель лише флагує, не вирішує)
  - інакше, якщо ширина конформного інтервалу перевищує поріг --low-conf-cut
    (p90 ширин на КАЛІБРУВАЛЬНОМУ ROI) -> "low_confidence" (модель сама невпевнена)
  - інакше -> "trust"

Це прототип на 1 test ROI (та сама методологічна слабкість, що й у §4: калібрування
по одному ROI на роль, не по стратах) - перший експериментальний доказ, а не
дисертаційний результат.

Запуск:
    python information_model.py --root data/raw \
        --test-roi s2_america_test/ROIs1970/132 \
        --train-roi s2_america_test/ROIs1158/106 --train-roi s2_africa_test/ROIs2017/32 \
        --train-roi s2_africa_test/ROIs2017/140 --train-roi s2_asiaEast_test/ROIs1868/73 \
        --step2-model results/step2_final/model.pt \
        --uncertainty-model results/uncertainty_final/model.pt --q 1.238 --low-conf-cut 0.2502 \
        --out results/information_model
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from sen12mscrts_dataset import n_bands, n_input_channels
from model import ResidualUNet, ResidualUNetUncertainty
from eval_protocol import all_metrics, CLEAN_PIXEL_THRESH
from train_step2 import build_dataset, compute_ndvi_rate_bounds, ndvi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--test-roi", action="append", required=True)
    ap.add_argument("--train-roi", action="append", required=True,
                     help="лише для оцінки фенологічної смуги правдоподібності (compute_ndvi_rate_bounds)")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0,
                     help="МАЄ збігатись із seed, яким навчено --step2-model/--uncertainty-model")
    ap.add_argument("--step2-model", required=True)
    ap.add_argument("--uncertainty-model", required=True)
    ap.add_argument("--q", type=float, required=True,
                     help="конформний множник для пари (μ кроку 2, σ методу 2): q_l1_pair з "
                          "results/validation_extras/*/summary.json, НЕ q з report.md σ-моделі")
    ap.add_argument("--low-conf-cut", type=float, required=True,
                     help="поріг ширини інтервалу на патч, визначений на КАЛІБРУВАЛЬНОМУ ROI "
                          "(low_conf_cut_from_calib з validation_extras), не на тесті")
    ap.add_argument("--out", default="results/information_model")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rate_lo, rate_hi = compute_ndvi_rate_bounds(args.root, args.train_roi)

    test_ds = build_dataset(args.root, args.test_roi, args.img_size, "random_clean", 0.3, args.seed)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    in_ch, out_ch = n_input_channels(include_s1=False), n_bands()
    step2 = ResidualUNet(in_ch=in_ch, out_ch=out_ch).to(device)
    step2.load_state_dict(torch.load(args.step2_model, map_location=device))
    step2.eval()
    unc = ResidualUNetUncertainty(in_ch=in_ch, out_ch=out_ch).to(device)
    unc.load_state_dict(torch.load(args.uncertainty_model, map_location=device))
    unc.eval()

    # provenance() iterates per-underlying-dataset index; для одного test ROI (найчастіший
    # випадок) test_ds сам і є SEN12MSCRTSPatchDataset. Для кількох ROI (ConcatDataset)
    # розгортаємо вручну по частинах - тут обмежуємось 1 ROI (з документації прототипу).
    if hasattr(test_ds, "provenance"):
        prov_ds = test_ds
    else:
        if len(test_ds.datasets) != 1:
            sys.exit("прототип інформаційної моделі підтримує лише 1 --test-roi (provenance по кількох ROI не реалізовано)")
        prov_ds = test_ds.datasets[0]

    records = []
    widths = []
    with torch.no_grad():
        for i, (x, y_ref, target, target_cloud, ref_target_scale) in enumerate(loader):
            x, y_ref, target = x.to(device), y_ref.to(device), target.to(device)
            gap = ref_target_scale.to(device).clamp(min=1.0).view(-1, 1, 1)

            recon = (y_ref + step2(x)).clamp(0, 1)
            _, log_sigma = unc(x)
            sigma = torch.exp(log_sigma)
            # Інтервал центровано на значенні, яке видається користувачу (recon), а q
            # відкалібровано саме для score |y - recon| / σ - інакше гарантія (2.18) не
            # стосується відновленого значення.
            lo, hi = (recon - args.q * sigma).clamp(0, 1), (recon + args.q * sigma).clamp(0, 1)
            width = float((hi - lo).mean().item())
            widths.append(width)

            ndvi_rate = float(((ndvi(recon) - ndvi(y_ref)) / gap).mean().item())
            dynamics_ok = rate_lo <= ndvi_rate <= rate_hi

            mask = target_cloud[0].numpy() < CLEAN_PIXEL_THRESH
            sanity = all_metrics(recon[0].cpu().numpy(), target[0].cpu().numpy(), mask=mask)

            prov = prov_ds.provenance(i)
            records.append({
                "entity:measured": {"roi": prov["roi"], "patch_idx": prov["patch_idx"],
                                     "file": prov["ref_file"], "date": prov["ref_date"]},
                "activity:reconstruction": {"model": str(args.step2_model)},
                "entity:reconstructed": {"mean_value": float(recon.mean().item())},
                "activity:calibration": {"model": str(args.uncertainty_model), "conformal_q": args.q},
                "entity:confidence": {"sigma_mean": float(sigma.mean().item()), "interval_width": width},
                "entity:recommendation": None,  # заповнюється нижче 2-м проходом (потрібна медіана width)
                "_ndvi_rate": ndvi_rate, "_dynamics_ok": dynamics_ok,
                "_sanity_vs_target": sanity,  # ЛИШЕ для перевірки прототипу - у проді target невідомий
            })

    width_cutoff = args.low_conf_cut
    for r in records:
        if not r["_dynamics_ok"]:
            flag, reason = "review_dynamics", (
                f"швидкість NDVI {r['_ndvi_rate']:.4f}/день поза межами "
                f"[{rate_lo:.4f},{rate_hi:.4f}] - неправдоподібна динаміка")
        elif r["entity:confidence"]["interval_width"] > width_cutoff:
            flag, reason = "low_confidence", (
                f"ширина інтервалу {r['entity:confidence']['interval_width']:.4f} > "
                f"поріг з калібрувального ROI ({width_cutoff:.4f}) - модель сама невпевнена")
        else:
            flag, reason = "trust", "динаміка в межах очікуваного, впевненість достатня"
        r["entity:recommendation"] = {"flag": flag, "reason": reason}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "records.jsonl", "w") as f:
        for r in records:
            clean = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    flags = [r["entity:recommendation"]["flag"] for r in records]
    counts = {flag: flags.count(flag) for flag in ("trust", "review_dynamics", "low_confidence")}

    # Санітарна перевірка прототипу (НЕ частина продакшн-рекомендації): чи патчі з
    # прапорцем review_dynamics/low_confidence дійсно мають вищу РЕАЛЬНУ похибку
    # відносно target, ніж "trust"? Якщо так - рекомендація несе реальну інформацію,
    # а не шум.
    trust_rmse = np.array([r["_sanity_vs_target"]["rmse"] for r in records if r["entity:recommendation"]["flag"] == "trust"])
    flagged_rmse = np.array([r["_sanity_vs_target"]["rmse"] for r in records if r["entity:recommendation"]["flag"] != "trust"])
    n_nan_target = int(np.isnan([r["_sanity_vs_target"]["rmse"] for r in records]).sum())

    report = [
        "# Інформаційна модель стану поля (прототип, §11)",
        "",
        f"Test ROI: {', '.join(args.test_roi)} ({len(records)} патчів). "
        f"Фенологічна смуга правдоподібності (з train ROI, день-масштаб): [{rate_lo:.4f},{rate_hi:.4f}] NDVI/день.",
        f"Конформний множник q = {args.q} (пара μ кроку 2 + σ методу 2, калібрувальний ROI). "
        f"Поріг low_confidence = ширина інтервалу > {width_cutoff:.4f} (p90 на калібрувальному ROI).",
        "",
        "## Розподіл рекомендацій",
        "",
        f"- trust: {counts['trust']} ({100*counts['trust']/len(records):.1f}%)",
        f"- review_dynamics: {counts['review_dynamics']} ({100*counts['review_dynamics']/len(records):.1f}%)",
        f"- low_confidence: {counts['low_confidence']} ({100*counts['low_confidence']/len(records):.1f}%)",
        "",
        "## Санітарна перевірка: чи прапорці несуть реальну інформацію про похибку?",
        "",
        f"RMSE відносно target (ground truth, доступний лише в цьому прототипі на test ROI, "
        f"НЕ використовується для самої рекомендації). {n_nan_target} патч(ів) мають повністю "
        f"хмарну ціль (RMSE невизначений, виключено через nanmean/nanstd):",
        f"- trust: {np.nanmean(trust_rmse):.4f}±{np.nanstd(trust_rmse):.4f} (n={len(trust_rmse)})"
        if len(trust_rmse) else "- trust: немає патчів з цим прапорцем",
        f"- review_dynamics/low_confidence: {np.nanmean(flagged_rmse):.4f}±{np.nanstd(flagged_rmse):.4f} (n={len(flagged_rmse)})"
        if len(flagged_rmse) else "- flagged: немає патчів з цими прапорцями",
        "",
        "Якщо середній RMSE flagged-групи помітно вищий за trust - прапорці несуть реальний "
        "сигнал про якість реконструкції, а не шум (перша перевірка корисності цієї моделі, "
        "не дисертаційний доказ - 1 test ROI, 1 seed).",
    ]
    (out_dir / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nЗбережено: {out_dir / 'records.jsonl'}, {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
