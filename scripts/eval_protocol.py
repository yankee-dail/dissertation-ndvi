#!/usr/bin/env python3
"""
Крок 1: чесний протокол оцінювання реконструкції оптичних індексів за хмарності.

Формат вхідних даних (один .npz на семпл, директорія --data):
    s2          float32 (T, C, H, W)  -- часовий ряд кадрів Sentinel-2 (вхід + ціль разом)
    cloud       float32 (T, H, W)     -- хмарність [0,1] на піксель для кожного кадру
    target      float32 (C, H, W)     -- "чистий" кадр-ціль (копія одного з кадрів s2)
    t_days      int32   (T,)          -- день (напр., відносно запуску S1) для кожного кадру s2
    t_target    int32   ()            -- день кадру-цілі
    target_idx  int32   ()            -- індекс кадру-цілі в s2 (для детекції витоку); -1, якщо невідомо

Методи:
    least_cloudy_raw    - найменш хмарний кадр з s2 як прогноз; ціль МОЖЕ бути серед кандидатів (як у наївному
                          базовому методі з літератури) -> ризик завищення.
    least_cloudy_strict - те саме, але кадр з індексом target_idx (і кадри з тим самим днем) виключено з кандидатів.
    linear_interp       - потрапіксельна лінійна інтер-/екстраполяція в часі за "чистими" пікселями
                          (cloud < --clean-thresh), кадр-ціль завжди виключено.

Кожен семпл позначається risk_of_leak=True, якщо:
    - найменш хмарний кандидат (без виключень) співпадає з target_idx, АБО
    - його хмарність <= --eps-cloud і день співпадає з t_target у межах --tol-same-days.

Результат подається у двох розрізах: "full" (усі семпли, як у наївному порівнянні з літератури)
і "leak_free" (лише семпли без ризику витоку) -- щоб показати, наскільки базовий метод був завищений.

Самоперевірка: python eval_protocol.py --selftest
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from skimage.metrics import structural_similarity as _ssim
except ImportError:
    _ssim = None

# --- пороги, які впливають на детекцію витоку й інтерполяцію ---
EPS_CLOUD = 0.01       # хмарність <= цього значення вважається "безхмарним" кадром
TOL_SAME_DAYS = 1      # кадр вважається тим самим днем, що й ціль, якщо різниця днів <= цього
CLEAN_PIXEL_THRESH = 0.2  # піксель вважається "чистим" для лінійної інтерполяції, якщо cloud < цього


# --------------------------- метрики ---------------------------

def _apply_mask(pred, target, mask):
    """mask: (H, W) boolean, True = чистий (валідний) піксель цілі. None = без маскування.

    Повертає сплощені (C, N) масиви лише з валідних пікселів. Якщо після маскування
    валідних пікселів немає (увесь патч хмарний), повертає None, None - викликач має
    пропустити такий семпл, а не рахувати метрику по хмарі.
    """
    if mask is None:
        return pred.reshape(pred.shape[0], -1), target.reshape(target.shape[0], -1)
    flat_mask = mask.reshape(-1)
    if not np.any(flat_mask):
        return None, None
    p = pred.reshape(pred.shape[0], -1)[:, flat_mask]
    t = target.reshape(target.shape[0], -1)[:, flat_mask]
    return p, t


def rmse(pred, target, mask=None):
    p, t = _apply_mask(pred, target, mask)
    if p is None:
        return float("nan")
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(pred, target, mask=None):
    p, t = _apply_mask(pred, target, mask)
    if p is None:
        return float("nan")
    return float(np.mean(np.abs(p - t)))


def sam_degrees(pred, target, mask=None, eps=1e-8):
    """Spectral Angle Mapper, у градусах, усереднений по (валідних) пікселях. pred/target: (C, H, W)."""
    p, t = _apply_mask(pred, target, mask)
    if p is None:
        return float("nan")
    dot = np.sum(p * t, axis=0)
    norm_p = np.linalg.norm(p, axis=0)
    norm_t = np.linalg.norm(t, axis=0)
    denom = norm_p * norm_t
    valid = denom > eps
    if not np.any(valid):
        return 0.0
    cos_angle = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
    angles = np.arccos(cos_angle)
    return float(np.degrees(np.mean(angles)))


def ssim_score(pred, target):
    """Усереднений по каналах SSIM. pred/target: (C, H, W), значення очікуються в [0,1].

    НЕ маскується: SSIM - віконна (не потрапіксельна) метрика, коректне маскування
    вимагало б вирізання довільної форми з ковзного вікна, що не підтримується
    бібліотекою напряму. Тому SSIM рахується по всьому зображенню, включно з
    хмарними пікселями цілі, якщо вони є, - це свідоме обмеження, не недогляд.
    """
    if _ssim is None:
        return float("nan")
    scores = []
    for c in range(pred.shape[0]):
        a, b = pred[c], target[c]
        rng = float(b.max() - b.min())
        if rng <= 1e-8:
            rng = 1.0
        scores.append(_ssim(a, b, data_range=rng))
    return float(np.mean(scores))


def all_metrics(pred, target, mask=None):
    return {
        "rmse": rmse(pred, target, mask),
        "mae": mae(pred, target, mask),
        "ssim": ssim_score(pred, target),  # без маски, див. docstring ssim_score
        "sam_deg": sam_degrees(pred, target, mask),
    }


# --------------------------- базові методи ---------------------------

def _frame_cloud_frac(cloud):  # cloud: (T, H, W) -> (T,)
    return cloud.reshape(cloud.shape[0], -1).mean(axis=1)


def least_cloudy(s2, cloud, exclude_idx=None):
    """Повертає (прогноз, обраний_індекс) - кадр з мінімальною хмарністю, з опційним виключенням індексів."""
    frac = _frame_cloud_frac(cloud)
    if exclude_idx:
        frac = frac.copy()
        frac[list(exclude_idx)] = np.inf
    if not np.any(np.isfinite(frac)):
        return None, -1
    idx = int(np.argmin(frac))
    return s2[idx].copy(), idx


def linear_interp(s2, cloud, t_days, exclude_idx, clean_thresh=CLEAN_PIXEL_THRESH):
    """Потрапіксельна лінійна інтер-/екстраполяція в часі за чистими пікселями (день цілі приймається як 0).

    s2: (T, C, H, W), cloud: (T, H, W), t_days: (T,)
    Повертає (C, H, W); якщо для пікселя <2 чистих спостережень - береться найближче чисте
    (або, за відсутності жодного, середнє по всіх доступних кадрах для цього пікселя).
    """
    keep = np.ones(s2.shape[0], dtype=bool)
    if exclude_idx:
        keep[list(exclude_idx)] = False
    days = t_days[keep]
    frames = s2[keep]            # (T', C, H, W)
    clean = cloud[keep] < clean_thresh  # (T', H, W)

    order = np.argsort(days)
    return _linear_interp_vectorized(days[order], frames[order], clean[order])


def _linear_interp_vectorized(days_sorted, frames_sorted, clean_sorted):
    t_total, c_total, h, w = frames_sorted.shape
    out = np.empty((c_total, h, w), dtype=np.float32)
    flat_clean = clean_sorted.reshape(t_total, -1)         # (T', HW)
    flat_frames = frames_sorted.reshape(t_total, c_total, -1)  # (T', C, HW)
    n_pix = flat_clean.shape[1]
    for p in range(n_pix):
        mask = flat_clean[:, p]
        n_clean = int(mask.sum())
        y, x = divmod(p, w)
        if n_clean >= 2:
            xp = days_sorted[mask]
            for c in range(c_total):
                fp = flat_frames[mask, c, p]
                out[c, y, x] = np.interp(0.0, xp, fp, left=fp[0], right=fp[-1]) if len(xp) > 1 else fp[0]
        elif n_clean == 1:
            idx = int(np.argmax(mask))
            out[:, y, x] = flat_frames[idx, :, p]
        else:
            out[:, y, x] = flat_frames[:, :, p].mean(axis=0)
    return out


# --------------------------- обробка одного семпла ---------------------------

def process_sample(npz_path, clean_thresh=CLEAN_PIXEL_THRESH, eps_cloud=EPS_CLOUD, tol_days=TOL_SAME_DAYS):
    d = np.load(npz_path)
    s2 = d["s2"].astype(np.float32)
    cloud = d["cloud"].astype(np.float32)
    target = d["target"].astype(np.float32)
    t_days = d["t_days"].astype(np.int64)
    t_target = int(d["t_target"])
    target_idx = int(d["target_idx"]) if "target_idx" in d else -1

    frac = _frame_cloud_frac(cloud)
    raw_idx = int(np.argmin(frac))
    same_day = np.abs(t_days - t_target) <= tol_days
    risk_of_leak = bool(
        (target_idx >= 0 and raw_idx == target_idx)
        or (frac[raw_idx] <= eps_cloud and same_day[raw_idx])
    )

    exclude = set()
    if target_idx >= 0:
        exclude.add(target_idx)
    exclude |= set(int(i) for i in np.where(same_day)[0])

    pred_raw, _ = least_cloudy(s2, cloud, exclude_idx=None)
    pred_strict, strict_idx = least_cloudy(s2, cloud, exclude_idx=exclude)
    t_days_rel = t_days - t_target  # інтерполяція оцінює значення у відносному дні цілі (0)
    pred_interp = linear_interp(s2, cloud, t_days_rel, exclude_idx=exclude, clean_thresh=clean_thresh)

    # Хмарні пікселі самої цілі не маскуються при її виборі (поріг - середня хмарність
    # патча), тож виключаємо їх тут, щоб не порівнювати прогноз із залишковою хмарою.
    target_clear_mask = cloud[target_idx] < clean_thresh if target_idx >= 0 else None

    out = {"sample": Path(npz_path).stem, "risk_of_leak": risk_of_leak}
    out["least_cloudy_raw"] = all_metrics(pred_raw, target, mask=target_clear_mask)
    if pred_strict is not None:
        out["least_cloudy_strict"] = all_metrics(pred_strict, target, mask=target_clear_mask)
    out["linear_interp"] = all_metrics(pred_interp, target, mask=target_clear_mask)
    return out


# --------------------------- агрегація й звіт ---------------------------

METHODS = ["least_cloudy_raw", "least_cloudy_strict", "linear_interp"]
METRICS = ["rmse", "mae", "ssim", "sam_deg"]


def bootstrap_diff_ci(values_a, values_b, n_boot=2000, seed=0):
    """CI для різниці mean(a) - mean(b) парним бутстрепом по семплах. a, b - однакової довжини.

    Пари, де a або b - NaN (ціль без жодного чистого пікселя після маскування хмар),
    виключаються з обох масивів разом, щоб бутстреп не давав NaN."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(values_a), np.asarray(values_b)
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def summarize(results, methods=METHODS):
    """np.nanmean/nanstd - семпли з ціллю без жодного чистого пікселя дають NaN (див.
    _apply_mask) і виключаються з агрегату, а не псують його."""
    table = {}
    for m in methods:
        rows = [r[m] for r in results if m in r]
        if not rows:
            continue
        table[m] = {
            metric: (float(np.nanmean([row[metric] for row in rows])), float(np.nanstd([row[metric] for row in rows])))
            for metric in METRICS
        }
        table[m]["n"] = len(rows)
        table[m]["n_valid"] = int(np.sum(~np.isnan([row["rmse"] for row in rows])))
    return table


def render_table_md(table, title):
    lines = [f"### {title}", "", "| метод | n | RMSE | MAE | SSIM | SAM (°) |", "|---|---|---|---|---|---|"]
    for m in METHODS:
        if m not in table:
            continue
        t = table[m]
        n_disp = f"{t['n']}" if t.get("n_valid", t["n"]) == t["n"] else f"{t['n']} ({t['n_valid']} з чистими пікс.)"
        lines.append(
            f"| {m} | {n_disp} | {t['rmse'][0]:.4f} ± {t['rmse'][1]:.4f} "
            f"| {t['mae'][0]:.4f} ± {t['mae'][1]:.4f} "
            f"| {t['ssim'][0]:.4f} ± {t['ssim'][1]:.4f} "
            f"| {t['sam_deg'][0]:.2f} ± {t['sam_deg'][1]:.2f} |"
        )
    return "\n".join(lines)


def paired_bootstrap_section(results, methods=METHODS, baseline="least_cloudy_raw"):
    lines = ["### Парна різниця RMSE проти least_cloudy_raw (bootstrap 95% CI)", ""]
    base_rows = [r[baseline]["rmse"] for r in results if baseline in r]
    if not base_rows:
        return ""
    for m in methods:
        if m == baseline:
            continue
        rows = [(r[m]["rmse"], r[baseline]["rmse"]) for r in results if m in r and baseline in r]
        if not rows:
            continue
        a = [x[0] for x in rows]
        b = [x[1] for x in rows]
        mean_diff, lo, hi = bootstrap_diff_ci(a, b)
        sign = "краще" if mean_diff < 0 else "гірше"
        lines.append(f"- {m} - {baseline}: Δ RMSE = {mean_diff:+.4f} [{lo:+.4f}, {hi:+.4f}] ({sign}, якщо <0)")
    return "\n".join(lines)


def run(data_dir, out_dir, clean_thresh, eps_cloud, tol_days, n_boot, seed):
    paths = sorted(Path(data_dir).glob("*.npz"))
    if not paths:
        print(f"Не знайдено .npz файлів у {data_dir}", file=sys.stderr)
        sys.exit(1)

    results = []
    for p in paths:
        try:
            results.append(process_sample(p, clean_thresh, eps_cloud, tol_days))
        except Exception as e:
            print(f"[warn] пропущено {p.name}: {e}", file=sys.stderr)

    n_leak = sum(1 for r in results if r["risk_of_leak"])
    leak_free = [r for r in results if not r["risk_of_leak"]]

    table_full = summarize(results)
    table_leakfree = summarize(leak_free)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_raw.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    report = [
        "# Звіт: чесна база (крок 1)",
        "",
        f"Усього семплів: {len(results)}. Семплів з ризиком витоку цілі у вхід: {n_leak} "
        f"({100 * n_leak / max(len(results), 1):.1f} %).",
        "",
        render_table_md(table_full, "Повний набір (як у наївному порівнянні з літератури)"),
        "",
        render_table_md(table_leakfree, "Підвибірка без витоку цілі у вхід"),
        "",
        paired_bootstrap_section(leak_free if leak_free else results),
        "",
        "Примітка: якщо RMSE least_cloudy_raw на повному наборі суттєво нижчий, ніж на підвибірці без витоку, "
        "це означає, що базовий метод був завищений через потрапляння кадру-цілі у вхід.",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nЗбережено: {out_dir / 'report.md'} та {out_dir / 'metrics_raw.json'}")


# --------------------------- самоперевірка ---------------------------

def _make_synthetic_sample(rng, leak=False):
    t, c, h, w = 6, 4, 8, 8
    base = rng.normal(0.4, 0.05, size=(c, h, w)).astype(np.float32)
    s2 = np.stack([base + rng.normal(0, 0.02, size=(c, h, w)) for _ in range(t)]).astype(np.float32)
    cloud = rng.uniform(0.3, 0.9, size=(t, h, w)).astype(np.float32)
    t_days = np.arange(t) * 5

    target_idx = t - 1
    if leak:
        # ціль і є глобально найменш хмарним кадром -> наївний метод неявно підгляне ціль
        cloud[target_idx] = 0.0
    else:
        # ціль відносно чиста, але НЕ є глобальним мінімумом; мінімум - інший, далекий у часі кадр
        cloud[target_idx] = 0.05
        cloud[0] = 0.01

    target = s2[target_idx].copy()

    return {
        "s2": s2, "cloud": cloud, "target": target,
        "t_days": t_days.astype(np.int64), "t_target": np.int64(t_days[target_idx]),
        "target_idx": np.int64(target_idx),
    }


def selftest():
    rng = np.random.default_rng(42)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i in range(6):
            sample = _make_synthetic_sample(rng, leak=(i % 2 == 0))
            np.savez(tmp / f"sample_{i:03d}.npz", **sample)

        out = tmp / "out"
        results = []
        for p in sorted(tmp.glob("*.npz")):
            results.append(process_sample(p))

        ok = True
        n_leak = sum(r["risk_of_leak"] for r in results)
        if n_leak == 0:
            print("[FAIL] жодного семплу не позначено як ризик витоку, хоча половина мала бути"); ok = False
        if n_leak == len(results):
            print("[FAIL] усі семпли позначено як ризик витоку"); ok = False

        for r in results:
            if "least_cloudy_strict" not in r or "linear_interp" not in r:
                print(f"[FAIL] {r['sample']}: відсутні очікувані методи"); ok = False
                continue
            if r["risk_of_leak"] and not (r["least_cloudy_raw"]["rmse"] <= 1e-6):
                # коли є витік, raw має бути майже ідеальним (бо прогнозує саму ціль)
                if r["least_cloudy_raw"]["rmse"] > 1e-3:
                    print(f"[FAIL] {r['sample']}: витік позначено, але RMSE raw не близький до 0"); ok = False
            if not np.isfinite(r["least_cloudy_strict"]["rmse"]):
                print(f"[FAIL] {r['sample']}: least_cloudy_strict дав не-число"); ok = False

        table = summarize(results)
        print(render_table_md(table, "Самоперевірка: синтетичні дані"))
        print(f"\nсемплів з ризиком витоку: {n_leak}/{len(results)}")
        print("\n[PASS]" if ok else "\n[FAIL]")
        sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="директорія з .npz семплами")
    ap.add_argument("--out", default="results", help="директорія для звіту")
    ap.add_argument("--clean-thresh", type=float, default=CLEAN_PIXEL_THRESH)
    ap.add_argument("--eps-cloud", type=float, default=EPS_CLOUD)
    ap.add_argument("--tol-same-days", type=int, default=TOL_SAME_DAYS)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true", help="запустити самоперевірку на синтетичних даних і вийти")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.data:
        ap.error("--data обов'язковий, якщо не задано --selftest")

    run(args.data, args.out, args.clean_thresh, args.eps_cloud, args.tol_same_days, args.n_boot, args.seed)


if __name__ == "__main__":
    main()
