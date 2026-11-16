#!/usr/bin/env python3
"""
Розділ 11: інформаційна технологія аналізу стану поля на відкритих даних Sentinel-2 L2A Полтавщини (2021).

Одиниця аналізу - ділянка 8 × 8 пікселів 20 м (160 × 160 м, 2,56 га) у зонах test і calib (розділ 7).
Дата чиста для ділянки, якщо безхмарних (CLD = 0) і дійсних пікселів ≥ --min-clear; значення - медіана NDVI.

Для кожної ділянки й кожного дня сезону технологія видає запис стану поля:
    оцінка NDVI = останнє чисте значення (утримання; за NDVI причинна модель розділу 7 не точніша за нього),
    90 % інтервал = ± q(пропуск[, місяць]), q калібровано конформно на зоні calib,
    джерело (знімок / прогноз), давність знімка, тривога детектора, позначка «потрібна перевірка» (q > Δ),
    походження (сцена, файл).
Калібрування й перевірка інтервалів - на парах чистих дат (t0 < t1) однієї ділянки: прогноз із t0 на t1,
як якби дати між ними були хмарні (пропуски до --max-gap діб). Варіанти калібрування:
    marginal - один q на всі пропуски;  gap - окремий q для кожного інтервалу тривалості пропуску;
    gap×month - окремий q для кожної пари (тривалість пропуску, місяць цільової дати).
Умовне покриття перевіряється окремо за місяцями й тривалістю пропуску.
Детектор: падіння NDVI між сусідніми чистими датами більше за конформний (1 − α)-квантиль падінь на calib
у тому самому інтервалі пропуску (частка тривог на calib = α за побудовою).

    python scripts/field_state.py --root data/l2a/poltava2021/zones --out results/field_state_poltava
"""
import argparse
import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import rasterio

B4, B8 = 3, 7
GAP_BINS = ((1, 10), (11, 20), (21, 40), (41, 60))
DELTAS = (0.10, 0.15, 0.20, 0.30)
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_patch_(\d+)")


def conf_q(res, alpha):
    res = np.sort(np.asarray(res, dtype=float))
    if len(res) == 0:
        return float("nan")
    k = min(int(np.ceil((len(res) + 1) * (1 - alpha))), len(res))
    return float(res[k - 1])


def gap_bin(g):
    return next((lo for lo, hi in GAP_BINS if lo <= g <= hi), None)


def load_zone(zone_dir, block, min_clear, skip=frozenset()):
    """-> {(patch, br, bc): {ordinal: ndvi}}, {ordinal: scene}. skip - дати сцен, відкинутих контролем якості."""
    series, scenes = defaultdict(dict), {}
    for f in sorted((zone_dir / "S2").glob("*/*.tif")):
        m = DATE_RE.search(f.name)
        if m.group(1) in skip:
            continue
        t, patch = date.fromisoformat(m.group(1)).toordinal(), int(m.group(2))
        scenes[t] = f.parent.name
        with rasterio.open(f) as r:
            a = r.read([B4 + 1, B8 + 1]).astype(np.float32) / 10000.0
        with rasterio.open(Path(str(f).replace("/S2/", "/CLD/"))) as r:
            cld = r.read(1)
        red, nir = a
        valid = (cld == 0) & (red > 0) & (nir > 0)
        ndvi = np.where(valid, (nir - red) / np.maximum(nir + red, 1e-6), np.nan)
        H, W = ndvi.shape
        nb = H // block
        v = valid[:nb * block, :nb * block].reshape(nb, block, nb, block).mean(axis=(1, 3))
        med = np.nanmedian(ndvi[:nb * block, :nb * block].reshape(nb, block, nb, block)
                           .transpose(0, 2, 1, 3).reshape(nb, nb, -1), axis=2)
        for br, bc in zip(*np.where(v >= min_clear)):
            x = float(med[br, bc])
            if -0.2 < x <= 1:
                series[(patch, int(br), int(bc))][t] = x
    return series, scenes


def pairs(series, max_gap):
    """Пари (t0, t1) чистих дат однієї ділянки: залишок |y1 − y0|, падіння y0 − y1, пропуск, місяць t1."""
    out = []
    for s in series.values():
        ts = sorted(s)
        for i, t1 in enumerate(ts):
            for t0 in ts[:i]:
                g = t1 - t0
                if g <= max_gap:
                    out.append((g, date.fromordinal(t1).month, s[t1] - s[t0], t0 == ts[i - 1]))
    return np.array([(g, mo, d, nb) for g, mo, d, nb in out], dtype=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/l2a/poltava2021/zones")
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--min-clear", type=float, default=0.9)
    ap.add_argument("--max-gap", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--alpha-event", type=float, default=0.05)
    ap.add_argument("--season", nargs=2, default=["2021-04-01", "2021-10-15"])
    ap.add_argument("--qc", default=None, help="results/data_audit/summary.json: відкинути позначені сцени")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)
    skip = {"calib": set(), "test": set()}
    if args.qc:
        region = root.parent.name
        for fl in json.loads(Path(args.qc).read_text())["flags"]:
            if fl["region"] == region and fl["zone"] in skip:
                skip[fl["zone"]].add(fl["date"])

    cal_s, _ = load_zone(root / "calib", args.block, args.min_clear, frozenset(skip["calib"]))
    tst_s, scenes = load_zone(root / "test", args.block, args.min_clear, frozenset(skip["test"]))
    P_cal, P_tst = pairs(cal_s, args.max_gap), pairs(tst_s, args.max_gap)
    res_cal, res_tst = np.abs(P_cal[:, 2]), np.abs(P_tst[:, 2])
    gb_cal = np.array([gap_bin(int(g)) for g in P_cal[:, 0]])
    gb_tst = np.array([gap_bin(int(g)) for g in P_tst[:, 0]])
    months = sorted(set(P_tst[:, 1].astype(int)))

    # --- калібрування інтервалів ---
    q_marg = conf_q(res_cal, args.alpha)
    q_gap = {lo: conf_q(res_cal[gb_cal == lo], args.alpha) for lo, _ in GAP_BINS}
    q_gm = {}
    for lo, _ in GAP_BINS:
        for mo in range(1, 13):
            sel = (gb_cal == lo) & (P_cal[:, 1] == mo)
            q_gm[(lo, mo)] = conf_q(res_cal[sel], args.alpha) if sel.sum() >= 200 else q_gap[lo]
    qv = {"marginal": np.full(len(P_tst), q_marg),
          "gap": np.array([q_gap[b] for b in gb_tst]),
          "gap×month": np.array([q_gm[(b, int(m))] for b, m in zip(gb_tst, P_tst[:, 1])])}
    cov = {}
    for k, q in qv.items():
        hit = res_tst <= q
        cov[k] = {"overall": float(hit.mean()), "halfwidth_mean": float(q.mean()),
                  "by_month": {int(m): float(hit[P_tst[:, 1] == m].mean()) for m in months},
                  "by_gap": {lo: float(hit[gb_tst == lo].mean()) for lo, _ in GAP_BINS},
                  "worst_month": float(min(hit[P_tst[:, 1] == m].mean() for m in months
                                           if (P_tst[:, 1] == m).sum() >= 200))}

    # --- детектор різких падінь (лише сусідні чисті дати) ---
    nb_cal, nb_tst = P_cal[:, 3] == 1, P_tst[:, 3] == 1
    drop_cal, drop_tst = -P_cal[:, 2], -P_tst[:, 2]
    q_ev = {lo: conf_q(drop_cal[nb_cal & (gb_cal == lo)], args.alpha_event) for lo, _ in GAP_BINS}
    thr_tst = np.array([q_ev[b] for b in gb_tst])
    alarm = nb_tst & (drop_tst > thr_tst)
    alarms_by_month = {int(m): {"rate": float(alarm[nb_tst & (P_tst[:, 1] == m)].mean()),
                                "n": int(alarm[P_tst[:, 1] == m].sum())} for m in months}

    # --- щоденні записи стану поля (тест) ---
    s0, s1 = (date.fromisoformat(d).toordinal() for d in args.season)
    clear_dates = sorted(scenes)
    need = {d: defaultdict(lambda: [0, 0]) for d in DELTAS}
    age_days, n_rec = [], 0
    examples = {}
    alarm_set = set()
    for key, s in tst_s.items():
        ts = sorted(s)
        for i in range(1, len(ts)):
            g = ts[i] - ts[i - 1]
            b = gap_bin(g)
            if b is not None and s[ts[i - 1]] - s[ts[i]] > q_ev[b]:
                alarm_set.add((key, ts[i]))
    ex_keys = set(list(tst_s)[::max(1, len(tst_s) // 6)][:6])
    for key, s in tst_s.items():
        ts = sorted(s)
        recs = []
        for t in range(s0, s1 + 1):
            before = [u for u in ts if u <= t]
            if not before:
                continue
            t0 = before[-1]
            g = t - t0
            mo = date.fromordinal(t).month
            if g == 0:
                hw, src = 0.0, "знімок"
            else:
                b = gap_bin(g)
                hw = q_gm[(b, mo)] if b is not None else float("inf")
                src = "прогноз (утримання)"
            n_rec += 1
            age_days.append(g)
            for d in DELTAS:
                need[d][mo][0] += hw > d
                need[d][mo][1] += 1
            if key in ex_keys:
                recs.append({"date": date.fromordinal(t).isoformat(), "ndvi": round(s[t0], 4),
                             "halfwidth": None if hw == float("inf") else round(hw, 4), "source": src,
                             "age_days": g, "alarm": (key, t0) in alarm_set and g == 0,
                             "provenance": {"scene_date": date.fromordinal(t0).isoformat(),
                                            "scene_id": f"poltava2021/test/ImgNo_{scenes[t0]}",
                                            "patch": key[0], "block": [key[1], key[2]],
                                            "method": "persist", "calibration": "conformal gap×month, zone calib"}})
        if key in ex_keys:
            examples["%d_%d_%d" % key] = recs
    need_out = {str(d): {"overall": sum(v[0] for v in need[d].values()) / sum(v[1] for v in need[d].values()),
                         "by_month": {int(m): v[0] / v[1] for m, v in sorted(need[d].items())}} for d in DELTAS}

    # --- групи культур за траєкторією NDVI (без карти культур): пік у квітні-червні - озимі,
    # пік у липні-серпні - ярі просапні, максимум < 0,5 - не рілля / пари ---
    def group(s):
        ts = sorted(s)
        v = np.array([s[t] for t in ts])
        if v.max() < 0.5 or len(ts) < 10:
            return "інше"
        pk = date.fromordinal(ts[int(v.argmax())]).month
        return "озимі" if pk <= 6 else "ярі"
    grp = {k: group(s) for k, s in tst_s.items()}
    by_grp = defaultdict(lambda: defaultdict(int))
    peak = defaultdict(list)
    for key, s in tst_s.items():
        ts = sorted(s)
        peak[grp[key]].append(date.fromordinal(ts[int(np.argmax([s[t] for t in ts]))]).timetuple().tm_yday)
        for i in range(1, len(ts)):
            b = gap_bin(ts[i] - ts[i - 1])
            if b is not None and s[ts[i - 1]] - s[ts[i]] > q_ev[b]:
                by_grp[grp[key]][date.fromordinal(ts[i]).month] += 1
    groups = {g: {"blocks": sum(1 for v in grp.values() if v == g),
                  "peak_doy_median": float(np.median(peak[g])) if peak[g] else None,
                  "alarms_by_month": dict(sorted(by_grp[g].items()))} for g in ("озимі", "ярі", "інше")}

    summary = {
        "qc_skipped": {k: sorted(v) for k, v in skip.items()},
        "groups": groups,
        "blocks": {"calib": len(cal_s), "test": len(tst_s)}, "block_m": args.block * 20,
        "scenes": len(scenes), "scene_dates": [date.fromordinal(t).isoformat() for t in clear_dates],
        "clear_obs_per_block_median": float(np.median([len(s) for s in tst_s.values()])),
        "pairs": {"calib": int(len(P_cal)), "test": int(len(P_tst))},
        "q": {"marginal": q_marg, "gap": {str(k): v for k, v in q_gap.items()},
              "gap×month": {f"{b}/{m}": v for (b, m), v in q_gm.items() if not np.isnan(v)}},
        "coverage": cov,
        "event": {"alpha": args.alpha_event, "q_by_gap": {str(k): v for k, v in q_ev.items()},
                  "alarm_rate_test": float(alarm[nb_tst].mean()), "by_month": alarms_by_month},
        "daily": {"records": n_rec, "age_median": float(np.median(age_days)),
                  "age_p90": float(np.percentile(age_days, 90)), "need_check": need_out},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    with open(out / "records_example.jsonl", "w") as f:
        for k, recs in examples.items():
            for r in recs:
                f.write(json.dumps({"block": k, **r}, ensure_ascii=False) + "\n")

    L = ["# Технологія аналізу стану поля: Полтавщина 2021 (Sentinel-2 L2A)", "",
         f"Ділянок {args.block * 20} × {args.block * 20} м: calib {len(cal_s)}, test {len(tst_s)}; сцен {len(scenes)}; "
         f"чистих спостережень на ділянку (медіана) {summary['clear_obs_per_block_median']:.0f}. "
         f"Пар для калібрування {len(P_cal)}, для перевірки {len(P_tst)}.", "",
         "## Покриття 90 % інтервалу на зоні test", "",
         "| Калібрування | Покриття | Найгірший місяць | Середня півширина | " +
         " | ".join(f"пропуск {lo}–{hi}" for lo, hi in GAP_BINS) + " |",
         "|---|---|---|---|" + "---|" * len(GAP_BINS)]
    for k, c in cov.items():
        L.append(f"| {k} | {c['overall']:.3f} | {c['worst_month']:.3f} | {c['halfwidth_mean']:.3f} | " +
                 " | ".join(f"{c['by_gap'][lo]:.3f}" for lo, _ in GAP_BINS) + " |")
    L += ["", "Покриття за місяцями: " + "; ".join(
        f"{k}: " + ", ".join(f"{m}: {v:.2f}" for m, v in c["by_month"].items()) for k, c in cov.items()), "",
          "Півширина gap: " + ", ".join(f"{lo}–{hi} діб {q_gap[lo]:.3f}" for lo, hi in GAP_BINS), "",
          f"## Детектор падінь (α = {args.alpha_event})", "",
          f"Частка тривог на test: {summary['event']['alarm_rate_test']:.3f}. За місяцями: " +
          ", ".join(f"{m}: {v['rate']:.3f} ({v['n']})" for m, v in alarms_by_month.items()), "",
          "## Потреба в перевірці (частка ділянко-днів, де півширина > Δ)", "",
          "| Δ | За сезон | " + " | ".join(f"міс. {m}" for m in sorted(need[DELTAS[0]])) + " |",
          "|---|---|" + "---|" * len(need[DELTAS[0]])]
    for d in DELTAS:
        v = need_out[str(d)]
        L.append(f"| {d} | {v['overall']:.2f} | " + " | ".join(f"{x:.2f}" for x in v["by_month"].values()) + " |")
    L += ["", "## Групи ділянок за траєкторією NDVI і тривоги за місяцями", "",
          "| Група | Ділянок | Медіана дня піку NDVI | Тривоги за місяцями |", "|---|---|---|---|"]
    for g, v in groups.items():
        L.append(f"| {g} | {v['blocks']} | {v['peak_doy_median']} | " +
                 ", ".join(f"{m}: {n}" for m, n in v["alarms_by_month"].items()) + " |")
    L += ["", f"Давність останнього чистого знімка: медіана {summary['daily']['age_median']:.0f} діб, "
          f"90-й процентиль {summary['daily']['age_p90']:.0f}."]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
