#!/usr/bin/env python3
"""
Розділ 9: узгодженість NDVI наземних камер PhenoCam і Sentinel-2 та відновлення пропущених супутникових дат.

Вхід - data/phenocam/<site>.json (scripts/phenocam_s2.py). Для кожної ділянки-року:
  - ряд S2: сцени з часткою чистих пікселів (SCL 4, 5) ≥ --min-clear; сцени з NDVI поза (-0,2; 1] відкидаються
    як радіометричні аномалії (частина сцен baseline 04.00 2022 р. без вбудованого зсуву +1000);
  - ряд камери: ndvi_50, ковзна медіана за 3 доби.
Кожна чиста дата S2 по черзі вважається пропущеною (хмара) і прогнозується методами:
  persist   - останнє попереднє чисте значення S2 (причинний);
  interp    - лінійна інтерполяція між сусідніми чистими датами (непричинний, потребує майбутнього знімка);
  cam_map   - NDVI камери в день t, переведений у шкалу S2 лінійним відображенням, підігнаним на попередніх
              чистих датах тієї ж ділянки-року (причинний; ≥ --min-fit дат);
  cam_delta - останнє чисте значення S2 плюс зміна відображеного NDVI камери від тієї дати (причинний).
Конформні інтервали: ділянки діляться навпіл (детерміновано за назвою) на калібрувальні й тестові;
q - квантиль |похибки| на калібрувальних ділянках з поправкою скінченної вибірки; покриття й ширина - на тестових.
Також Мондріанівське калібрування за довжиною пропуску (до найближчої попередньої чистої дати).
Двічі: калібрувальні/тестові міняються місцями.

    python scripts/phenocam_analysis.py --data data/phenocam --out results/phenocam
"""
import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np

from phenocam_s2 import is_crop

METHODS = ("persist", "interp", "cam_map", "cam_delta")
GAP_BINS = ((1, 10), (11, 20), (21, 40), (41, 400))


def d2i(s):
    return date.fromisoformat(s).toordinal()


def load(data, min_clear):
    sites = []
    for f in sorted(Path(data).glob("*.json")):
        if f.name == "sites.json":
            continue
        d = json.loads(f.read_text())
        if not is_crop(d.get("species")):
            continue
        cam = {d2i(k): v["ndvi"] for k, v in d["camera"].items()}
        s2, n_anom = {}, 0
        for r in d["s2"]:
            if r["clear_frac"] < min_clear or "ndvi_clear" not in r:
                continue
            if not -0.2 < r["ndvi_clear"] <= 1:
                n_anom += 1
                continue
            s2.setdefault(d2i(r["date"]), []).append(r["ndvi_clear"])  # дві сцени за день (сусідні орбіти)
        s2 = {k: float(np.mean(v)) for k, v in s2.items()}
        sites.append({"site": d["site"], "species": d.get("species"), "koeppen": d.get("koeppen"),
                      "lat": d["lat"], "lon": d["lon"], "years": d["years"], "cam": cam, "s2": s2, "n_anom": n_anom})
    return sites


def cam_at(cam, t):
    v = [cam[t + k] for k in (-1, 0, 1) if t + k in cam]
    return float(np.median(v)) if v else None


def predictions(site, min_fit, g=1):
    """Список записів: ціль t, справжнє S2, прогнози методів (None, якщо метод неможливий), довжина пропуску.
    g > 1 імітує хмарний пропуск: дати S2 у (t − g; t + g) вважаються закритими, підгонка - лише на датах ≤ t − g."""
    recs = []
    for y in site["years"]:
        days = sorted(t for t in site["s2"] if date.fromordinal(t).year == y)
        for t in days:
            before = [u for u in days if u <= t - g]
            if not before:
                continue
            after = [u for u in days if u >= t + g]
            prev, nxt = before[-1], (after[0] if after else None)
            truth = site["s2"][t]
            p = {"persist": site["s2"][prev], "interp": None, "cam_map": None, "cam_delta": None}
            if nxt is not None:
                w = (t - prev) / (nxt - prev)
                p["interp"] = (1 - w) * site["s2"][prev] + w * site["s2"][nxt]
            fit = [(c, site["s2"][u]) for u in before if (c := cam_at(site["cam"], u)) is not None]
            ct, cp = cam_at(site["cam"], t), cam_at(site["cam"], prev)
            if len(fit) >= min_fit and ct is not None:
                x, yv = np.array(fit).T
                if np.ptp(x) > 0.02:
                    a, b = np.polyfit(x, yv, 1)
                    p["cam_map"] = float(a * ct + b)
                    if cp is not None:
                        p["cam_delta"] = float(site["s2"][prev] + a * (ct - cp))
            recs.append({"site": site["site"], "t": t, "truth": truth, "gap": t - prev, "pred": p})
    return recs


def agreement(site, min_pairs=10):
    pairs = [(c, v) for t, v in site["s2"].items() if (c := cam_at(site["cam"], t)) is not None]
    if len(pairs) < min_pairs:
        return None
    c, s = np.array(pairs).T
    a, b = np.polyfit(c, s, 1)
    resid = s - (a * c + b)
    return {"n": len(pairs), "r": float(np.corrcoef(c, s)[0, 1]), "slope": float(a), "intercept": float(b),
            "rmse_map": float(np.sqrt(np.mean(resid ** 2))), "bias_raw": float(np.mean(s - c))}


def conf_q(res, alpha):
    res = np.sort(np.asarray(res))
    n = len(res)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(res[min(k, n) - 1]) if n else np.nan


def split(name):
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 2


def fmt(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/phenocam")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-clear", type=float, default=0.9)
    ap.add_argument("--min-fit", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--sim-gaps", type=int, nargs="*", default=[10, 20, 30],
                    help="імітовані хмарні пропуски, діб у кожен бік від цілі")
    ap.add_argument("--min-r", type=float, default=0.5,
                    help="ділянки з r камера-S2 нижче порогу (камера не бачить того ж поля) - окремо")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sites = load(args.data, args.min_clear)
    agr = {s["site"]: agreement(s) for s in sites}
    good = [s for s in sites if agr[s["site"]] and agr[s["site"]]["r"] >= args.min_r]
    mism = [s["site"] for s in sites if agr[s["site"]] and agr[s["site"]]["r"] < args.min_r]
    recs = [r for s in good for r in predictions(s, args.min_fit)]
    sim = {g: [r for s in good for r in predictions(s, args.min_fit, g)] for g in args.sim_gaps}
    sim = {g: [r for r in rr if all(r["pred"][m] is not None for m in METHODS)] for g, rr in sim.items()}
    # порівняння на спільній множині: усі чотири методи можливі
    common = [r for r in recs if all(r["pred"][m] is not None for m in METHODS)]
    err = {m: np.array([abs(r["pred"][m] - r["truth"]) for r in common]) for m in METHODS}

    L = ["# Розділ 9: NDVI наземних камер PhenoCam і Sentinel-2", ""]
    L += [f"Ділянок із даними: {len(sites)}; з ≥ 10 спільними датами: {sum(v is not None for v in agr.values())}; "
          f"узгоджених (r ≥ {args.min_r}): {len(good)}. Ділянки з r < {args.min_r} (камера бачить інше поле, ніж "
          f"квадрат S2): {', '.join(mism) or 'немає'}. Відкинуто сцен-аномалій (NDVI поза (−0,2; 1]): "
          f"{sum(s['n_anom'] for s in sites)}.", ""]
    L += ["## 9.3. Узгодженість (чисті дати S2, частка чистих пікселів ≥ %.2f)" % args.min_clear, "",
          "| Ділянка | Культури | Кліматичний тип | Пар | r | Нахил | Зсув | RMSE після відображення | S2 − камера |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in sites:
        a = agr[s["site"]]
        if a:
            L.append(f"| {s['site']} | {str(s['species'] or '')[:40]} | {s['koeppen'] or ''} | {a['n']} | "
                     f"{a['r']:.2f} | {a['slope']:.2f} | {a['intercept']:+.2f} | {a['rmse_map']:.3f} | {a['bias_raw']:+.3f} |")
    rs = np.array([agr[s["site"]]["r"] for s in good])
    rm = np.array([agr[s["site"]]["rmse_map"] for s in good])
    if len(rs):
        L += ["", f"Узгоджені ділянки: медіана r = {np.median(rs):.2f} (міжквартильний {np.percentile(rs, 25):.2f}–"
              f"{np.percentile(rs, 75):.2f}), медіана RMSE після відображення = {np.median(rm):.3f}."]

    L += ["", "## 9.4. Відновлення пропущеної дати S2", "",
          f"Спільна множина (усі методи можливі): {len(common)} цільових дат на {len({r['site'] for r in common})} ділянках.", "",
          "| Метод | Причинний | MAE | Медіана |похибки| | P90 |похибки| |", "|---|---|---|---|---|"]
    for m in METHODS:
        L.append(f"| {m} | {'ні' if m == 'interp' else 'так'} | {fmt(err[m].mean())} | {fmt(np.median(err[m]))} | "
                 f"{fmt(np.percentile(err[m], 90))} |")
    L += ["", "MAE за довжиною пропуску (діб від останньої чистої дати):", "",
          "| Пропуск, діб | Дат | " + " | ".join(METHODS) + " |", "|---|---|" + "---|" * len(METHODS)]
    for lo, hi in GAP_BINS:
        sel = [r for r in common if lo <= r["gap"] <= hi]
        if sel:
            L.append(f"| {lo}–{hi} | {len(sel)} | " + " | ".join(
                fmt(np.mean([abs(r["pred"][m] - r["truth"]) for r in sel])) for m in METHODS) + " |")

    L += ["", "Імітований хмарний пропуск: дати S2 ближче ніж G діб до цілі (в обидва боки) вважаються закритими.", "",
          "| G, діб | Дат | " + " | ".join(f"MAE {m}" for m in METHODS) + " | Частка, де cam_delta краща за persist |",
          "|---|---|" + "---|" * (len(METHODS) + 1)]
    for g, rr in sim.items():
        if rr:
            e = {m: np.array([abs(r["pred"][m] - r["truth"]) for r in rr]) for m in METHODS}
            L.append(f"| {g} | {len(rr)} | " + " | ".join(fmt(e[m].mean()) for m in METHODS)
                     + f" | {np.mean(e['cam_delta'] < e['persist']):.2f} |")
    L += ["", f"## 9.4. Конформні інтервали (1 − α = {1 - args.alpha:.2f}), калібрування на іншій половині ділянок", "",
          "| Пропуски | Метод | Варіант | Розбиття | Калібр. дат | Тест. дат | Покриття | Середня півширина |",
          "|---|---|---|---|---|---|---|---|"]
    summary = {}
    sets = [("природні", common)] + [(f"G = {g}", rr) for g, rr in sim.items() if rr]
    for set_name, data in sets:
        for m in ("persist", "cam_map", "cam_delta"):
            for variant in ("спільний q", "Мондріан за пропуском"):
                covs, widths = [], []
                for fold in (0, 1):
                    cal = [r for r in data if split(r["site"]) != fold]
                    tst = [r for r in data if split(r["site"]) == fold]
                    res = lambda rr: [abs(r["pred"][m] - r["truth"]) for r in rr]
                    if variant == "спільний q":
                        qs = [conf_q(res(cal), args.alpha)] * len(tst)
                    else:
                        qb = {b: conf_q(res([r for r in cal if b[0] <= r["gap"] <= b[1]]), args.alpha) for b in GAP_BINS}
                        qs = [next(qb[b] for b in GAP_BINS if b[0] <= r["gap"] <= b[1]) for r in tst]
                    cov = np.mean([abs(r["pred"][m] - r["truth"]) <= q for r, q in zip(tst, qs)])
                    covs.append(cov)
                    widths.append(np.nanmean(qs))
                    L.append(f"| {set_name} | {m} | {variant} | {fold} | {len(cal)} | {len(tst)} | {cov:.3f} | "
                             f"{np.nanmean(qs):.3f} |")
                summary[f"{set_name}/{m}/{variant}"] = {"coverage": float(np.mean(covs)),
                                                        "halfwidth": float(np.mean(widths))}
    (out / "summary.md").write_text("\n".join(L) + "\n")
    (out / "summary.json").write_text(json.dumps({
        "n_sites": len(sites), "n_good": len(good), "mismatched": mism, "n_common": len(common),
        "mae": {m: float(err[m].mean()) for m in METHODS}, "conformal": summary,
        "agreement": agr}, indent=1, ensure_ascii=False))
    print("\n".join(L))


if __name__ == "__main__":
    main()
