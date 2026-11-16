#!/usr/bin/env python3
"""
Задача 2 (БПЛА): сегментація стану посіву за мультиспектральними знімками БПЛА
з конформними множинами класів.

Дані: Suiçmez et al. (2026), Zenodo 10.5281/zenodo.20332029, CC BY 4.0.
Два поля кукурудзи (водний стрес, іржа), фрагменти 224x224x6, маски з 5 класів:
0 ґрунт/фон, 1 слабкий водний стрес, 2 сильний водний стрес, 3 здорова кукурудза, 4 іржа.

Розбиття просторове: кожне поле ділиться на блоки BLOCK x BLOCK фрагментів, блоки
випадково (фіксований split-seed) ідуть у train / calib / test (60/20/20).
Навчальні фрагменти, що межують (8-сусідство) з calib/test, відкидаються як буфер.

Порівнюються:
  unet       - U-Net (UNetTrunk з model.py) на 5 каналах + NDVI + NDRE;
  rf_all     - випадковий ліс по пікселях на тих самих 7 ознаках;
  rf_indices - випадковий ліс лише на NDVI і NDRE (чи вистачає самих індексів).

Конформне калібрування (для unet): оцінка s = 1 - p_y, поріг - на calib.
  marginal  - один поріг для всіх класів;
  mondrian  - окремий поріг для кожного класу (покриття для кожного класу окремо).
Піксель із множиною з >1 класу позначається як «недостовірно».

    python scripts/uav_stress_seg.py --seeds 0 1 2 3 4 --out results/uav_stress_seg
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import UNetTrunk  # noqa: E402

CLASSES = ["soil", "low_water_stress", "high_water_stress", "healthy", "rust"]
NC = len(CLASSES)
BLOCK = 4  # фрагментів у блоці по кожній осі (4 x 224 px ≈ 8,6 м)
TILE = 224
EPS = 1e-6


def features(img):
    """img: H x W x 6 (float16). Повертає 7 x H x W: смуги 1-5, NDVI, NDRE."""
    b = img[..., :5].astype(np.float32)
    red, nir, re = b[..., 2], b[..., 3], b[..., 4]
    ndvi = (nir - red) / (nir + red + EPS)
    ndre = (nir - re) / (nir + re + EPS)
    return np.concatenate([b, ndvi[..., None], ndre[..., None]], axis=-1).transpose(2, 0, 1)


def load(root):
    rows = list(csv.DictReader(open(root / "meta" / "patches.csv", encoding="utf-8-sig")))
    X, Y, meta = [], [], []
    for r in rows:
        stem = Path(r["image_npy"].replace("\\", "/")).stem
        X.append(features(np.load(root / "images" / f"{stem}.npy")))
        Y.append(np.array(Image.open(root / "masks" / f"{stem}.png"), dtype=np.int64))
        meta.append((r["src_task"], int(r["x0"]) // TILE, int(r["y0"]) // TILE))
    return np.stack(X), np.stack(Y), meta


def spatial_split(meta, split_seed):
    rng = np.random.default_rng(split_seed)
    blocks = sorted({(s, cx // BLOCK, cy // BLOCK) for s, cx, cy in meta})
    order = rng.permutation(len(blocks))
    n = len(blocks)
    part = {}
    for rank, i in enumerate(order):
        part[blocks[i]] = "train" if rank < 0.6 * n else ("calib" if rank < 0.8 * n else "test")
    split = [part[(s, cx // BLOCK, cy // BLOCK)] for s, cx, cy in meta]
    held = {(s, cx, cy) for (s, cx, cy), p in zip(meta, split) if p != "train"}
    for i, (s, cx, cy) in enumerate(meta):
        if split[i] == "train" and any((s, cx + dx, cy + dy) in held for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
            split[i] = "buffer"
    return np.array(split)


class SegNet(nn.Module):
    def __init__(self, in_ch, base=24):
        super().__init__()
        self.trunk = UNetTrunk(in_ch, base)
        self.out = nn.Conv2d(base, NC, 1)

    def forward(self, x):
        return self.out(self.trunk(x))


def augment(x, y):
    k = int(torch.randint(0, 4, (1,)))
    x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (1, 2))
    if torch.rand(1) < 0.5:
        x, y = torch.flip(x, (3,)), torch.flip(y, (2,))
    return x, y


def train_unet(Xtr, Ytr, seed, dev, epochs, bs=16, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = SegNet(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            x, y = augment(Xt[idx].to(dev), Yt[idx].to(dev))
            loss = F.cross_entropy(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  seed {seed} epoch {ep}: loss {tot / len(Xt):.4f}", flush=True)
    return net


@torch.no_grad()
def predict_proba(net, X, dev, bs=32):
    net.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(F.softmax(net(torch.from_numpy(X[i:i + bs]).to(dev)), dim=1).cpu().numpy().astype(np.float32))
    return np.concatenate(out)  # N x C x H x W


def seg_metrics(pred, y):
    cm = np.bincount(y.ravel() * NC + pred.ravel(), minlength=NC * NC).reshape(NC, NC)
    tp = np.diag(cm).astype(float)
    iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
    f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)
    present = cm.sum(1) > 0
    return {
        "pixel_acc": float(tp.sum() / cm.sum()),
        "miou": float(iou[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "iou": {CLASSES[c]: float(iou[c]) for c in range(NC) if present[c]},
        "confusion": cm.tolist(),
    }


def conformal_q(scores, alpha):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[min(k, n) - 1]) if n else 1.0


def conformal_eval(Pcal, Ycal, Ptest, Ytest, alpha, rng, n_cal=400_000):
    pc = Pcal.transpose(0, 2, 3, 1).reshape(-1, NC)
    yc = Ycal.ravel()
    sel = rng.choice(len(yc), size=min(n_cal, len(yc)), replace=False)
    pc, yc = pc[sel], yc[sel]
    s_cal = 1.0 - pc[np.arange(len(yc)), yc]
    pt = Ptest.transpose(0, 2, 3, 1).reshape(-1, NC)
    yt = Ytest.ravel()
    res = {}
    qs = {"marginal": np.full(NC, conformal_q(s_cal, alpha)),
          "mondrian": np.array([conformal_q(s_cal[yc == c], alpha) for c in range(NC)])}
    for name, q in qs.items():
        sets = (1.0 - pt) <= q[None, :]  # Mondrian: клас c входить, якщо його оцінка ≤ q_c
        size = sets.sum(1)
        covered = sets[np.arange(len(yt)), yt]
        top = pt.argmax(1)
        single = size == 1
        res[name] = {
            "q": q.tolist(),
            "coverage": float(covered.mean()),
            "coverage_per_class": {CLASSES[c]: float(covered[yt == c].mean()) for c in range(NC) if (yt == c).any()},
            "mean_set_size": float(size.mean()),
            "empty_share": float((size == 0).mean()),
            "uncertain_share": float((size > 1).mean()),
            "acc_confident": float((top[single] == yt[single]).mean()) if single.any() else None,
            "acc_uncertain": float((top[size > 1] == yt[size > 1]).mean()) if (size > 1).any() else None,
        }
    return res


def pixels(X, Y, n, rng, cols):
    f = X[:, cols].transpose(0, 2, 3, 1).reshape(-1, len(cols))
    y = Y.ravel()
    sel = rng.choice(len(y), size=min(n, len(y)), replace=False)
    return f[sel], y[sel]


def run_rf(Xtr, Ytr, Xte, Yte, cols, seed, n_train=300_000):
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(seed)
    f, y = pixels(Xtr, Ytr, n_train, rng, cols)
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=seed).fit(f, y)
    ft = Xte[:, cols].transpose(0, 2, 3, 1).reshape(-1, len(cols))
    pred = np.concatenate([rf.predict(ft[i:i + 2_000_000]) for i in range(0, len(ft), 2_000_000)])
    return seg_metrics(pred.reshape(Yte.shape), Yte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/uav_maize/02_processed_patches")
    ap.add_argument("--out", default="results/uav_stress_seg")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--skip-rf", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    X, Y, meta = load(Path(args.data))
    split = spatial_split(meta, args.split_seed)
    counts = {p: int((split == p).sum()) for p in ("train", "buffer", "calib", "test")}
    print("device", dev, "split", counts, flush=True)
    tr, ca, te = split == "train", split == "calib", split == "test"
    mu = X[tr].mean((0, 2, 3), keepdims=True)
    sd = X[tr].std((0, 2, 3), keepdims=True) + EPS
    Xn = ((X - mu) / sd).astype(np.float32)
    class_px = {p: np.bincount(Y[split == p].ravel(), minlength=NC).tolist() for p in ("train", "calib", "test")}
    json.dump({"split_counts": counts, "class_pixels": class_px, "block": BLOCK, "split_seed": args.split_seed,
               "patches": [[m[0], m[1], m[2], s] for m, s in zip(meta, split.tolist())]},
              open(out / "split.json", "w"), indent=1)

    per_seed = {}
    for seed in args.seeds:
        if (out / f"seed{seed}.json").exists():  # продовження перерваного прогону
            per_seed[seed] = json.load(open(out / f"seed{seed}.json"))
            continue
        t0 = time.time()
        net = train_unet(Xn[tr], Y[tr], seed, dev, args.epochs)
        Pca, Pte = predict_proba(net, Xn[ca], dev), predict_proba(net, Xn[te], dev)
        r = {"unet": seg_metrics(Pte.argmax(1), Y[te]),
             "conformal": conformal_eval(Pca, Y[ca], Pte, Y[te], args.alpha, np.random.default_rng(seed))}
        if not args.skip_rf:
            r["rf_all"] = run_rf(X[tr], Y[tr], X[te], Y[te], list(range(7)), seed)
            r["rf_indices"] = run_rf(X[tr], Y[tr], X[te], Y[te], [5, 6], seed)
        r["seconds"] = round(time.time() - t0, 1)
        per_seed[seed] = r
        json.dump(r, open(out / f"seed{seed}.json", "w"), indent=1)
        torch.save(net.state_dict(), out / f"seed{seed}.pt")
        print(f"seed {seed}: unet mIoU {r['unet']['miou']:.3f}"
              + (f", rf_all {r['rf_all']['miou']:.3f}, rf_indices {r['rf_indices']['miou']:.3f}" if "rf_all" in r else "")
              + f", conformal cov {r['conformal']['mondrian']['coverage']:.3f}", flush=True)

    def agg(get):
        v = np.array([get(per_seed[s]) for s in args.seeds], dtype=float)
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "values": v.tolist()}

    summary = {"alpha": args.alpha, "epochs": args.epochs, "seeds": args.seeds, "split_counts": counts}
    models = ["unet"] + ([] if args.skip_rf else ["rf_all", "rf_indices"])
    for m in models:
        summary[m] = {k: agg(lambda r, m=m, k=k: r[m][k]) for k in ("miou", "macro_f1", "pixel_acc")}
        summary[m]["iou"] = {c: agg(lambda r, m=m, c=c: r[m]["iou"][c]) for c in per_seed[args.seeds[0]][m]["iou"]}
    for mode in ("marginal", "mondrian"):
        cf = per_seed[args.seeds[0]]["conformal"][mode]
        summary[f"conformal_{mode}"] = {
            k: agg(lambda r, mode=mode, k=k: r["conformal"][mode][k])
            for k in ("coverage", "mean_set_size", "uncertain_share", "empty_share", "acc_confident", "acc_uncertain")
        }
        summary[f"conformal_{mode}"]["coverage_per_class"] = {
            c: agg(lambda r, mode=mode, c=c: r["conformal"][mode]["coverage_per_class"][c]) for c in cf["coverage_per_class"]
        }
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps({m: summary[m]["miou"] for m in models}, indent=1))


if __name__ == "__main__":
    main()
