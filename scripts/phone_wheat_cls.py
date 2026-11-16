#!/usr/bin/env python3
"""
Задача 2 (наземні фото): класифікація хвороб пшениці за польовими фото
з конформними множинами класів.

Дані: Long et al. (2023), Zenodo 10.5281/zenodo.7573133, CC BY 4.0.
999 фото, 5 класів: BrownRust, Healthy, Mildew, Septoria, YellowRust.

Розбиття групове: номер у назві файлу ділиться на GROUP, тож послідовні кадри
(імовірно, зняті підряд на одній ділянці) не потрапляють одночасно в train і test.
Групи всередині класу випадково (split-seed) ідуть у train / calib / test (60/20/20).
Додатково рахується кількість майже-дублікатів (aHash) між train і test.

Порівнюються:
  finetune - ResNet18 (ImageNet), донавчання всієї мережі;
  linprobe - логістична регресія на заморожених ознаках ResNet18.

Конформне калібрування (finetune): s = 1 - p_y, marginal і mondrian, як у uav_stress_seg.py.

    python scripts/phone_wheat_cls.py --seeds 0 1 2 3 4 --out results/phone_wheat_cls
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

CLASSES = ["BrownRust", "Healthy", "Mildew", "Septoria", "YellowRust"]
NC = len(CLASSES)
GROUP = 25
SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def list_files(root):
    items = []
    for p in sorted(root.iterdir()):
        m = re.match(r"([A-Za-z]+?)(\d+)\.(png|jpe?g)$", p.name, re.I)
        if m and m.group(1) in CLASSES:
            items.append((p, CLASSES.index(m.group(1)), int(m.group(2)), m.group(3).lower()))
    return items


def group_split(items, split_seed):
    rng = np.random.default_rng(split_seed)
    split = [None] * len(items)
    for c in range(NC):
        groups = sorted({n // GROUP for _, y, n, _ in items if y == c})
        order = rng.permutation(len(groups))
        g = len(groups)
        part = {groups[i]: ("train" if r < 0.6 * g else "calib" if r < 0.8 * g else "test") for r, i in enumerate(order)}
        for k, (_, y, n, _) in enumerate(items):
            if y == c:
                split[k] = part[n // GROUP]
    return np.array(split)


def ahash(img):
    a = np.asarray(img.convert("L").resize((8, 8), Image.BILINEAR), dtype=np.float32)
    return (a > a.mean()).ravel()


def load_images(items, size):
    imgs, hashes = [], []
    for p, *_ in items:
        im = Image.open(p).convert("RGB")
        hashes.append(ahash(im))
        im.thumbnail((size * 2, size * 2))  # зберігаємо запас для випадкового кадрування
        imgs.append(im.copy())
    return imgs, np.stack(hashes)


TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(SIZE, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.1),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
EVAL_TF = transforms.Compose([
    transforms.Resize(SIZE), transforms.CenterCrop(SIZE),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def backbone():
    return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)


@torch.no_grad()
def predict(net, imgs, dev, bs=32):
    net.eval()
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.stack([EVAL_TF(im) for im in imgs[i:i + bs]]).to(dev)
        out.append(F.softmax(net(x), 1).cpu().numpy())
    return np.concatenate(out)


def train_finetune(imgs, y, seed, dev, epochs, bs=32):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = backbone()
    net.fc = nn.Linear(net.fc.in_features, NC)
    net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    yt = torch.tensor(y)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(imgs))
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs].tolist()
            x = torch.stack([TRAIN_TF(imgs[j]) for j in idx]).to(dev)
            loss = F.cross_entropy(net(x), yt[idx].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  seed {seed} epoch {ep}: loss {tot / len(imgs):.4f}", flush=True)
    return net


@torch.no_grad()
def features(imgs, dev, bs=32):
    net = backbone()
    net.fc = nn.Identity()
    net.to(dev).eval()
    return np.concatenate([net(torch.stack([EVAL_TF(im) for im in imgs[i:i + bs]]).to(dev)).cpu().numpy()
                           for i in range(0, len(imgs), bs)])


def cls_metrics(pred, y):
    cm = np.bincount(y * NC + pred, minlength=NC * NC).reshape(NC, NC)
    tp = np.diag(cm).astype(float)
    f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)
    return {"acc": float(tp.sum() / cm.sum()), "macro_f1": float(f1.mean()),
            "f1": {CLASSES[c]: float(f1[c]) for c in range(NC)}, "confusion": cm.tolist()}


def conformal_q(scores, alpha):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[min(k, n) - 1]) if n else 1.0


def conformal_eval(Pc, yc, Pt, yt, alpha):
    s = 1.0 - Pc[np.arange(len(yc)), yc]
    qs = {"marginal": np.full(NC, conformal_q(s, alpha)),
          "mondrian": np.array([conformal_q(s[yc == c], alpha) for c in range(NC)])}
    res = {}
    for name, q in qs.items():
        sets = (1.0 - Pt) <= q[None, :]
        size = sets.sum(1)
        cov = sets[np.arange(len(yt)), yt]
        top = Pt.argmax(1)
        single = size == 1
        res[name] = {
            "q": q.tolist(), "coverage": float(cov.mean()),
            "coverage_per_class": {CLASSES[c]: float(cov[yt == c].mean()) for c in range(NC)},
            "mean_set_size": float(size.mean()), "empty_share": float((size == 0).mean()),
            "uncertain_share": float((size > 1).mean()),
            "acc_confident": float((top[single] == yt[single]).mean()) if single.any() else None,
            "acc_uncertain": float((top[size > 1] == yt[size > 1]).mean()) if (size > 1).any() else None,
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/wheat_phone/photos")
    ap.add_argument("--out", default="results/phone_wheat_cls")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = list_files(Path(args.data))
    split = group_split(items, args.split_seed)
    y = np.array([it[1] for it in items])
    imgs, H = load_images(items, SIZE)
    tr, ca, te = (split == p for p in ("train", "calib", "test"))

    # майже-дублікати між train і test (відстань Геммінга aHash ≤ 4 з 64)
    d = (H[te][:, None, :] != H[tr][None, :, :]).sum(-1)
    near_dup = int((d.min(1) <= 4).sum())
    fmt = {c: {f: int(sum(1 for it in items if it[1] == ci and it[3] == f)) for f in ("png", "jpeg", "jpg")}
           for ci, c in enumerate(CLASSES)}
    info = {"n": len(items), "counts": {p: {CLASSES[c]: int(((split == p) & (y == c)).sum()) for c in range(NC)}
                                        for p in ("train", "calib", "test")},
            "group": GROUP, "split_seed": args.split_seed, "near_dup_test_vs_train": near_dup, "formats": fmt}
    json.dump(info, open(out / "split.json", "w"), indent=1)
    print("device", dev, json.dumps(info), flush=True)

    Ftr = Fca = Fte = None
    per_seed = {}
    for seed in args.seeds:
        t0 = time.time()
        net = train_finetune([imgs[i] for i in np.where(tr)[0]], y[tr], seed, dev, args.epochs)
        Pc = predict(net, [imgs[i] for i in np.where(ca)[0]], dev)
        Pt = predict(net, [imgs[i] for i in np.where(te)[0]], dev)
        r = {"finetune": cls_metrics(Pt.argmax(1), y[te]), "conformal": conformal_eval(Pc, y[ca], Pt, y[te], args.alpha)}
        if Ftr is None:
            Fall = features(imgs, dev)
            Ftr, Fca, Fte = Fall[tr], Fall[ca], Fall[te]
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=3000, C=1.0, random_state=seed).fit(Ftr, y[tr])
        r["linprobe"] = cls_metrics(lr.predict(Fte), y[te])
        r["seconds"] = round(time.time() - t0, 1)
        per_seed[seed] = r
        json.dump(r, open(out / f"seed{seed}.json", "w"), indent=1)
        torch.save(net.state_dict(), out / f"seed{seed}.pt")
        print(f"seed {seed}: finetune acc {r['finetune']['acc']:.3f} f1 {r['finetune']['macro_f1']:.3f}, "
              f"linprobe acc {r['linprobe']['acc']:.3f}, cov {r['conformal']['mondrian']['coverage']:.3f}", flush=True)

    def agg(get):
        v = np.array([get(per_seed[s]) for s in args.seeds], dtype=float)
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "values": v.tolist()}

    summary = {"alpha": args.alpha, "epochs": args.epochs, "seeds": args.seeds, "split": info}
    for m in ("finetune", "linprobe"):
        summary[m] = {k: agg(lambda r, m=m, k=k: r[m][k]) for k in ("acc", "macro_f1")}
        summary[m]["f1"] = {c: agg(lambda r, m=m, c=c: r[m]["f1"][c]) for c in CLASSES}
    for mode in ("marginal", "mondrian"):
        summary[f"conformal_{mode}"] = {
            k: agg(lambda r, mode=mode, k=k: np.nan if r["conformal"][mode][k] is None else r["conformal"][mode][k])
            for k in ("coverage", "mean_set_size", "uncertain_share", "empty_share", "acc_confident", "acc_uncertain")}
        summary[f"conformal_{mode}"]["coverage_per_class"] = {
            c: agg(lambda r, mode=mode, c=c: r["conformal"][mode]["coverage_per_class"][c]) for c in CLASSES}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps({m: summary[m]["acc"] for m in ("finetune", "linprobe")}, indent=1))


if __name__ == "__main__":
    main()
