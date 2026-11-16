#!/usr/bin/env python3
"""
Розділ 10.3-10.4: розпізнавання хвороб за фото листка з конформними множинами класів і перенесення
з лабораторних знімків (PlantVillage) на польові (PlantDoc).

Дані (лише для навчання й перевірки; зображення й ваги моделей не поширюються):
  - PlantVillage (Hughes, Salathé, arXiv:1511.08060; tfds 'plant_village'): лабораторні знімки листків на
    однорідному тлі. Беруться лише 28 класів, що мають відповідник у PlantDoc.
  - PlantDoc (Singh et al., CODS-COMAD 2020; github.com/pratikkayal/PlantDoc-Dataset, CC BY 4.0):
    польові фото з інтернету, 28 класів (27 у тесті авторів). Точні й майже точні дублікати (16 × 16 aHash) лишаються по одному.
PlantDoc на кожен запуск ділиться стратифіковано: 50 % донавчання, 25 % калібрування, 25 % тест.

Сценарії (для кожного seed):
  S0  навчання PV, калібрування PV, тест PV            - гарантія в межах одного домену;
  S1  навчання PV, калібрування PV, тест PlantDoc      - перенесення в поле без змін;
  S2  навчання PV, калібрування PlantDoc, тест PlantDoc - лише місцеве калібрування;
  S3  навчання PV + донавчання PlantDoc, калібрування й тест PlantDoc;
  S4  навчання лише PlantDoc (від ImageNet), калібрування й тест PlantDoc - чи допомагають лабораторні дані.
Множини: LAC (оцінка 1 − p_y) і APS (без рандомізації), α = 0,1. Виводяться точність top-1, точність
«здоровий / уражений», покриття, середній розмір множини, частка одноелементних множин і точність на них.

    python scripts/disease_transfer.py --work /content/work --out results/disease_transfer --seeds 0 1 2 3 4
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# PlantDoc -> ключові слова назви класу PlantVillage (tfds), усі мають бути в назві
MAP = {
    "Apple Scab Leaf": ("apple", "scab"),
    "Apple leaf": ("apple", "healthy"),
    "Apple rust leaf": ("apple", "rust"),
    "Bell_pepper leaf spot": ("pepper", "spot"),
    "Bell_pepper leaf": ("pepper", "healthy"),
    "Blueberry leaf": ("blueberry", "healthy"),
    "Cherry leaf": ("cherry", "healthy"),
    "Corn Gray leaf spot": ("corn", "gray"),
    "Corn leaf blight": ("corn", "blight"),
    "Corn rust leaf": ("corn", "rust"),
    "Peach leaf": ("peach", "healthy"),
    "Potato leaf early blight": ("potato", "early"),
    "Potato leaf late blight": ("potato", "late"),
    "Raspberry leaf": ("raspberry", "healthy"),
    "Soyabean leaf": ("soybean", "healthy"),
    "Squash Powdery mildew leaf": ("squash", "powdery"),
    "Strawberry leaf": ("strawberry", "healthy"),
    "Tomato Early blight leaf": ("tomato", "early"),
    "Tomato Septoria leaf spot": ("tomato", "septoria"),
    "Tomato leaf bacterial spot": ("tomato", "bacterial"),
    "Tomato leaf late blight": ("tomato", "late"),
    "Tomato leaf mosaic virus": ("tomato", "mosaic"),
    "Tomato leaf yellow virus": ("tomato", "yellow"),
    "Tomato leaf": ("tomato", "healthy"),
    "Tomato mold leaf": ("tomato", "mold"),
    "Tomato two spotted spider mites leaf": ("tomato", "spider"),
    "grape leaf black rot": ("grape", "black rot"),
    "grape leaf": ("grape", "healthy"),
}
CLASSES = sorted(MAP)
HEALTHY = {c for c, k in MAP.items() if "healthy" in k}
PD_REPO = "https://github.com/pratikkayal/PlantDoc-Dataset.git"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WORKERS = int(os.environ.get("DT_WORKERS", "4"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------- дані ----------

def map_pv_names(names):
    """Індекс назви класу PlantVillage -> клас PlantDoc (ключові слова MAP)."""
    pv2pd = {}
    for pd_cls, keys in MAP.items():
        norm = lambda n: n.lower().replace("_", " ")
        hits = [i for i, n in enumerate(names) if all(k in norm(n) for k in keys)]
        if pd_cls == "Apple rust leaf":
            hits = [i for i in hits if "cedar" in names[i].lower()]
        if pd_cls == "Corn leaf blight":
            hits = [i for i in hits if "northern" in names[i].lower()]
        if pd_cls == "Bell_pepper leaf spot":
            hits = [i for i in hits if "bacterial" in names[i].lower()]
        if len(hits) != 1:
            raise SystemExit(f"{pd_cls}: неоднозначна відповідність {[names[i] for i in hits]}")
        pv2pd[hits[0]] = pd_cls
    log("відповідність класів:", {names[k]: v for k, v in pv2pd.items()})
    return pv2pd


def prepare_plantvillage(work):
    """Потрібні класи PlantVillage у JPEG: work/pv/<клас PlantDoc>/<n>.jpg. Джерело - tfds 'plant_village';
    якщо tfds недоступний, - git-репозиторій spMohanty/PlantVillage-Dataset (raw/color)."""
    root = work / "pv"
    if (root / ".done").exists():
        return root
    counts = {}

    def save(img, c):
        d = root / c
        d.mkdir(parents=True, exist_ok=True)
        n = counts.get(c, 0)
        img.save(d / f"{n}.jpg", quality=95)
        counts[c] = n + 1

    try:
        import tensorflow_datasets as tfds
        ds, info = tfds.load("plant_village", split="train", as_supervised=True, with_info=True,
                             data_dir=str(work / "tfds"))
        pv2pd = map_pv_names(info.features["label"].names)
        for img, lab in tfds.as_numpy(ds):
            if int(lab) in pv2pd:
                save(Image.fromarray(img), pv2pd[int(lab)])
    except Exception as e:
        log("tfds недоступний, клоную GitHub-репозиторій PlantVillage:", repr(e)[:200])
        counts.clear()
        repo = work / "PlantVillage-Dataset"
        if not repo.exists():
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/spMohanty/PlantVillage-Dataset.git",
                            str(repo)], check=True)
        dirs = sorted(d for d in (repo / "raw" / "color").iterdir() if d.is_dir())
        pv2pd = map_pv_names([d.name for d in dirs])
        for i, c in pv2pd.items():
            for p in sorted(dirs[i].iterdir()):
                if p.suffix.lower() in IMG_EXT:
                    with Image.open(p) as im:
                        save(im.convert("RGB"), c)
    (root / ".done").write_text(json.dumps(counts, ensure_ascii=False))
    log("PlantVillage:", sum(counts.values()), "зображень")
    return root


def ahash(path):
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((16, 16), Image.BILINEAR), dtype=np.float32)
    return hashlib.md5(np.packbits(a > a.mean()).tobytes()).hexdigest()


def prepare_plantdoc(work):
    """Клон PlantDoc, усі зображення train + test разом, без дублікатів: список (шлях, клас)."""
    repo = work / "PlantDoc-Dataset"
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1", PD_REPO, str(repo)], check=True)
    items, seen, n_dup, n_bad = [], set(), 0, 0
    for split in ("train", "test"):
        for cdir in sorted((repo / split).iterdir()):
            if not cdir.is_dir():
                continue
            c = cdir.name.strip()
            if c not in MAP:
                raise SystemExit(f"невідомий клас PlantDoc: {c}")
            for p in sorted(cdir.iterdir()):
                if p.suffix.lower() not in IMG_EXT:
                    continue
                try:
                    h = ahash(p)
                except Exception:
                    n_bad += 1
                    continue
                if h in seen:
                    n_dup += 1
                    continue
                seen.add(h)
                items.append((str(p), CLASSES.index(c)))
    log(f"PlantDoc: {len(items)} зображень, відкинуто дублікатів {n_dup}, нечитаних {n_bad}")
    return items, {"n": len(items), "duplicates": n_dup, "unreadable": n_bad}


def stratified(items, fracs, seed):
    rng = random.Random(seed)
    by = {}
    for it in items:
        by.setdefault(it[1], []).append(it)
    parts = [[] for _ in fracs]
    for c in sorted(by):
        v = by[c][:]
        rng.shuffle(v)
        cuts = np.cumsum([0] + [int(round(f * len(v))) for f in fracs[:-1]] + [len(v)])
        cuts[-1] = len(v)
        for k in range(len(fracs)):
            parts[k] += v[cuts[k]:cuts[k + 1]]
    return parts


class ImgDS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        with Image.open(p) as im:
            return self.tf(im.convert("RGB")), y


def transforms():
    from torchvision import transforms as T
    norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train = T.Compose([T.RandomResizedCrop(224, scale=(0.5, 1.0)), T.RandomHorizontalFlip(),
                       T.RandomVerticalFlip(), T.ColorJitter(0.3, 0.3, 0.3, 0.05), T.ToTensor(), norm])
    test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])
    return train, test


# ---------- навчання і прогноз ----------

def train(model, items, epochs, lr, tf, dev, bs=64, workers=WORKERS):
    dl = DataLoader(ImgDS(items, tf), batch_size=bs, shuffle=True, num_workers=workers, drop_last=True,
                    pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl), pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    model.train()
    for ep in range(epochs):
        tot, n = 0.0, 0
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
                loss = lossf(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, n = tot + loss.item() * len(y), n + len(y)
        log(f"  епоха {ep + 1}/{epochs}: втрата {tot / n:.3f}")
    return model


@torch.no_grad()
def predict(model, items, tf, dev, bs=128, workers=WORKERS):
    model.eval()
    dl = DataLoader(ImgDS(items, tf), batch_size=bs, shuffle=False, num_workers=workers)
    ps, ys = [], []
    for x, y in dl:
        with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
            ps.append(torch.softmax(model(x.to(dev)).float(), 1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


# ---------- конформні множини ----------

def scores(p, y, kind):
    if kind == "lac":
        return 1 - p[np.arange(len(y)), y]
    order = np.argsort(-p, 1)
    cum = np.cumsum(np.take_along_axis(p, order, 1), 1)
    rank = np.argsort(order, 1)[np.arange(len(y)), y]
    return cum[np.arange(len(y)), rank]


def qhat(s, alpha):
    n = len(s)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(s)[k - 1])


def sets(p, q, kind):
    if kind == "lac":
        return p >= 1 - q
    order = np.argsort(-p, 1)
    cum = np.cumsum(np.take_along_axis(p, order, 1), 1)
    keep_sorted = (cum - np.take_along_axis(p, order, 1)) < q  # клас входить, поки сума до нього < q
    keep_sorted[:, 0] = True
    out = np.zeros_like(p, dtype=bool)
    np.put_along_axis(out, order, keep_sorted, 1)
    return out


def evaluate(p_cal, y_cal, p_test, y_test, alpha):
    pred = p_test.argmax(1)
    healthy = np.array([c in HEALTHY for c in CLASSES])
    r = {"acc": float((pred == y_test).mean()),
         "acc_healthy_vs_diseased": float((healthy[pred] == healthy[y_test]).mean()),
         "n_cal": int(len(y_cal)), "n_test": int(len(y_test))}
    for kind in ("lac", "aps"):
        q = qhat(scores(p_cal, y_cal, kind), alpha)
        S = sets(p_test, q, kind)
        size = S.sum(1)
        single = size == 1
        r[kind] = {"q": q, "coverage": float(S[np.arange(len(y_test)), y_test].mean()),
                   "mean_size": float(size.mean()), "frac_single": float(single.mean()),
                   "acc_single": float((pred[single] == y_test[single]).mean()) if single.any() else None,
                   "frac_empty": float((size == 0).mean())}
    return r


def new_model(arch, n_cls, dev):
    import timm
    return timm.create_model(arch, pretrained=True, num_classes=n_cls).to(dev).to(memory_format=torch.channels_last)


def run_seed(seed, pv_root, pd_items, args, dev):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf_train, tf_test = transforms()
    pv_items = [(str(p), CLASSES.index(c.name)) for c in sorted(pv_root.iterdir()) if c.is_dir()
                for p in sorted(c.glob("*.jpg"))]
    pv_tr, pv_cal, pv_te = stratified(pv_items, [0.8, 0.1, 0.1], seed)
    pd_ft, pd_cal, pd_te = stratified(pd_items, [0.5, 0.25, 0.25], seed)
    log(f"seed {seed}: PV {len(pv_tr)}/{len(pv_cal)}/{len(pv_te)}, PlantDoc {len(pd_ft)}/{len(pd_cal)}/{len(pd_te)}")
    res = {}

    log("навчання на PlantVillage")
    m = train(new_model(args.arch, len(CLASSES), dev), pv_tr, args.epochs, args.lr, tf_train, dev)
    p_pvc, y_pvc = predict(m, pv_cal, tf_test, dev)
    p_pvt, y_pvt = predict(m, pv_te, tf_test, dev)
    p_pdc, y_pdc = predict(m, pd_cal, tf_test, dev)
    p_pdt, y_pdt = predict(m, pd_te, tf_test, dev)
    res["S0_pv_pv_pv"] = evaluate(p_pvc, y_pvc, p_pvt, y_pvt, args.alpha)
    res["S1_pv_pv_field"] = evaluate(p_pvc, y_pvc, p_pdt, y_pdt, args.alpha)
    res["S2_pv_field_field"] = evaluate(p_pdc, y_pdc, p_pdt, y_pdt, args.alpha)

    log("донавчання на PlantDoc")
    m = train(m, pd_ft, args.ft_epochs, args.ft_lr, tf_train, dev, bs=32)
    p_pdc, y_pdc = predict(m, pd_cal, tf_test, dev)
    p_pdt, y_pdt = predict(m, pd_te, tf_test, dev)
    res["S3_pvft_field_field"] = evaluate(p_pdc, y_pdc, p_pdt, y_pdt, args.alpha)
    del m

    log("навчання лише на PlantDoc")
    m = train(new_model(args.arch, len(CLASSES), dev), pd_ft, args.ft_epochs, args.lr, tf_train, dev, bs=32)
    p_pdc, y_pdc = predict(m, pd_cal, tf_test, dev)
    p_pdt, y_pdt = predict(m, pd_te, tf_test, dev)
    res["S4_field_field_field"] = evaluate(p_pdc, y_pdc, p_pdt, y_pdt, args.alpha)
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()
    return res


# ---------- звіт ----------

NAMES = {"S0_pv_pv_pv": "S0: лабораторія → лабораторія",
         "S1_pv_pv_field": "S1: лабораторія, калібр. лабораторія → поле",
         "S2_pv_field_field": "S2: лабораторія, калібр. поле → поле",
         "S3_pvft_field_field": "S3: лабораторія + донавчання в полі → поле",
         "S4_field_field_field": "S4: лише польові дані → поле"}


def report(all_res, meta, alpha):
    seeds = sorted(all_res)
    ms = lambda v: f"{np.mean(v):.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"
    L = ["# Розпізнавання хвороб: перенесення з лабораторії в поле, конформні множини", "",
         f"Запусків: {len(seeds)} ({', '.join(map(str, seeds))}); α = {alpha}; класів {len(CLASSES)}. "
         f"PlantDoc після вилучення дублікатів: {meta['plantdoc']['n']} зображень "
         f"(дублікатів {meta['plantdoc']['duplicates']}, нечитаних {meta['plantdoc']['unreadable']}). "
         f"Модель: {meta['arch']}, {meta['epochs']} епох на PlantVillage, {meta['ft_epochs']} епох у полі.", "",
         "| Сценарій | Точність top-1 | Здоровий / уражений | Покриття LAC | Розмір множини LAC | "
         "Частка одноелементних | Точність на одноелементних | Покриття APS | Розмір APS |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, name in NAMES.items():
        g = lambda f: [f(all_res[s][k]) for s in seeds]
        acc_single = [v for v in g(lambda r: r["lac"]["acc_single"]) if v is not None]
        L.append(f"| {name} | {ms(g(lambda r: r['acc']))} | {ms(g(lambda r: r['acc_healthy_vs_diseased']))} | "
                 f"{ms(g(lambda r: r['lac']['coverage']))} | {ms(g(lambda r: r['lac']['mean_size']))} | "
                 f"{ms(g(lambda r: r['lac']['frac_single']))} | {ms(acc_single) if acc_single else '-'} | "
                 f"{ms(g(lambda r: r['aps']['coverage']))} | {ms(g(lambda r: r['aps']['mean_size']))} |")
    L += ["", "Дані: PlantVillage (Hughes, Salathé, arXiv:1511.08060), PlantDoc (Singh et al., 2020, CC BY 4.0). "
          "Зображення й навчені ваги не поширюються."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="/content/work")
    ap.add_argument("--out", default="results/disease_transfer")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--arch", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ft-epochs", type=int, default=10)
    ap.add_argument("--ft-lr", type=float, default=5e-5)
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("пристрій:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
    work, out = Path(args.work), Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    pv_root = prepare_plantvillage(work)
    pd_items, pd_meta = prepare_plantdoc(work)
    meta = {"arch": args.arch, "epochs": args.epochs, "ft_epochs": args.ft_epochs, "plantdoc": pd_meta,
            "pv_counts": json.loads((pv_root / ".done").read_text())}
    all_res = {}
    for s in args.seeds:
        f = out / f"seed{s}.json"
        if f.exists():
            all_res[s] = json.loads(f.read_text())
            continue
        all_res[s] = run_seed(s, pv_root, pd_items, args, dev)
        f.write_text(json.dumps(all_res[s], indent=1, ensure_ascii=False))
        rep = report(all_res, meta, args.alpha)
        (out / "report.md").write_text(rep)
        print(rep, flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    rep = report(all_res, meta, args.alpha)
    (out / "report.md").write_text(rep)
    print("\n===== ПІДСУМОК (скопіюйте все нижче) =====\n" + rep + "\n" + json.dumps(
        {str(k): v for k, v in all_res.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
