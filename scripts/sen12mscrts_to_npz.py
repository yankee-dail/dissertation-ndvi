#!/usr/bin/env python3
"""
Експортер офіційного датасету SEN12MS-CR-TS (Ebel et al., 2022) у формат .npz,
очікуваний eval_protocol.py.

Реальна структура датасету (підтверджено кодом офіційного завантажувача
github.com/PatrickTUM/SEN12MS-CR-TS, data/dataLoader.py):

    <root>/ROIs<seed>/<roi>/S2/<tdx>/*.tif     # <tdx> в 0..29 - індекс часової точки
    <root>/ROIs<seed>/<roi>/S1/<tdx>/*.tif     # той самий патч, той самий порядок файлів

Ім'я .tif містить дату як 6-й токен через "_", напр. ..._2018-06-21.tif
(парситься як img.split('/')[-1].split('_')[5] у офіційному коді).

Офіційний розподіл ROI на train/val/test (region="all") захардкоджено нижче,
скопійовано з data/dataLoader.py. Завантаження: дивись README.md цього проєкту.

Хмарність НЕ зберігається в датасеті - вона обчислюється на льоту детектором.
Тут за замовчуванням використано простий поріг яскравості (грубе наближення,
достатнє для прототипування конвеєра). Для точнішої маски встановіть пакет
s2cloudless і передайте --cloud-detector s2cloudless.

Приклад:
    python sen12mscrts_to_npz.py --root /path/to/SEN12MSCRTS --split test \
        --out ../data/sen12mscrts_test --max-patches-per-roi 5
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import rasterio
except ImportError:
    sys.exit(
        "Потрібен пакет rasterio для читання .tif: pip install rasterio\n"
        "(у venv цього проєкту: source .venv/bin/activate && pip install rasterio)"
    )

S1_LAUNCH = datetime(2014, 4, 3)

# --- офіційний розподіл ROI, скопійовано з data/dataLoader.py (region='all') ---
ROI = {
    "ROIs1158": ["106"],
    "ROIs1868": ["17", "36", "56", "73", "85", "100", "114", "119", "121", "126", "127", "139", "142", "143"],
    "ROIs1970": ["20", "21", "35", "40", "57", "65", "71", "82", "83", "91", "112", "116", "119", "128", "132",
                 "133", "135", "139", "142", "144", "149"],
    "ROIs2017": ["8", "22", "25", "32", "49", "61", "63", "69", "75", "103", "108", "115", "116", "117", "130",
                 "140", "146"],
}
_ALL_ROI = [(seed, roi) for seed, rois in ROI.items() for roi in rois]
SPLITS = {
    "test": [("ROIs1868", "119"), ("ROIs1970", "139"), ("ROIs2017", "108"), ("ROIs2017", "63"),
              ("ROIs1158", "106"), ("ROIs1868", "73"), ("ROIs2017", "32"), ("ROIs1868", "100"),
              ("ROIs1970", "132"), ("ROIs2017", "103"), ("ROIs1868", "142"), ("ROIs1970", "20"),
              ("ROIs2017", "140")],
    "val": [("ROIs2017", "22"), ("ROIs1970", "65"), ("ROIs2017", "117"), ("ROIs1868", "127"), ("ROIs1868", "17")],
}
SPLITS["train"] = [p for p in _ALL_ROI if p not in SPLITS["test"] and p not in SPLITS["val"]]
SPLITS["all"] = SPLITS["train"] + SPLITS["val"] + SPLITS["test"]

N_TIME_POINTS = 30
REFL_MIN, REFL_MAX = 0.0, 10000.0  # діапазон відбиття Sentinel-2, як у офіційному process_MS


def parse_date(filename: str) -> datetime:
    token = filename.split("_")[5]
    return datetime.strptime(token, "%Y-%m-%d")


def simple_cloud_prob(img):
    """Грубий детектор хмар за яскравістю. img: (C, H, W), сире відбиття [0, 10000].

    Хмари зазвичай яскраві в усіх каналах одночасно; це НЕ замінює s2cloudless,
    лише дає робочий прототип конвеєра.
    """
    bright = np.clip(img, REFL_MIN, REFL_MAX).mean(axis=0) / REFL_MAX  # (H, W) в [0,1]
    lo, hi = 0.25, 0.45
    return np.clip((bright - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def s2cloudless_prob(img):
    """img: (C, H, W) з 13 каналами Sentinel-2 L1C у порядку, очікуваному s2cloudless."""
    from s2cloudless import S2PixelCloudDetector  # опційна залежність

    detector = S2PixelCloudDetector(threshold=0.4, all_bands=True, average_over=4, dilation_size=2)
    moved = np.moveaxis(np.clip(img, REFL_MIN, REFL_MAX) / REFL_MAX, 0, -1)[None, ...]
    return detector.get_cloud_probability_maps(moved)[0].astype(np.float32)


def read_tif(path):
    with rasterio.open(path) as tif:
        return tif.read().astype(np.float32)


def list_patch_series(root: Path, roi_dir: str, roi: str, modality: str):
    """Повертає list[list[Path]]: [tdx][patch_idx] -> шлях до .tif для всіх 30 часових точок."""
    series = []
    for tdx in range(N_TIME_POINTS):
        d = root / roi_dir / roi / modality / str(tdx)
        if not d.is_dir():
            raise FileNotFoundError(f"Відсутня директорія часової точки: {d}")
        files = sorted(f for f in d.iterdir() if f.suffix == ".tif")
        series.append(files)
    lengths = {len(f) for f in series}
    if len(lengths) != 1:
        raise ValueError(f"{roi_dir}/{roi}/{modality}: різна кількість патчів між часовими точками: {lengths}")
    return series


def pick_target_idx(frac, rng, max_cloud, mode):
    """Обирає індекс кадру-цілі.

    mode='argmin': ціль = найменш хмарний кадр (як в офіційному sample_type='cloudy_cloudfree').
        Це робить 'least_cloudy_raw' тавтологічно ідеальним - ціль за визначенням є
        найкращим кандидатом. Залишено лише для відтворення цього відомого ефекту.
    mode='random_clean': ціль - випадковий кадр із хмарністю <= max_cloud, незалежний від
        того, чи він глобально найчистіший. Це реалістичніший сценарій ("потрібна дата",
        а не "найчистіша дата") і дає змістовний, не завжди 100%, показник ризику витоку.
    """
    if mode == "argmin":
        return int(np.argmin(frac))
    candidates = np.where(frac <= max_cloud)[0]
    if len(candidates) == 0:
        return int(np.argmin(frac))
    return int(rng.choice(candidates))


def export_roi(root, roi_dir, roi, out_dir, split_name, cloud_fn, max_patches, bands,
                target_mode="random_clean", target_max_cloud=0.3, seed=0):
    s2_series = list_patch_series(root, roi_dir, roi, "S2")
    n_patches = len(s2_series[0])
    if max_patches:
        n_patches = min(n_patches, max_patches)

    rng = np.random.default_rng(seed)
    written = 0
    for pdx in range(n_patches):
        paths = [s2_series[tdx][pdx] for tdx in range(N_TIME_POINTS)]
        frames = [read_tif(p) for p in paths]
        if bands:
            frames = [f[bands] for f in frames]
        s2 = np.stack(frames).astype(np.float32)  # (T, C, H, W), сире відбиття
        cloud = np.stack([cloud_fn(f) for f in frames]).astype(np.float32)  # (T, H, W)
        t_days = np.array([(parse_date(p.name) - S1_LAUNCH).days for p in paths], dtype=np.int64)

        frac = cloud.reshape(cloud.shape[0], -1).mean(axis=1)
        target_idx = pick_target_idx(frac, rng, target_max_cloud, target_mode)
        target = s2[target_idx].copy()

        # нормалізація відбиття до [0,1], як у офіційному process_MS(method='default')
        s2_norm = np.clip(s2, REFL_MIN, REFL_MAX) / REFL_MAX
        target_norm = np.clip(target, REFL_MIN, REFL_MAX) / REFL_MAX

        out_path = Path(out_dir) / f"{roi_dir}_{roi}_p{pdx:04d}.npz"
        np.savez_compressed(
            out_path,
            s2=s2_norm.astype(np.float32),
            cloud=cloud,
            target=target_norm.astype(np.float32),
            t_days=t_days,
            t_target=np.int64(t_days[target_idx]),
            target_idx=np.int64(target_idx),
            roi=np.array(f"{roi_dir}/{roi}"),
            split=np.array(split_name),
        )
        written += 1
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="коренева директорія злитого датасету SEN12MS-CR-TS")
    ap.add_argument("--out", required=True, help="куди писати .npz")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--max-patches-per-roi", type=int, default=None, help="обмеження для швидкого тесту")
    ap.add_argument("--cloud-detector", default="simple", choices=["simple", "s2cloudless"])
    ap.add_argument("--bands", default=None, help="кома-список індексів каналів для збереження, напр. 1,2,3,7")
    ap.add_argument("--target-mode", default="random_clean", choices=["argmin", "random_clean"],
                     help="argmin = ціль завжди найчистіша (тавтологічно ідеальна для least_cloudy_raw); "
                          "random_clean = ціль - випадковий достатньо чистий кадр, незалежний від глобального мінімуму")
    ap.add_argument("--target-max-cloud", type=float, default=0.3,
                     help="максимальна хмарність кадру, який можна обрати ціллю в режимі random_clean")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cloud_fn = s2cloudless_prob if args.cloud_detector == "s2cloudless" else simple_cloud_prob
    bands = [int(x) for x in args.bands.split(",")] if args.bands else None

    total = 0
    for roi_dir, roi in SPLITS[args.split]:
        roi_path = root / roi_dir / roi
        if not roi_path.is_dir():
            print(f"[warn] пропущено відсутній ROI {roi_dir}/{roi}", file=sys.stderr)
            continue
        n = export_roi(root, roi_dir, roi, out_dir, args.split, cloud_fn, args.max_patches_per_roi, bands,
                        target_mode=args.target_mode, target_max_cloud=args.target_max_cloud, seed=args.seed)
        print(f"{roi_dir}/{roi}: експортовано {n} патчів")
        total += n

    print(f"\nУсього: {total} .npz файлів у {out_dir}")


if __name__ == "__main__":
    main()
