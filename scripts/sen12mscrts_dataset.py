"""
PyTorch Dataset, який читає SEN12MS-CR-TS напряму з розпакованої директорії (без
проміжного експорту в .npz - це важливо при обмеженому дисковому просторі: .npz з
повним 30-кадровим рядом на патч дублює дані, тут читання йде напряму з .tif).

Очікує структуру (справжню, підтверджену на реальному завантаженому зрізі):
    <root>/<roi_dir>/<roi>/S2/<tdx>/*.tif,  tdx в 0..29

Для кожного патча:
    - ціль (target_idx) обирається режимом 'random_clean' (див. sen12mscrts_to_npz.py) -
      випадковий достатньо чистий кадр, НЕЗАЛЕЖНИЙ від глобального мінімуму хмарності,
      щоб уникнути тавтології "ціль = найчистіший кадр".
    - y_ref - найчистіший кадр СЕРЕД ПОПЕРЕДНІХ у часі (причинний базовий рівень:
      "останнє відоме чисте спостереження"), або, якщо такого немає, найчистіший
      серед усіх кадрів окрім цілі.
    - x_near - кадр, найближчий у часі до цілі (окрім ref і самої цілі) - умовний
      вхід, що несе найсвіжішу (можливо, хмарну) інформацію про поточний стан поля.

Повертається (x, y_ref, target), де x = concat[y_ref, x_near, cloud(x_near), day_gap].
Модель передбачає залишок: ŷ = y_ref + Δθ(x).
"""
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

REFL_MAX = 10000.0


def read_tif(path):
    with rasterio.open(path) as tif:
        return tif.read().astype(np.float32)


def read_tif_decimated(path, out_hw=32):
    """Швидке декантоване читання лише для оцінки хмарності (не для фінального зображення)."""
    with rasterio.open(path) as tif:
        return tif.read(out_shape=(tif.count, out_hw, out_hw)).astype(np.float32)


def simple_cloud_prob(img):
    bright = np.clip(img, 0, REFL_MAX).mean(axis=0) / REFL_MAX
    lo, hi = 0.25, 0.45
    return np.clip((bright - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def pick_target_idx(frac, rng, max_cloud, mode="random_clean"):
    if mode == "argmin":
        return int(np.argmin(frac))
    candidates = np.where(frac <= max_cloud)[0]
    if len(candidates) == 0:
        return int(np.argmin(frac))
    return int(rng.choice(candidates))


class SEN12MSCRTSPatchDataset(Dataset):
    def __init__(self, root, roi_dir, roi, patch_indices, n_time_points=30,
                 target_mode="random_clean", target_max_cloud=0.3, seed=0, img_size=None):
        self.root = Path(root)
        self.roi_dir, self.roi = roi_dir, roi
        self.patch_indices = list(patch_indices)
        self.n_t = n_time_points
        self.target_mode = target_mode
        self.target_max_cloud = target_max_cloud
        self.seed = seed
        self.img_size = img_size

        self.series = []  # [tdx] -> sorted list of patch file paths
        for tdx in range(n_time_points):
            d = self.root / roi_dir / roi / "S2" / str(tdx)
            files = sorted(f for f in d.iterdir() if f.suffix == ".tif")
            self.series.append(files)
        lengths = {len(f) for f in self.series}
        if len(lengths) != 1:
            raise ValueError(f"різна кількість патчів між часовими точками: {lengths}")

        # Одноразовий прохід: обираємо (target_idx, ref_idx, near_idx) на декантованих
        # (швидких) версіях зображень, щоб потім на кожній епосі читати лише 3 повнорозмірні
        # .tif на семпл замість усіх 30 (10x менше дискового І/О на епоху).
        self._picks = {}
        for pdx in self.patch_indices:
            self._picks[pdx] = self._precompute_picks(pdx)

    def _precompute_picks(self, pdx):
        paths = [self.series[tdx][pdx] for tdx in range(self.n_t)]
        small = [read_tif_decimated(p) for p in paths]
        frac = np.array([float(simple_cloud_prob(f).mean()) for f in small])
        t_days = np.array([self._parse_day(p.name) for p in paths], dtype=np.int64)

        rng = np.random.default_rng(self.seed + pdx)
        target_idx = pick_target_idx(frac, rng, self.target_max_cloud, self.target_mode)

        before = [t for t in range(self.n_t) if t_days[t] < t_days[target_idx]]
        if before:
            ref_idx = before[int(np.argmin(frac[before]))]
        else:
            others = [t for t in range(self.n_t) if t != target_idx]
            ref_idx = others[int(np.argmin(frac[others]))]

        rest = [t for t in range(self.n_t) if t not in (target_idx, ref_idx)]
        near_idx = rest[int(np.argmin(np.abs(t_days[rest] - t_days[target_idx])))]

        return {"target_idx": target_idx, "ref_idx": ref_idx, "near_idx": near_idx, "t_days": t_days}

    def __len__(self):
        return len(self.patch_indices)

    def _maybe_resize(self, arr):
        # arr: (C, H, W). Прості strided-кроки замість інтерполяції - досить для прототипу швидкості.
        if self.img_size is None:
            return arr
        h, w = arr.shape[-2:]
        step_h, step_w = h // self.img_size, w // self.img_size
        return arr[:, ::step_h, ::step_w][:, : self.img_size, : self.img_size]

    def __getitem__(self, i):
        pdx = self.patch_indices[i]
        pick = self._picks[pdx]
        target_idx, ref_idx, near_idx, t_days = pick["target_idx"], pick["ref_idx"], pick["near_idx"], pick["t_days"]

        y_ref_raw = read_tif(self.series[ref_idx][pdx])
        x_near_raw = read_tif(self.series[near_idx][pdx])
        target_raw = read_tif(self.series[target_idx][pdx])

        y_ref = self._maybe_resize((np.clip(y_ref_raw, 0, REFL_MAX) / REFL_MAX).astype(np.float32))
        x_near_norm = (np.clip(x_near_raw, 0, REFL_MAX) / REFL_MAX).astype(np.float32)
        x_near = self._maybe_resize(x_near_norm)
        cloud_near = self._maybe_resize(simple_cloud_prob(x_near_raw)[None].astype(np.float32))
        target = self._maybe_resize((np.clip(target_raw, 0, REFL_MAX) / REFL_MAX).astype(np.float32))

        day_gap = np.full((1,) + y_ref.shape[1:], (t_days[near_idx] - t_days[target_idx]) / 365.0,
                           dtype=np.float32)
        x = np.concatenate([y_ref, x_near, cloud_near, day_gap], axis=0)

        # Хмарність ЦІЛІ (а не входу): target_max_cloud - поріг на СЕРЕДНЮ хмарність усього
        # патча, тож окремі пікселі цілі можуть лишатись хмарними. Повертаємо карту, щоб
        # метрики могли виключити ці пікселі замість порівнювати прогноз із хмарою.
        target_cloud = self._maybe_resize(simple_cloud_prob(target_raw)[None].astype(np.float32))[0]

        return torch.from_numpy(x), torch.from_numpy(y_ref), torch.from_numpy(target), torch.from_numpy(target_cloud)

    @staticmethod
    def _parse_day(filename):
        from datetime import datetime
        token = filename.split("_")[5]
        d = datetime.strptime(token, "%Y-%m-%d")
        return (d - datetime(2014, 4, 3)).days


def n_bands():
    return 13


def n_input_channels():
    return 13 + 13 + 1 + 1  # y_ref + x_near + cloud_near + day_gap
